#include "decoder/differentiable-decoder.h"

namespace kaldi {

DifferentiableDecoderOptions::DifferentiableDecoderOptions()
  : beam(5),
    learning_rate_theta(0.5),
    converge_delta_theta(0.1),
    converge_fraction_theta(0.8),
    max_num_iterations(1) {} // XXX default values

void DifferentiableDecoderOptions::Register(OptionsItf *opts) {
  opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
  opts->Register("learning-rate-theta", &learning_rate_theta, "Learning rate for theta");
  opts->Register("converge-delta-theta", &converge_delta_theta,
                 "Convergence threshold for each theta");
  opts->Register("converge_fraction_theta", &converge_fraction_theta,
                 "If the fraction of number of converged theta's to number of total theta's \
                 is greater than or equal to this value, then we say the algorithm converged");
  opts->Register("max_num_iterations", &max_num_iterations,
                 "Maximum number of iterations.");
}

DifferentiableDecoder::DifferentiableDecoder(const DifferentiableDecoderOptions &config,
                        const fst::Fst<fst::StdArc> &fst) : config_(config), fst_(fst) {}

DifferentiableDecoder::~DifferentiableDecoder() {
  ClearToks(cur_toks_);
  ClearToks(prev_toks_);
}

void DifferentiableDecoder::InitDecoding() {
  ClearToks(cur_toks_);
  ClearToks(prev_toks_);
  StateId start_state = fst_.Start();
  KALDI_ASSERT(start_state != fst::kNoStateId);
  StdArc dummy_arc(0, 0, StdWeight::One(), start_state);
  Token *start_tok = new Token(start_state, -1);
  start_tok->AddArc(dummy_arc, NULL, 0.0);
  start_tok->alpha_ = 1.0;
  cur_toks_[start_state] = start_tok;
  token_map_[std::make_pair(start_state, -1)] = start_tok;
}

void DifferentiableDecoder::Reset() {
  ClearToks(cur_toks_);
  StateId start_state = fst_.Start();
  TokenMap::iterator find_iter = token_map_.find(std::make_pair(start_state, -1));
  find_iter->second->PrintTokenInfo();
  KALDI_ASSERT(find_iter != token_map_.end());
  cur_toks_[start_state] = find_iter->second;
}

void DifferentiableDecoder::Decode(DecodableInterface *decodable) {
  KALDI_LOG << "Total #frames: " << decodable->NumFramesReady();
  first_round_ = true;
  converged_ = false;
  int32 num_iterations = 0;
  while (!converged_ && num_iterations < config_.max_num_iterations) {
    KALDI_LOG << "num_iterations: " << num_iterations;
    KALDI_LOG << cur_toks_.size() << ", " << prev_toks_.size();

    if (first_round_) { // first round
      InitDecoding();
    } else {
      Reset();
    }
    num_frames_decoded_ = -1;
    ProcessNonemitting();
    CalculateAlphas(cur_toks_);
    PruneToks(&cur_toks_);
    num_frames_decoded_++;

    if (Forward(decodable)) {
      Backward();
    } else {
      break;
    }
    first_round_ = false;
    num_iterations++;
  }
}

bool DifferentiableDecoder::Forward(DecodableInterface *decodable) {
  while (!decodable->IsLastFrame(num_frames_decoded_ - 1)) {
    ClearToks(prev_toks_); // This will clear dangling toks but retain wanted ones
    cur_toks_.swap(prev_toks_);
    KALDI_LOG << "Advancing frame: " << num_frames_decoded_;
    ProcessEmitting(decodable);
    ProcessNonemitting();
    CalculateAlphas(cur_toks_);
    PruneToks(&cur_toks_);

    num_frames_decoded_++;
  }
  if (cur_toks_.empty()) {
    KALDI_VLOG(2) << "No token reached the end of the file.\n";
    return false;
  }
  return true;
}

void DifferentiableDecoder::ProcessEmitting(DecodableInterface *decodable) {
  KALDI_LOG << "# prev_toks_: " << prev_toks_.size();
  for (TokenIterator iter = prev_toks_.begin();
       iter != prev_toks_.end();
       ++iter) {
    StateId state = iter->first;
    Token *tok = iter->second;
    for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst_, state);
         !aiter.Done();
         aiter.Next()) {
      const StdArc &arc = aiter.Value();
      if (arc.ilabel != 0) { // propagate...
        BaseFloat acoustic_cost = decodable->LogLikelihood(num_frames_decoded_,
                                                           arc.ilabel);
        PropagateState(tok, &arc, acoustic_cost);
      }
    }
  }
}

void DifferentiableDecoder::ProcessNonemitting() {
  KALDI_LOG << "# cur_toks_: " << cur_toks_.size();
  std::vector<StateId> queue;
  for (TokenIterator iter = cur_toks_.begin();
       iter != cur_toks_.end();
       ++iter) {
    queue.push_back(iter->first);
  }

  while (!queue.empty()) {
    StateId state = queue.back();
    queue.pop_back();
    Token *tok = cur_toks_[state];
    KALDI_ASSERT(tok != NULL && state == tok->arcs_[0]->arc_.nextstate);
    for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst_, state);
         !aiter.Done();
         aiter.Next()) {
      const StdArc &arc = aiter.Value();
      if (arc.ilabel == 0) { // propagate nonemitting only...
        if (PropagateState(tok, &arc, 0.0) >= 0) {
          queue.push_back(arc.nextstate);

          int s = arc.nextstate;
          Token *tmp_tok = cur_toks_[s];
          KALDI_ASSERT(s == tmp_tok->arcs_[0]->arc_.nextstate);
        }
      }
    }
  }
}

int DifferentiableDecoder::PropagateState(Token *tok,
                                          const StdArc *arc,
                                          BaseFloat acoustic_cost) {
  if (first_round_) {
     TokenIterator find_iter = cur_toks_.find(arc->nextstate);
     if (find_iter == cur_toks_.end()) {
       Token *new_tok = new Token(arc->nextstate, num_frames_decoded_);
       new_tok->AddArc(*arc, tok, acoustic_cost);
       cur_toks_[arc->nextstate] = new_tok;
       std::pair<StateId, int32> tok_key(arc->nextstate, num_frames_decoded_);
       token_map_[tok_key] = new_tok;
     } else {
       find_iter->second->AddArc(*arc, tok, acoustic_cost);
     }
   } else {
     std::pair<StateId, int32> tok_key(arc->nextstate, num_frames_decoded_);
     TokenMap::iterator find_iter = token_map_.find(tok_key);
     if (find_iter == token_map_.end() || find_iter->second == NULL) {
       // pruned in previous rounds
       return -1;
     } else if (cur_toks_.find(arc->nextstate) == cur_toks_.end()) {
       cur_toks_[arc->nextstate] = find_iter->second;
     }
   }

  return 0;
}

void DifferentiableDecoder::Backward() {
  KALDI_ASSERT(!cur_toks_.empty());

  std::vector<Token*> queue;
  for (TokenIterator iter = cur_toks_.begin();
       iter != cur_toks_.end();
       ++iter) {
    iter->second->prev_derivs_.push_back(1.0);
    queue.push_back(iter->second);
  }

  int32 count_converged_theta = 0, count_total_theta = 0;

  while(!queue.empty()) {
    //KALDI_LOG << "queue size: " << queue.size();
    Token *tok = queue.back();
    queue.pop_back();

    // Calculate ln(derivative of P(O|M) with respect to tok->alpha_)
    BaseFloat deriv_alpha = 1.0;
    KALDI_ASSERT(tok->prev_derivs_.size() > 0);
    for (int32 i = 0; i < tok->prev_derivs_.size(); i++) { // states with incoming edge from tok
      deriv_alpha = LogAdd(deriv_alpha, tok->prev_derivs_[i]);
    }

    // Update theta_'s
    std::vector<BaseFloat> params;
    bool is_start_state = false;
    for (int32 i = 0; i < tok->arcs_.size(); i++) {
      TokenArc *arc = tok->arcs_[i];
      if (arc->prev_ == NULL) {
        KALDI_ASSERT(tok->arcs_.size() == 1); // must be the very first dummy arc,
                                              // otherwise there's dangling arc
        is_start_state = true;
        break;
      }
      params.push_back(-arc->theta_ + Log(tok->denominator_) +
            arc->scores_ + arc->prev_->alpha_);
    }
    if (is_start_state) { // no params to update, continue with the rest of the queue
      continue;
    }

    for (int32 i = 0; i < tok->arcs_.size(); i++) {
      TokenArc *arc = tok->arcs_[i];
      BaseFloat deriv_alpha_theta = CalculateDerivAlphaTheta(tok, i, &params); // in ln()
      BaseFloat delta_theta = LogAdd(deriv_alpha, deriv_alpha_theta); // in ln()
      arc->theta_ -= config_.learning_rate_theta * Exp(delta_theta);

      count_total_theta++;
      if (delta_theta <= config_.converge_delta_theta) {
        count_converged_theta++;
      }

      arc->prev_->prev_derivs_.push_back(
            LogAdd(arc->theta_ - Log(tok->denominator_) + arc->scores_, deriv_alpha));
      queue.push_back(arc->prev_);
    }
  }

  if ((BaseFloat)count_converged_theta / count_total_theta >= config_.converge_fraction_theta) {
    converged_ = true;
  }
}

bool DifferentiableDecoder::GetBestPath(std::vector<int32> &words,
                                        DecodableInterface *decodable) {
  if (!Forward(decodable)) {
    KALDI_VLOG(2) << "No tokens reached the end of file.";
  }
  if (cur_toks_.size() == 0) { // no output
    return false;
  }

  // Find start tok
  Token *tok = cur_toks_.begin()->second;
  for (TokenIterator iter = cur_toks_.begin();
      iter != cur_toks_.end();
      ++iter) {
    if (iter->second->alpha_ > tok->alpha_) {
      tok = iter->second;
    }
  }
  // Back trace
  words.clear();
  while (tok != NULL)  {
    if (tok->arcs_.size() == 1 && tok->arcs_[0]->prev_ == NULL) { // reach the starting node
      break;
    }
    const TokenArc &arc = GetBestArc(tok);
    if (arc.arc_.olabel != 0) {
      words.push_back(arc.arc_.olabel);
    }
    tok = arc.prev_;
  }

  return true;
}

const DifferentiableDecoder::TokenArc& DifferentiableDecoder::GetBestArc(const Token *tok) const {
  KALDI_ASSERT(tok != NULL);
  BaseFloat max_weight = -std::numeric_limits<double>::infinity();
  int32 idx = 0;
  for (int32 i = 0; i < tok->arcs_.size(); i++) {
    TokenArc *arc = tok->arcs_[i];
    KALDI_ASSERT(arc->prev_ != NULL);
    BaseFloat weight = arc->theta_ + arc->prev_->alpha_ + arc->scores_;
    if (weight > max_weight) {
      max_weight = weight;
      idx = i;
    }
  }
  return *(tok->arcs_[idx]);
}

void DifferentiableDecoder::CalculateAlphas(std::unordered_map<StateId, Token*> &toks) {
  max_alpha_ = -std::numeric_limits<double>::infinity();

  for (std::unordered_map<StateId, Token*>::iterator iter = toks.begin();
       iter != toks.end();
       ++iter) {
    Token *tok = iter->second;
    KALDI_ASSERT(tok->arcs_.size() > 0);
    tok->alpha_ = 1.0;
    tok->denominator_ = 1.0; // sum of exp(theta_)'s
    for (int32 i = 0; i < tok->arcs_.size(); i++) {
      TokenArc *token_arc = tok->arcs_[i];
      if (token_arc->prev_ == NULL) { // starting state
        KALDI_ASSERT(tok->arcs_.size() == 1);
        return;
      }
      tok->denominator_ = LogAdd(tok->denominator_, token_arc->theta_);
      tok->alpha_ = LogAdd(tok->alpha_, token_arc->theta_ + token_arc->scores_ +
                                        token_arc->prev_->alpha_);
    }
    tok->alpha_ -= tok->denominator_;
    max_alpha_ = std::max(tok->alpha_, max_alpha_);
  }
}

BaseFloat DifferentiableDecoder::CalculateDerivAlphaTheta(Token *tok, int32 arc_idx,
                                                          std::vector<BaseFloat> *params) {
  KALDI_ASSERT(tok->arcs_.size() == params->size());
  BaseFloat res = 1.0;
  for (int32 i = 0; i < tok->arcs_.size(); i++) {
    if (i == arc_idx) continue;
    res = LogAdd(res, (*params)[i]);
  }

  TokenArc *arc = tok->arcs_[arc_idx];
  return arc->theta_ - Log(tok->denominator_) +
    LogAdd(Log(1 - Exp(arc->theta_) / tok->denominator_) +
          arc->scores_ + arc->prev_->alpha_, res);
}

void DifferentiableDecoder::ClearToks(std::unordered_map<StateId, Token*> &toks) {
  for (TokenIterator iter = toks.begin();
       iter != toks.end(); ++iter) {
    Token::TokenDelete(iter->second, &token_map_);
  }
  toks.clear();
}

void DifferentiableDecoder::PruneToks(std::unordered_map<StateId, Token*> *toks) {
  if (toks->size() == 0) {
    KALDI_VLOG(2) << "No tokens to prune.\n";
    return;
  }
  std::vector<StateId> retained;
  BaseFloat cutoff = max_alpha_ - config_.beam;

  int32 old_num_toks = toks->size();
  KALDI_LOG << "max_alpha_: " << max_alpha_ << ", cutoff: " << cutoff;
  for (TokenIterator iter = toks->begin();
       iter != toks->end();
       ++iter) {
    Token *tok = iter->second;
    if (tok->alpha_ >= cutoff || tok->ref_count_ > 1) { // handels eps arcs
      retained.push_back(iter->first);
    } else {
      KALDI_ASSERT(tok->arcs_.size() > 0);
      Token::TokenDelete(tok, &token_map_);
    }
  }
  std::unordered_map<StateId, Token*> tmp;
  for (int32 i = 0; i < retained.size(); i++) {
    tmp[retained[i]] = (*toks)[retained[i]];
  }
  tmp.swap(*toks);

  KALDI_LOG << "Tokens pruned: " << old_num_toks - toks->size()
    << "," << (old_num_toks - toks->size()) / (double)old_num_toks * 100 << "%";
}

DifferentiableDecoder::Token::Token(StateId state_id, StateId frame_id) : alpha_(1.0),
  denominator_(1.0), ref_count_(1), state_id_(state_id), frame_id_(frame_id) {}

void DifferentiableDecoder::Token::AddArc(const StdArc &arc, Token *prev,
                                          BaseFloat acoustic_cost) {
  arcs_.push_back(new TokenArc(arc, prev, acoustic_cost));
}

DifferentiableDecoder::TokenArc::TokenArc(const StdArc &arc, Token *prev,
                                                 BaseFloat acoustic_cost) :
  theta_(0.5), prev_(prev) {
  arc_.ilabel = arc.ilabel;
  arc_.olabel = arc.olabel;
  arc_.weight = arc.weight;
  arc_.nextstate = arc.nextstate;
  scores_ = -acoustic_cost - arc.weight.Value();
  if (prev) {
    prev->ref_count_++;
  }
}

void DifferentiableDecoder::Token::TokenDelete(Token *tok, TokenMap *token_map) {
  while (--tok->ref_count_ == 0) { // tok is to be deleted
    for (int32 i = 0 ; i < tok->arcs_.size(); i++) {
      Token *prev = tok->arcs_[i]->prev_;
      if (prev == NULL) continue;
      Token::TokenDelete(prev, token_map);
      delete tok->arcs_[i];
    }
    std::pair<StateId, int32> tok_key(tok->state_id_, tok->frame_id_);
    KALDI_ASSERT(token_map->find(tok_key) != token_map->end());
    delete tok;
    (*token_map)[tok_key] = NULL; // remove token from token_map_
  }
}

void DifferentiableDecoder::Token::PrintTokenInfo() {
  KALDI_LOG << "arcs_.size: " << arcs_.size() << ", " << "alpha_: "  << alpha_ << ", denominator_: " << denominator_;
}

} // end namespace kaldi.
