#ifndef DECODER_DIFFERENTIABLE_DECODER_H_
#define DECODER_DIFFERENTIABLE_DECODER_H_

#include "util/stl-utils.h"
#include "fst/fstlib.h"
#include "itf/options-itf.h"
#include "itf/decodable-itf.h"

namespace kaldi {

struct pair_hash {
  template <class T1, class T2>
  std::size_t operator() (const std::pair<T1, T2> &pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

struct DifferentiableDecoderOptions {
  BaseFloat beam;
  BaseFloat learning_rate_theta;
  BaseFloat converge_delta_theta;
  BaseFloat converge_fraction_theta;
  int32 max_num_iterations;

  DifferentiableDecoderOptions();

  void Register(OptionsItf *opts);
};

class DifferentiableDecoder {
 public:
  typedef fst::StdArc StdArc;
  typedef StdArc::Weight StdWeight;
  typedef StdArc::Label Label;
  typedef StdArc::StateId StateId;

  DifferentiableDecoder(const DifferentiableDecoderOptions &config,
                        const fst::Fst<fst::StdArc> &fst);

  ~DifferentiableDecoder();

  // Decode this utterance.
  void Decode(DecodableInterface *decodable);

  bool GetBestPath(std::vector<int32> &words, DecodableInterface *decodable);

 private:

  class Token;

  // <StateId, frame> -> Token*
  typedef std::unordered_map<std::pair<StateId, int32>, Token*, pair_hash> TokenMap;
  typedef std::unordered_map<StateId, Token*>::iterator TokenIterator;

  class TokenArc {
   public:
    StdArc arc_;
    BaseFloat theta_; // theta in the original form (not log)
    BaseFloat scores_; // -(ams + lms), log prob
    Token* prev_;

    TokenArc(const StdArc &arc, Token *prev, BaseFloat acoustic_cost);
  };

  class Token {
   public:
    std::vector<TokenArc*> arcs_; // incoming arcs
    BaseFloat alpha_; // ln(alpha)
    BaseFloat denominator_; // log of Sum of exp(theta)'s
    // LogAdd of ln(derivative of P(O|M) with respect to previous alpha_)
    // and ln(derivative of prev alpha_ with respect to this alpha_).
    // LogAdd them up to get ln(derivative of P(O|M) with respect to this alpha_).
    std::vector<BaseFloat> prev_derivs_;

    int32 ref_count_;

    // The key info is used to remove token from token_map_ after token is deleted.
    StateId state_id_;
    int32 frame_id_;

    Token(StateId state_id, int32 frame_id);

    // Add an incoming arc
    void AddArc(const StdArc &arc, Token *prev, BaseFloat acoustic_cost);

    static void TokenDelete(Token *tok, TokenMap *token_map);

    void PrintTokenInfo();
  };


  void InitDecoding();

  void Reset();

  // Returns if any tokens reached the end of the file.
  bool Forward(DecodableInterface *decodable);

  // Backprop to update parameters
  void Backward();

  void ProcessEmitting(DecodableInterface *decodable);

  void ProcessNonemitting();

  int PropagateState(Token *tok, const StdArc *arc, BaseFloat acoustic_cost);

  const TokenArc& GetBestArc(const Token *tok) const;

  // For each token in toks,
  // aggregate params of incoming arcs and calculate alpha.
  // max_alpha_ will be updated as well.
  void CalculateAlphas(std::unordered_map<StateId, Token*> &tok);

  BaseFloat CalculateDerivAlphaTheta(Token *tok, int32 arc_idx,
                                     std::vector<BaseFloat> *params);

  DifferentiableDecoderOptions config_;
  const fst::Fst<fst::StdArc> &fst_;

  std::unordered_map<StateId, Token*> cur_toks_;
  std::unordered_map<StateId, Token*> prev_toks_;
  TokenMap token_map_;
  BaseFloat max_alpha_; // Max alpha of the current frame, will be used to prune toks.
  int32 num_frames_decoded_;
  bool first_round_;
  bool converged_;

  void ClearToks(std::unordered_map<StateId, Token*> &toks);
  void PruneToks(std::unordered_map<StateId, Token*> *toks);

  KALDI_DISALLOW_COPY_AND_ASSIGN(DifferentiableDecoder);
};

} // end namespace kaldi.

#endif
