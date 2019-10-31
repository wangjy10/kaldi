#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/differentiable-decoder.h"
#include "decoder/decodable-matrix.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "base/timer.h"

namespace kaldi {

fst::Fst<fst::StdArc> *ReadNetwork(std::string filename) {
  // read decoding network FST
  Input ki(filename); // use ki.Stream() instead of is.
  if (!ki.Stream().good()) KALDI_ERR << "Could not open decoding-graph FST "
                                      << filename;
  fst::FstHeader hdr;
  if (!hdr.Read(ki.Stream(), "<unknown>")) {
    KALDI_ERR << "Reading FST: error reading FST header.";
  }
  if (hdr.ArcType() != fst::StdArc::Type()) {
    KALDI_ERR << "FST with arc type " << hdr.ArcType() << " not supported.";
  }
  fst::FstReadOptions ropts("<unspecified>", &hdr);

  fst::Fst<fst::StdArc> *decode_fst = NULL;

  if (hdr.FstType() == "vector") {
    decode_fst = fst::VectorFst<fst::StdArc>::Read(ki.Stream(), ropts);
  } else if (hdr.FstType() == "const") {
    decode_fst = fst::ConstFst<fst::StdArc>::Read(ki.Stream(), ropts);
  } else {
    KALDI_ERR << "Reading FST: unsupported FST type: " << hdr.FstType();
  }
  if (decode_fst == NULL) { // fst code will warn.
    KALDI_ERR << "Error reading FST (after reading header).";
    return NULL;
  } else {
    return decode_fst;
  }
}

}


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage = "";
    ParseOptions po(usage);
    BaseFloat acoustic_scale = 0.1;

    std::string word_syms_filename;
    DifferentiableDecoderOptions decoder_opts;
    decoder_opts.Register(&po);
    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for words [for debug output]");
    po.Read(argc, argv);

    std::string model_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        words_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    Int32VectorWriter words_writer(words_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") {
      word_syms = fst::SymbolTable::ReadText(word_syms_filename);
      if (!word_syms)
        KALDI_ERR << "Could not read symbol table from file "<<word_syms_filename;
    }

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    // It's important that we initialize decode_fst after loglikes_reader, as it
    // can prevent crashes on systems installed without enough virtual memory.
    // It has to do with what happens on UNIX systems if you call fork() on a
    // large process: the page-table entries are duplicated, which requires a
    // lot of virtual memory.
    fst::Fst<fst::StdArc> *decode_fst = ReadNetwork(fst_rxfilename);

    kaldi::int64 frame_count = 0;
    DifferentiableDecoder decoder(decoder_opts, *decode_fst);

    Timer timer;

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      Matrix<BaseFloat> features (feature_reader.Value());
      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << key;
        continue;
      }

      DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features, acoustic_scale);
      decoder.Decode(&gmm_decodable);

      std::vector<int32> words;
      decoder.GetBestPath(words, &gmm_decodable);
      words_writer.Write(key, words);
      if (word_syms != NULL) {
				std::cerr << key << ' ' << words.size();
				for (size_t i = 0; i < words.size(); i++) {
					std::string s = word_syms->Find(words[i]);
					if (s == "")
						KALDI_ERR << "Word-id " << words[i] <<" not in symbol table.";
					std::cerr << s << ' ';
				}
				std::cerr << '\n';
			}
      frame_count += features.NumRows();

      break;
    }

    double elapsed = timer.Elapsed();

    KALDI_LOG << "Time taken [excluding initialization] "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);

    delete word_syms;
    delete decode_fst;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
