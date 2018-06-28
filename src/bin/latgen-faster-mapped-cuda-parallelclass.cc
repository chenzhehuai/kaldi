// bin/latgen-faster-mapped-cuda-parallelclass.cc

// Copyright      2018  Zhehuai Chen

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <nvToolsExt.h>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "fstext/fstext-lib.h"
#include "cudamatrix/cu-device.h"
#include "base/timer.h"
#include "util/kaldi-thread.h"

#include "decoder/cuda-decoder-utils.h"
#include "decoder/decoder-wrappers.h"
#include "decoder/lattice-faster-decoder-cuda.h"
#include "decoder/cuda-lattice-decoder.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::Fst;
    using fst::StdArc;

    const char *usage =
      "Generate lattices, reading log-likelihoods as matrices\n"
      " (model is needed only for the integer mappings in its transition-model)\n"
      "Usage: latgen-faster-mapped-cuda [options] trans-model-in \
      (fst-in|fsts-rspecifier) loglikes-rspecifier"
      " lattice-wspecifier [ words-wspecifier [alignments-wspecifier] ]\n";
    ParseOptions po(usage);
    Timer timer;
    bool allow_partial = false;
    BaseFloat acoustic_scale = 0.1;
    CudaLatticeDecoderConfig config;
    TaskSequencerConfig sequencer_config; // has --num-threads option

    std::string word_syms_filename;
    config.Register(&po);
    sequencer_config.Register(&po);
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial,
                "If true, produce output even if end state was not reached.");

    po.Read(argc, argv);

    if (po.NumArgs() < 4 || po.NumArgs() > 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
                fst_in_str = po.GetArg(2),
                feature_rspecifier = po.GetArg(3),
                lattice_wspecifier = po.GetArg(4),
                words_wspecifier = po.GetOptArg(5),
                alignment_wspecifier = po.GetOptArg(6);

    TransitionModel trans_model;
    ReadKaldiObject(model_in_filename, &trans_model);

    int num_threads =  sequencer_config.num_threads;
    bool determinize = config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
           : lattice_writer.Open(lattice_wspecifier)))
      KALDI_ERR << "Could not open table for writing lattices: "
                << lattice_wspecifier;

    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_filename;

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;

    double elapsed = 0;

    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
      SequentialBaseFloatMatrixReader loglike_reader(feature_rspecifier);
      // GPU version of WFST
      CudaFst decode_fst_cuda;
      CuDevice::Instantiate().SelectGpuId(1);
      CuDevice::Instantiate().AllowMultithreading();
      // Input FST is just one FST, not a table of FSTs.
      Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);
      decode_fst_cuda.Initialize(*decode_fst);

      timer.Reset();
      {
        nnet0::ExamplesRepository repository;
        Mutex examples_mutex;

        DecodeUtteranceLatticeFasterClassCuda c(
                  decode_fst_cuda, config, trans_model, word_syms, 
                  acoustic_scale, determinize, allow_partial, &alignment_writer,
                  &words_writer, &compact_lattice_writer, &lattice_writer,
                  &tot_like, &frame_count, &num_success, &num_fail, NULL,
                  &repository, &examples_mutex);
        MultiThreader<DecodeUtteranceLatticeFasterClassCuda> m(num_threads, c);

        // iterate over all feature files
        nnet0::NnetExample *example;
        std::vector<nnet0::NnetExample*> examples;
        for (; !loglike_reader.Done(); loglike_reader.Next()) {
          example = new nnet0::FeatureExample(&loglike_reader);
          if (example->PrepareData(examples))
          {
            for (int i = 0; i < examples.size(); i++)
              repository.AcceptExample(examples[i]);
            if (examples[0] != example)
                      delete example;
          }
          else
            delete example;
        }
        repository.ExamplesDone();
      }
      elapsed += timer.Elapsed();

      decode_fst_cuda.Finalize();
      delete decode_fst; // delete this only after decoder goes out of scope. 
    } else { // We have different FSTs for different utterances.
      KALDI_ERR << "Unimplemented yet. ";
    }

    KALDI_LOG << "Time taken (inc. constr.&deconstr.) " << elapsed
              << "s: real-time factor (ALL) assuming 100 frames/sec is "
              << (elapsed * 100.0 / frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like / frame_count) <<
              " over "
              << frame_count << " frames.";

    delete word_syms;
    if (num_success != 0) return 0;
    else return 1;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
