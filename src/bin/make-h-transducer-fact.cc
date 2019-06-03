// bin/make-h-transducer.cc
// Copyright 2009-2011 Microsoft Corporation

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

#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "tree/context-dep.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"
#include "fstext/table-matcher.h"
#include "fstext/fstext-utils.h"
#include "fstext/context-fst.h"

namespace kaldi {

// The H transducer has a separate outgoing arc for each of the symbols in
// ilabel_info.
void GetHTransducerFact(const std::vector<std::vector<int32> > &ilabel_info,
                        const ContextDependencyInterface &ctx_dep,
                        const TransitionModel &trans_model,
                        const HTransducerConfig &config,
                        std::vector<int32> *disambig_syms_left,
                        std::string fst_out_filename) {
  KALDI_ASSERT(ilabel_info.size() >= 1 &&
               ilabel_info[0].size() == 0); // make sure that eps == eps.
  HmmCacheType cache;
  // "cache" is an optimization that prevents GetHmmAsFsa repeating work
  // unnecessarily.
  using namespace fst;
  typedef StdArc Arc;
  typedef Arc::Weight Weight;
  typedef Arc::StateId StateId;
  typedef Arc::Label Label;

  std::vector<const VectorFst<Arc> *> fsts(ilabel_info.size(), NULL);
  std::vector<int32> phones = trans_model.GetPhones();

  KALDI_ASSERT(disambig_syms_left != 0);
  disambig_syms_left->clear();

  int32 first_disambig_sym =
      trans_model.NumTransitionIds() +
      1; // First disambig symbol we can have on the input side.
  int32 next_disambig_sym = first_disambig_sym;

  if (ilabel_info.size() > 0)
    KALDI_ASSERT(ilabel_info[0].size() == 0); // make sure epsilon is epsilon...

  for (int32 j = 1; j < static_cast<int32>(ilabel_info.size()); j++) { // zero is eps.
    KALDI_ASSERT(!ilabel_info[j].empty());
    if (ilabel_info[j].size() == 1 && ilabel_info[j][0] <= 0) { // disambig symbol

      // disambiguation symbol.
      int32 disambig_sym_left = next_disambig_sym++;
      disambig_syms_left->push_back(disambig_sym_left);
      // get acceptor with one path with "disambig_sym" on it.
      VectorFst<Arc> *fst = new VectorFst<Arc>;
      fst->AddState();
      fst->AddState();
      fst->SetStart(0);
      fst->SetFinal(1, Weight::One());
      fst->AddArc(0, Arc(disambig_sym_left, disambig_sym_left, Weight::One(), 1));
      fsts[j] = fst;
    } else { // Real phone-in-context.
      std::vector<int32> phone_window = ilabel_info[j];

      VectorFst<Arc> *fst =
          GetHmmAsFsa(phone_window, ctx_dep, trans_model, config, &cache);
      BaseFloat self_loop_scale = 1.0;
      bool reorder = true;
      AddSelfLoops(trans_model, std::vector<int32>(), self_loop_scale, reorder, true,
                   fst);
      fsts[j] = fst;
    }
  }

  TableWriter<fst::VectorFstHolder> fst_writer(fst_out_filename);
  int count = 0;
  int fst_size = 0;
  for (auto &i : fsts) {
    if (!i) {
      fst_writer.Write(std::to_string(count), VectorFst<Arc>());
      KALDI_LOG << "empty at " << count;
    } else {
      fst_writer.Write(std::to_string(count), *i);
      fst_size = i->NumStates();
    }
    count++;
  }
  KALDI_LOG << count << " in total; each with state nums: " << fst_size;

  SortAndUniq(&fsts); // remove duplicate pointers, which we will have
  // in general, since we used the cache.
  DeletePointers(&fsts);
  return;
}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Make H transducer from transition-ids to context-dependent phones, \n"
        " without self-loops [use add-self-loops to add them]\n"
        "Usage:   make-h-transducer <ilabel-info-file> <tree-file> "
        "<transition-gmm/acoustic-model> [<H-fst-out>]\n"
        "e.g.: \n"
        " make-h-transducer ilabel_info  1.tree 1.mdl > H.fst\n"
        "Another example can be referred to: "
        " egs/mini_librispeech/s5/local/mkgraph.sh";
    ParseOptions po(usage);

    HTransducerConfig hcfg;
    std::string disambig_out_filename;
    hcfg.Register(&po);
    po.Register("disambig-syms-out", &disambig_out_filename,
                "List of disambiguation symbols on input of H [to be output from "
                "this program]");

    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string ilabel_info_filename = po.GetArg(1);
    std::string tree_filename = po.GetArg(2);
    std::string model_filename = po.GetArg(3);
    std::string fst_out_filename;
    if (po.NumArgs() >= 4) fst_out_filename = po.GetArg(4);
    if (fst_out_filename == "-") fst_out_filename = "";

    std::vector<std::vector<int32> > ilabel_info;
    {
      bool binary_in;
      Input ki(ilabel_info_filename, &binary_in);
      fst::ReadILabelInfo(ki.Stream(), binary_in, &ilabel_info);
    }

    ContextDependency ctx_dep;
    ReadKaldiObject(tree_filename, &ctx_dep);

    TransitionModel trans_model;
    ReadKaldiObject(model_filename, &trans_model);

    std::vector<int32> disambig_syms_out;

#if _MSC_VER
    if (fst_out_filename == "") _setmode(_fileno(stdout), _O_BINARY);
#endif

    // The work gets done here.
    GetHTransducerFact(ilabel_info, ctx_dep, trans_model, hcfg, &disambig_syms_out,
                       fst_out_filename);
    if (disambig_out_filename != "") { // if option specified..
      if (disambig_out_filename == "-") disambig_out_filename = "";
      if (!WriteIntegerVectorSimple(disambig_out_filename, disambig_syms_out))
        KALDI_ERR << "Could not write disambiguation symbols to "
                  << (disambig_out_filename == "" ? "standard output"
                                                  : disambig_out_filename);
    }

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
