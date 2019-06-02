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
void GetCTCTransducerFact(int pdf_num,
                        std::string fst_out_filename) {
  HmmCacheType cache;
  // "cache" is an optimization that prevents GetHmmAsFsa repeating work
  // unnecessarily.
  using namespace fst;
  typedef StdArc Arc;
  typedef Arc::Weight Weight;
  typedef Arc::StateId StateId;
  typedef Arc::Label Label;

  static const int kBlankId = 1;
  std::vector<const VectorFst<Arc> *> fsts(pdf_num+1, NULL);
  // zero is eps.
  // proc blank as 1
  {
    VectorFst<Arc> *fst = new VectorFst<Arc>;
    for (int i=0; i<2;i++) fst->AddState();
    fst->SetStart(0);
    fst->SetFinal(1, Weight::One());
    fst->AddArc(0, Arc(kBlankId, kBlankId, Weight::One(), 1));
    fsts[1] = fst;
  }
  //
  // proc phone
  for (int32 j = kBlankId+1; j < pdf_num+1; j++) { 
    VectorFst<Arc> *fst = new VectorFst<Arc>;
    for (int i=0; i<3;i++) fst->AddState();
    fst->SetStart(0);
    fst->SetFinal(1, Weight::One());
    fst->SetFinal(2, Weight::One());
    fst->AddArc(0, Arc(j, j, Weight::One(), 1));
    fst->AddArc(1, Arc(j, j, Weight::One(), 1));
    fst->AddArc(1, Arc(kBlankId, kBlankId, Weight::One(), 2));
    fst->AddArc(2, Arc(kBlankId, kBlankId, Weight::One(), 2));
    fsts[j] = fst;
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
        "Make CTC transducer from transition-ids to context-dependent phones, \n"
        " without self-loops [use add-self-loops to add them]\n"
        "Usage:   make-h-transducer <ilabel-info-file> <tree-file> "
        "<transition-gmm/acoustic-model> [<H-fst-out>]\n"
        "e.g.: \n"
        " make-ctc-transducer-fact 121 > H.fst\n";
    ParseOptions po(usage);

    HTransducerConfig hcfg;
    hcfg.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() < 1 || po.NumArgs() > 2) {
      po.PrintUsage();
      exit(1);
    }

    int pdf_num = std::stoi(po.GetArg(1));
    std::string fst_out_filename;
    if (po.NumArgs() > 1) fst_out_filename = po.GetArg(2);
#if _MSC_VER
    if (fst_out_filename == "") _setmode(_fileno(stdout), _O_BINARY);
#endif

    // The work gets done here.
    GetCTCTransducerFact(pdf_num,
                       fst_out_filename);

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
