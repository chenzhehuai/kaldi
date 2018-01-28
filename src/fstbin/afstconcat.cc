// fstbin/afstconcat.cc

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)

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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"
#include "fstext/fstext-utils.h"
#include "fstext/kaldi-fst-io.h"




int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;
    

    const char *usage =
        "TODO\n"
        "\n"
        "Usage:  afstconcat (fst1-rxfilename|fst1-rspecifier) "
        "(fst2-rxfilename|fst2-rspecifier) [(out-rxfilename|out-rspecifier)]\n";

    ParseOptions po(usage);

    AfstConcatOptions opts;
    std::vector<int32>& disambig_in = opts.disambig_in;

    po.Register("connect", &opts.connect, "If true, trim FST before output.");
    po.Read(argc, argv);


    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string disambig_rxfilename = po.GetArg(1),
        fst1_in_str = po.GetArg(2),
        fst2_in_str = po.GetArg(3),
        fst_out_str = po.GetOptArg(4);


    if (disambig_rxfilename != "")
      if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_in))
        KALDI_ERR << "afstconcat: Could not read disambiguation symbols from "
                  << PrintableRxfilename(disambig_rxfilename);

    if (disambig_in.empty()) {
      KALDI_WARN << "Disambiguation symbols list is empty; this likely "
                 << "indicates an error in data preparation.";
    }

    // Note: the "table" in is_table_1 and similar variables has nothing
    // to do with the "table" in "afstconcat"; is_table_1 relates to
    // whether we are dealing with a single FST or a whole set of FSTs.
    bool is_table_1 =
        (ClassifyRspecifier(fst1_in_str, NULL, NULL) != kNoRspecifier),
        is_table_2 =
        (ClassifyRspecifier(fst2_in_str, NULL, NULL) != kNoRspecifier),
        is_table_out =
        (ClassifyWspecifier(fst_out_str, NULL, NULL, NULL) != kNoWspecifier);
    if (is_table_out != (is_table_1 || is_table_2))
      KALDI_ERR << "Incompatible combination of archives and files";
    
    if (!is_table_1 && !is_table_2) { // Only dealing with files...
      VectorFst<StdArc> *fst1 = ReadFstKaldi(fst1_in_str);
      
      VectorFst<StdArc> *fst2 = ReadFstKaldi(fst2_in_str);

      // Checks if <fst1> is olabel sorted and <fst2> is ilabel sorted.
      if (fst1->Properties(fst::kOLabelSorted, true) == 0) {
        KALDI_WARN << "The first FST is not olabel sorted.";
      }
      if (fst2->Properties(fst::kILabelSorted, true) == 0) {
        KALDI_WARN << "The second FST is not ilabel sorted.";
      }
      
      //VectorFst<StdArc> composed_fst;
      //TableCompose(*fst1, *fst2, &composed_fst, opts);
      Concat<StdArc>(*fst1, *fst2, opts);
      WriteFstKaldi(*fst1, fst_out_str);

      delete fst1;
      delete fst2;

      return 0;
    } else {
      KALDI_ERR << "The combination of tables/non-tables and match-type that you "
                << "supplied is not currently supported.  Either implement this, "
                << "ask the maintainers to implement it, or call this program "
                << "differently.";
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

