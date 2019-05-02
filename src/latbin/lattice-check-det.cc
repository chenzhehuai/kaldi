// latbin/lattice-copy.cc

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
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {
  int32 CheckDeterministicLattices(std::string filename,
      SequentialCompactLatticeReader *lattice_reader,
      CompactLatticeWriter *lattice_writer,
      bool include = true, bool ignore_missing = false,
      bool sorted = false) {
    using namespace fst;
    unordered_set<std::string, StringHasher> subset;
    std::set<std::string> subset_list;

    int32 num_total = 0;
    size_t num_success = 0;
    for (; !lattice_reader->Done(); lattice_reader->Next()){
      auto& clat = lattice_reader->Value();
      int fail=0;
      for (StateIterator<CompactLattice> siter(clat); !siter.Done(); siter.Next()) {
        unordered_set<int> label_set;
        auto s = siter.Value();
        for (ArcIterator<CompactLattice> aiter(clat, s); !aiter.Done(); aiter.Next()) {
          auto &arc = aiter.Value();
          auto r = label_set.insert(arc.olabel);
          if (!r.second) { // already exist
            fail++;
            KALDI_LOG << s << " " << arc.olabel;
            break;
          }
        }
      }
      int success=clat.NumStates() - fail;
      num_total += clat.NumStates();
      num_success += success;
    }

    KALDI_LOG << " successful " << num_success << " out of " << num_total
      <<" "<< 100.0f*num_success/num_total << "\% states.";

    if (ignore_missing) return 0;

    return (num_success != 0 ? 0 : 1);
  }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage = "";

    ParseOptions po(usage);
    bool write_compact = true, ignore_missing = false;
    std::string include_rxfilename;
    std::string exclude_rxfilename;

    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1);

    RspecifierOptions opts;
    ClassifyRspecifier(lats_rspecifier, NULL, &opts);
    bool sorted = opts.sorted;

    int32 n_done = 0;

    SequentialCompactLatticeReader lattice_reader(lats_rspecifier);

    CheckDeterministicLattices(include_rxfilename,
          &lattice_reader, NULL,
          true, ignore_missing, sorted);

    KALDI_LOG << "Done copying " << n_done << " lattices.";

    if (ignore_missing) return 0;

    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
