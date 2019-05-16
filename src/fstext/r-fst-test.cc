// fstext/r-fst-test.cc

// Copyright 2009-2011  Microsoft Corporation

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


#include "fstext/r-fst.h"
#include "fstext/rand-fst.h"
#include "fstext/fstext-utils.h"
#include "base/kaldi-math.h"

namespace fst
{


// Don't instantiate with log semiring, as RandEquivalent may fail.
static void TestRFst() {
  typedef StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;

  VectorFst<Arc> *fst = RandFst<StdArc>();

  {
    FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }

  std::unique_ptr<Fst<Arc>> fst_copy(Convert(*fst, "compact_r_fst_compactor"));

  {
    FstPrinter<Arc> fstprinter(*fst_copy, NULL, NULL, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }

  fst_copy->Write("/tmp/test.fst");
  KALDI_ASSERT(RandEquivalent(*fst, *fst_copy,
                              5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));
  delete fst;
}


} // namespace fst

int main() {
  kaldi::g_kaldi_verbose_level = 4;
  using namespace fst;
  for (int i = 0; i < 25; i++) {
    TestRFst();
  }
}
