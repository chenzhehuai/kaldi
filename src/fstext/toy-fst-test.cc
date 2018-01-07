// fstext/toy-fst-test.cc

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

#include "fstext/toy-fst.h"
#include "fstext/ext-print.h"
#include "fstext/fst-test-utils.h"
#include "base/kaldi-math.h"

namespace fst{


// Don't instantiate with log semiring, as RandEquivalent may fail.
template<class Arc>  void TestPrint() {

  VectorFst<Arc> *fst1 = RandFst<Arc>();
  ILabelCompare<Arc> ilabel_comp;
  ArcSort(fst1, ilabel_comp);

  ToyFst< Arc, VectorState<Arc> > toyfst1(*fst1);

  std::cout <<"fst1 FST\n";
  {
    ExtPrint<Arc> fstprinter(*fst1, NULL, NULL, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  } 

  std::cout <<"toy-fst FST\n";
  {
    ExtPrint<Arc> fstprinter(toyfst1, NULL, NULL, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }  

  if (RandEquivalent(*fst1, toyfst1, 3/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 20/*path length-- max?*/))
  {
    std::cout <<"same\n";
  }
  else
  {
    std::cout <<"diff\n";
  }

  ToyFst2< Arc, VectorState<Arc> > toyfst2(*fst1);
  std::cout <<"toy-fst FST\n";
  {
    ExtPrint<Arc> fstprinter(toyfst2, NULL, NULL, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }  

  delete fst1;
}


} // namespace fst

int main() {
  using namespace fst;
  for (int i = 0;i < 1;i++) {
    TestPrint<fst::StdArc>();
  }
}


