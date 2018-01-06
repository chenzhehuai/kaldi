// fstext/ext-print-test.cc

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

#include "fstext/ext-print.h"
#include "fstext/fst-test-utils.h"
#include "base/kaldi-math.h"

namespace fst{


// Don't instantiate with log semiring, as RandEquivalent may fail.
template<class Arc>  void TestPrint() {

  VectorFst<Arc> *fst1 = RandFst<Arc>();

  std::cout <<"print FST\n";
  {
    FstPrinter<Arc> fstprinter(fst1, NULL, NULL, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }
  std::cout <<"ext-print FST\n";
  {
    ExtPrint<Arc> fstprinter(fst1, NULL, NULL, NULL, false, true, "\t");
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

