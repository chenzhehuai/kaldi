// fstext/ext-print.h

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

#ifndef KALDI_EXT_PRINT_H_
#define KALDI_EXT_PRINT_H_

#include "fst/script/print-impl.h"


namespace fst {

template <class Arc>
class ExtPrint : public FstPrinter<Arc> {
public:
 ExtPrint(const Fst<Arc> &fst, const SymbolTable *isyms,
             const SymbolTable *osyms, const SymbolTable *ssyms, bool accep,
             bool show_weight_one, const string &field_separator,
             const string &missing_symbol = "") : 
          FstPrinter<Arc>(fst, isyms, osyms, ssyms, accep, show_weight_one, field_separator, missing_symbol) {}

 void Print(std::ostream *ostrm, const string &dest) {
    *ostrm << "from inherited class\n";
    FstPrinter<Arc>::Print(ostrm, dest);
    *ostrm << "end of from inherited class\n";
 }

};

} // end namespace fst
#endif
