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

/*
struct ExtPrintOptions {
  float table_ratio;  // we construct the table if it would be at least this full.
  int min_table_size;
  ExtPrintOptions(): table_ratio(0.25), min_table_size(4) { }
};
*/

template <class Arc>
class ExtPrint : public FstPrinter {
 public:

 private:

 void Print(std::ostream *ostrm, const string &dest) {
    *ostrm_ << "UNFINI\n";
 }

};

/*
struct TableComposeOptions: public ExtPrintOptions {
  bool connect;  // Connect output
  ComposeFilter filter_type;  // Which pre-defined filter to use
  MatchType table_match_type;

  explicit TableComposeOptions(const ExtPrintOptions &mo,
                               bool c = true, ComposeFilter ft = SEQUENCE_FILTER,
                               MatchType tms = MATCH_OUTPUT)
      : ExtPrintOptions(mo), connect(c), filter_type(ft), table_match_type(tms) { }
  TableComposeOptions() : connect(true), filter_type(SEQUENCE_FILTER),
                          table_match_type(MATCH_OUTPUT) { }
};


template<class Arc>
void TableCompose(const Fst<Arc> &ifst1, const Fst<Arc> &ifst2,
                  MutableFst<Arc> *ofst,
                  const TableComposeOptions &opts = TableComposeOptions()) {
  typedef Fst<Arc> F;
  CacheOptions nopts;
  nopts.gc_limit = 0;  // Cache only the last state for fastest copy.
  if (opts.table_match_type == MATCH_OUTPUT) {
    // ComposeFstImplOptions templated on matcher for fst1, matcher for fst2.
    ComposeFstImplOptions<ExtPrint<F>, SortedMatcher<F> > impl_opts(nopts);
    impl_opts.matcher1 = new ExtPrint<F>(ifst1, MATCH_OUTPUT, opts);
    *ofst = ComposeFst<Arc>(ifst1, ifst2, impl_opts);
  } else {
    assert(opts.table_match_type == MATCH_INPUT) ;
    // ComposeFstImplOptions templated on matcher for fst1, matcher for fst2.
    ComposeFstImplOptions<SortedMatcher<F>, ExtPrint<F> > impl_opts(nopts);
    impl_opts.matcher2 = new ExtPrint<F>(ifst2, MATCH_INPUT, opts);
    *ofst = ComposeFst<Arc>(ifst1, ifst2, impl_opts);
  }
  if (opts.connect) Connect(ofst);
}

*/


} // end namespace fst
#endif
