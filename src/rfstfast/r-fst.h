// rfstfast/r-fst.h

// Copyright      2019  Zhehuai Chen

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

#ifndef KALDI_FSTEXT_R_FST_FAST_H_
#define KALDI_FSTEXT_R_FST_FAST_H_

#include "util/kaldi-io.h"
#include <fst/fstlib.h>

namespace fst {
using namespace kaldi;

// A user-defined compactor for R FST.
template <class A>
class RFstFastCompactor {
 public:
  typedef A Arc;
  typedef typename A::Label Label;
  typedef typename A::StateId StateId;
  typedef typename A::Weight Weight;
  static const Label kMaxILabel = (1 << 15);
  static const Label kMaxOLabel = (1 << 19);
  static const uint kMaxStateId = (1 << 30);

#pragma pack(4)
  struct RFstElement {
    float weight;
    unsigned nextstate : 30;
    unsigned olabel : 19;
    unsigned ilabel : 15;
    RFstElement(Label ilabel, Label olabel, StateId nextstate, float weight)
        : weight(weight), nextstate(nextstate), olabel(olabel), ilabel(ilabel) {}

    RFstElement() {}
  };
  typedef RFstElement Element;

  Element Compact(StateId s, const A &arc) const {
    auto ilabel = (arc.ilabel == kNoLabel) ? kMaxILabel - 1 : arc.ilabel;
    auto olabel = (arc.olabel == kNoLabel) ? kMaxOLabel - 1 : arc.olabel;
    auto nextstate = (arc.nextstate == kNoStateId) ? kMaxStateId - 1 : arc.nextstate;

    KALDI_ASSERT(sizeof(Element) == 12);
    KALDI_ASSERT(ilabel < kMaxILabel);
    KALDI_ASSERT(olabel < kMaxOLabel);
    KALDI_ASSERT(nextstate < kMaxStateId);
    auto weight = arc.weight.Value();
    return Element(ilabel, olabel, nextstate, weight);
  }
  Arc Expand(StateId s, const Element &p, uint32 f = kArcValueFlags) const {
    StateId nextstate = p.nextstate;
    Label olabel = p.olabel;
    Label ilabel = p.ilabel;
    if (ilabel == kMaxILabel - 1) {
      ilabel = kNoLabel;
      olabel = kNoLabel;
      nextstate = kNoStateId;
    }
    auto weight = p.weight; 
    return Arc(ilabel, olabel, weight, nextstate);
  }

  ssize_t Size() const { return -1; }

  uint64 Properties() const { return 0ULL; }

  bool Compatible(const Fst<A> &fst) const { return true; }

  static const string &Type() {
    static const string *const type = new string("r_fst_fast_compactor");
    return *type;
  }

  bool Write(std::ostream &strm) const { return true; }

  static RFstFastCompactor *Read(std::istream &strm) { return new RFstFastCompactor; }
};

template <class Arc, class Unsigned = uint32>
using RFstFast = CompactFst<Arc, RFstFastCompactor<Arc>, Unsigned>;

using StdRFstFast = RFstFast<StdArc, uint32>;

} // end namespace fst.

#endif
