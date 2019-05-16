// fstext/rand-fst.h

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

#ifndef KALDI_FSTEXT_R_FST_H_
#define KALDI_FSTEXT_R_FST_H_

#include "util/kaldi-io.h"
#include <fst/fstlib.h>
#include "half.h"

namespace fst {
using namespace kaldi;

// A user-defined compactor for R FST.
template <class A>
class RFstCompactor {
 public:
  typedef A Arc;
  typedef typename A::Label Label;
  typedef typename A::StateId StateId;
  typedef typename A::Weight Weight;
  static const Label kMaxILabel = (1<<15);
  static const Label kMaxOLabel = (1<<19);
  static const uint kMaxStateId = (1<<30);

  struct RFstElement {
    char data_[10];
    RFstElement(uint16 first, uint64 second) {
      memcpy(data_, &first, sizeof(first));
      memcpy(data_+2, &second, sizeof(second));
    }
    RFstElement() { }
    uint16 First() const {
      uint16 ret;
      memcpy(&ret, data_, sizeof(uint16));
      return ret;
    }
    uint64 Second() const {
      uint64 ret;
      memcpy(&ret, data_+2, sizeof(uint64));
      return ret;
    }
  };
  typedef RFstElement Element;

  Element Compact(StateId s, const A &arc) const {
    auto ilabel = (arc.ilabel == kNoLabel)?kMaxILabel-1:arc.ilabel;
    auto olabel = (arc.olabel == kNoLabel)?kMaxOLabel-1:arc.olabel;
    auto nextstate = (arc.nextstate == kNoStateId)?kMaxStateId-1:arc.nextstate;

    KALDI_ASSERT(ilabel < kMaxILabel);
    KALDI_ASSERT(olabel < kMaxOLabel);
    KALDI_ASSERT(nextstate < kMaxStateId);
    uint64 pack_v = ilabel;
    pack_v = (pack_v<<19) + olabel;
    pack_v = (pack_v<<30) + nextstate;
    auto weight = half_from_float(reinterpret_cast<const uint32&>(arc.weight.Value()));
    return Element(weight, pack_v);
  }
  Arc Expand(StateId s, const Element &p, uint32 f = kArcValueFlags) const {
    KALDI_ASSERT(sizeof(p)==10);
    uint64 pack_v = p.Second();
    StateId nextstate = (pack_v & (kMaxStateId-1));
    pack_v >>= 30;
    Label olabel = (pack_v & (kMaxOLabel-1));
    pack_v >>= 19;
    Label ilabel = pack_v;
    if (ilabel == kMaxILabel-1) {
      KALDI_ASSERT(olabel == kMaxOLabel-1);
      KALDI_ASSERT(nextstate == kMaxStateId - 1);
      ilabel = kNoLabel;
      olabel = kNoLabel;
      nextstate = kNoStateId;
    }
    // auto f_weight = reinterpret_cast<float&>(weight)
    auto weight = half_to_float(p.First());
    BaseFloat f_weight;
    memcpy(&f_weight, &weight, sizeof(BaseFloat));
    return Arc(ilabel, olabel, f_weight, nextstate);
  }

  ssize_t Size() const { return -1; }

  uint64 Properties() const { return 0ULL; }

  bool Compatible(const Fst<A> &fst) const { return true; }

  static const string &Type() {
    static const string *const type = new string("r_fst_compactor");
    return *type;
  }

  bool Write(std::ostream &strm) const { 
    return true; 
  }

  static RFstCompactor *Read(std::istream &strm) {
    return new RFstCompactor;
  }
};

template <class Arc, class Unsigned = uint32>
using RFst =
    CompactFst<Arc, RFstCompactor<Arc>, Unsigned>;

using StdRFst = RFst<StdArc, uint32>;


} // end namespace fst.


#endif

