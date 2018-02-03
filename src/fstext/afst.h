#ifndef AFST_H_
#define AFST_H_

#include "fst/arc.h"

namespace afst {
  template <class W>
  struct AFSTArcTpl  {//: public fst::ArcTpl<W>
    using Label = int64;
    using StateId = int64;
    using Weight = W;
    Label ilabel;
    Label olabel;
    Weight weight;
    StateId nextstate;

    AFSTArcTpl() {}

    AFSTArcTpl(Label ilabel, Label olabel, Weight weight, StateId nextstate)
        : ilabel(ilabel),
          olabel(olabel),
          weight(std::move(weight)),
          nextstate(nextstate) {}

    static const string &Type() {
      static const string *const type =
          new std::string(Weight::Type() == "tropical" ? "standard" : Weight::Type());
      return *type;
    }
  };
  using StdArc = AFSTArcTpl<fst::TropicalWeight>;

inline void EncodeIlabel(uint64 &ilabel, const uint32 hfst_id, const uint16 afst_id, const uint16 disambig_sym) {
      ilabel = hfst_id;
      ilabel=(ilabel<<16)+afst_id;
      ilabel=(ilabel<<16)+disambig_sym;
  }
inline void DecodeIlabel(const uint64 ilabel, uint32 &hfst_id, uint16 &afst_id, uint16 &disambig_sym) {
      disambig_sym = (uint16)ilabel;
      afst_id = (uint16)(ilabel>>16);
      hfst_id = (uint32)(ilabel>>32);
      return;
  }
inline uint16 GetDecodedIlabel(const uint64 &ilabel) { return (uint16)ilabel; }

}   // namespace fst

#endif  // AFST_H_
