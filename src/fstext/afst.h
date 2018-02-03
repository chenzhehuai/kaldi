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

#define GET_LOWER_BITS(v,n) ((1<<(n+1))-1)&v  

//hfst_id 14 //afst_id 8 //disambig_sym 10
inline void EncodeIlabel(uint32 &ilabel, const uint16 hfst_id, const uint16 afst_id, const uint16 disambig_sym) {
  ilabel = hfst_id;
  ilabel=(ilabel<<14)+afst_id;
  ilabel=(ilabel<<8)+disambig_sym;
}
inline void DecodeIlabel(const uint32 ilabel, uint16 &hfst_id, uint16 &afst_id, uint16 &disambig_sym) {
  disambig_sym = (uint16)GET_LOWER_BITS(ilabel, 10);
  afst_id = (uint16)GET_LOWER_BITS((ilabel>>10), 8);
  hfst_id = (uint16)GET_LOWER_BITS((ilabel>>18), 14);
  return;
}
inline uint16 GetDecodedIlabel(const uint32 &ilabel) { 
  return (uint16)GET_LOWER_BITS(ilabel, 10); 
}

}   // namespace fst

#endif  // AFST_H_
