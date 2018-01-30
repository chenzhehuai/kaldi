// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions and classes to compute the concatenation of two FSTs.

#ifndef AFST_CONCAT_H_
#define AFST_CONCAT_H_

#include <algorithm>
#include <vector>

#include <fst/mutable-fst.h>


namespace fst {

struct AfstConcatOptions {
  bool connect;  // Connect output
  bool del_disambig_sym;  // after concatenation 
  std::vector<int32> disambig_in;

  AfstConcatOptions() : connect(true), del_disambig_sym(true) { disambig_in.resize(0); }
};


// Computes the concatenation (product) of two FSTs. If FST1 transduces string
// x to y with weight a and FST2 transduces string w to v with weight b, then
// their concatenation transduces string xw to yv with weight Times(a, b).
//
// This version modifies its MutableFst argument (in first position).
//
// Complexity:
//
//   Time: O(V1 + E1 + V2 + E2)
//   Space: O(V1 + V2 + E2)
//
// where Vi is the number of states, and Ei is the number of arcs, of the ith
// FST.
template <class Arc>
void ConcatAfst(MutableFst<Arc> *fst1, const Fst<Arc> &fst2, 
  const AfstConcatOptions &opts = AfstConcatOptions()) {
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using SoaPair = pair<StateId, Weight>;
  // Checks that the symbol table are compatible.
  if (!CompatSymbols(fst1->InputSymbols(), fst2.InputSymbols()) ||
      !CompatSymbols(fst1->OutputSymbols(), fst2.OutputSymbols())) {
    FSTERROR() << "Concat: Input/output symbol tables of 1st argument "
               << "does not match input/output symbol tables of 2nd argument";
    fst1->SetProperties(kError, kError);
    return;
  }
  const auto props1 = fst1->Properties(kFstProperties, false);
  const auto props2 = fst2.Properties(kFstProperties, false);
  const auto start1 = fst1->Start();
  const auto start2 = fst2.Start();
  if (start1 == kNoStateId) {
    if (props2 & kError) fst1->SetProperties(kError, kError);
    return;
  }
  const auto numstates1 = fst1->NumStates();
  if (fst2.Properties(kExpanded, false)) {
    fst1->ReserveStates(numstates1 + CountStates(fst2));
  }
  unordered_map<int32, SoaPair> soa_of_new_fst2;
  SoaPair default_pair;
  default_pair.first = kNoStateId;
  default_pair.second = 0;
  for (auto val : opts.disambig_in)
  {
    soa_of_new_fst2[val]=default_pair; //default state
  }
  if (start2 != kNoStateId) {
    fst1->SetProperties(ConcatProperties(props1, props2), kFstProperties);
  } else {
    KALDI_ERR << "fst2 has no Start()";
  }
  //get whole fst2
  for (StateIterator<Fst<Arc>> siter2(fst2); !siter2.Done(); siter2.Next()) {
    const auto s1 = fst1->AddState();
    const auto s2 = siter2.Value();
    fst1->SetFinal(s1, fst2.Final(s2));
    fst1->ReserveArcs(s1, fst2.NumArcs(s2));
    for (ArcIterator<Fst<Arc>> aiter(fst2, s2); !aiter.Done(); aiter.Next()) {
      auto arc = aiter.Value();
      arc.nextstate += numstates1;
      if (s2 == start2 && soa_of_new_fst2.count(arc.ilabel) > 0) {
        soa_of_new_fst2[arc.ilabel].first = arc.nextstate;
        soa_of_new_fst2[arc.ilabel].second = arc.weight;
      }
      fst1->AddArc(s1, arc);
    }
  }
  //check #SOA in fst2
  for (auto val : opts.disambig_in)
  {
    if (soa_of_new_fst2[val].first == kNoStateId) {
      KALDI_LOG << "symbol " << val << " not found in the start of fst2;"
      << "this symbol should be normal disambig symbol from LG.fst," <<
      " thus we won't concat it from fst1 to fst2";
      soa_of_new_fst2.erase(val);
    }
  }  
  
  //connect fst1 & fst2
  for (StateId s1 = 0; s1 < numstates1; ++s1) {
    //disconnect previous fst1 final state
    //fst1->SetFinal(s1, Weight::Zero());
    for (ArcIterator<Fst<Arc>> aiter(*fst1, s1); !aiter.Done(); aiter.Next()) {
      auto arc = aiter.Value();
      if (s1 != start1 && soa_of_new_fst2.count(arc.ilabel) > 0
        && arc.nextstate != soa_of_new_fst2[arc.ilabel].first) {
        Weight final_weight = fst1->Final(arc.nextstate);
        if (final_weight == Weight::Zero()) {
            final_weight = 0;
        }
        arc.nextstate = soa_of_new_fst2[arc.ilabel].first;
        arc.weight = arc.weight.Value()+soa_of_new_fst2[arc.ilabel].second.Value()+final_weight.Value();
        if (opts.del_disambig_sym)
        {
            arc.ilabel = 0; //after concatenation, delete disambig_syms_ 
        }
        fst1->AddArc(s1, arc);
      }
    }
  }
  for (StateId s1 = 0; s1 < numstates1; ++s1) {
      fst1->SetFinal(s1, Weight::Zero());
  }

  if (opts.connect) Connect(fst1);
}


}  // namespace fst

#endif  // FST_CONCAT_H_
