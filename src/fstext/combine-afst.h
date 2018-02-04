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


struct AfstCombineOptions {
  bool connect;  // Connect output
  bool del_disambig_sym;  // after combination 

  AfstConcatOptions() : connect(true), del_disambig_sym(true) {}
};

template <class Arc>
class AFSTCombine {
public:
  using namespace kaldi;
  using Label = typename Arc::Label;
  using Weight = typename Arc::Weight;
  using StateId = typename Arc::StateId;
  using SoaEoaPair = std::pair<StateId, StateId>;
  using SoaEoaPairVec = std::vector<SoaEoaPair>
  using DisamSymMap = unordered_map<Label, Label>;


  AFSTCombine(AfstCombineOptions opts) : opts_(opts), afst_num_(0), hfstid_num_(0),
              disambig_sym_start_(std::INT_MAX), disambig_sym_end_(0) {}

  CombineAfst() {
    // TODO:

  }
  ~CombineAfst() {
    if (hfst_) delete(hfst_);
    CleanupFstVec(afst_vec_);
  }

  int CombineMain() {
    const auto numstates1 = hfst_.NumStates();
    for (StateId s1 = 0; s1 < numstates1; ++s1) {
      for (ArcIterator<Fst<Arc>> aiter(fst, s1); !aiter.Done(); aiter.Next()) {
        auto arc = aiter.Value();
        uint16 hfst_id, afst_id, disam_id;
        DecodeIlabel(arc.ilabel, hfst_id, afst_id, disam_id);
        assert(hfst_disam_map_.count(disam_id) > 0) //it's disambig
        if (hfst_id) { //disam_id arc, to be connected to AFSTs
          if (hfstid2vecid_map_.count(hfst_id) == 0) {
            hfstid_soa_eoa_pair_vec_vec_.emplace_back();
            SoaEoaPairVec.resize(disambig_sym_end_ - disambig_sym_start_ + 1);
            hfstid2vecid_map_[hfst_id] = hfstid_num_;
            hfstid_num_++;
          }
          SoaEoaPairVec& pair_vec = hfstid_soa_eoa_pair_vec_vec_[
                                    hfstid2vecid_map_[hfst_id]];
          //according to SOA or EOA, modify the pair.first or pair.second
          SoaEoaPair& pair = pair_vec[disam_id - disambig_sym_start_];
          if (disam_id%2 == 0) { // SOA HFST-right-E
            pair.first = arc.nextstate;
          } else { //EOA HFST-left-S
            pair.second = s1;
          }
        }
      }
    }

    if (opts_.del_disambig_sym || opts_.connect) {
      assert(0);
    }
  }

  const Fst* GetHfst() const { return hfst_; }

  const void WriteCombineResult(const std::string& fst_name) const { 
    WriteFstKaldi(*hfst_, fst_name);
  }

  int InitHfst(const std::string& fst_name, const std::string& map_name) {
    LoadFstWithSymMap(fst_name, map_name, hfst_, hfst_disam_map_);
  }
  
  int InitSingleAfst(const std::string& fst_name, const Label& afst_id, 
    const std::string& map_name) {
    afstid2vecid_map_[afst_id] = afst_num_;
    afst_disam_map_vec_.emplace_back();
    afst_vec_.emplace_back();
    LoadFstWithSymMap(fst_name, map_name, afst_vec_[afst_num_], 
                      afst_disam_map_vec_[afst_num_]);

    afst_soa_eoa_pair_vec_vec_.emplace_back();
    InitAfstSoaEoaPair(afst_vec_[afst_num_], afst_disam_map_vec_[afst_num_],
                        afst_soa_eoa_pair_vec_vec_[afst_num_])
    afst_num_++;
  }

private:

  // NOTICE: actually, this information can be saved in previous processing
  void InitAfstSoaEoaPair(Fst& fst, DisamSymMap& map, SoaEoaPairVec& info_pair_vec) {
    info_pair_vec.resize(disambig_sym_end_ - disambig_sym_start_ + 1);
    const auto numstates1 = fst.NumStates();
    for (StateId s1 = 0; s1 < numstates1; ++s1) {
      for (ArcIterator<Fst<Arc>> aiter(fst, s1); !aiter.Done(); aiter.Next()) {
        auto arc = aiter.Value();
        if (map.count(arc.ilabel) > 0) {//it's disambig
          //get which pair from map[arc.ilabel];
          //according to SOA or EOA, modify the pair.first or pair.second
          Label disam_id = afst::GetDecodedIlabel(map[arc.ilabel]);
          SoaEoaPair& pair = info_pair_vec[disam_id - disambig_sym_start_];
          if (disam_id%2 == 0) { // SOA AFST-left-E
            pair.first = arc.nextstate;
          } else { //EOA AFST-right-S
            pair.second = s1;
          }
        }
      }
    }
  }

  int LoadFstWithSymMap(const std::string& fst_name, const std::string& map_name,
                        Fst*& fst, DisamSymMap& map) {
    fst = ReadFstKaldi(fst_name);
    if (fst) {
      return 0;
    } else {
      KALDI_ERR << "read FST error in: " << fst_name;
      return 1;
    }
    std::vector<std::vector<Label>> vec;
    if (!ReadIntegerVectorVectorSimple(map_name, &vec))
      KALDI_ERR << "Error reading label map from " << map_name;
    return 1;
    for (const auto &n : vec) {
      assert(n.size() == 2);
      map[n[0]] = n[1];
      //actually, this information can be saved in previous processing
      disambig_sym_start_ = std::min(disambig_sym_start_, n[1]);
      disambig_sym_end_ = std::max(disambig_sym_end_, n[1]);
    }
  }
  void CleanupFstVec(std::vector<Fst*> fst_vec) {
    for (const auto &n : fst_vec) if (n) delete(n);
    fst_vec->clear();
  }

  std::unordered_map<int32, int32> afstid2vecid_map_;
  std::vector<DisamSymMap> afst_disam_map_vec_;
  //each afst needs a vec of pairs
  std::vector<SoaEoaPairVec> afst_soa_eoa_pair_vec_vec_;
  std::vector<Fst*> afst_vec_;
  int32 afst_num_;
  int32 disambig_sym_start_;
  int32 disambig_sym_end_;

  std::unordered_map<int32, int32> hfstid2vecid_map_;
  //each hfstid needs a vec of pairs
  std::vector<SoaEoaPairVec> hfstid_soa_eoa_pair_vec_vec_; 
  DisamSymMap hfst_disam_map_;
  MutableFst<Arc>* hfst_;
  int32 hfstid_num_;

  AfstCombineOptions opts_;
}

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
            KALDI_WARN << "call eps removal before this function ";
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
