// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions and classes to compute the concatenation of two FSTs.

#ifndef AFST_CONCAT_H_
#define AFST_CONCAT_H_

#include <algorithm>
#include <vector>

#include "base/kaldi-common.h"
#include <fst/mutable-fst.h>
#include "fstext/kaldi-fst-io.h"
#include <fstext/afst.h>

namespace fst {

struct AfstCombineOptions {
  bool connect;  // Connect output
  bool del_disambig_sym;  // after combination 
  int32 disambig_sym_end;
  int32 disambig_sym_start;

  AfstCombineOptions () : connect(true), del_disambig_sym(true),
      disambig_sym_end(0), disambig_sym_start(INT_MAX) {}
};

template <class Arc>
class AFSTCombine {
public:
  using Label = typename Arc::Label;
  using Weight = typename Arc::Weight;
  using StateId = typename Arc::StateId;
  using SoaEoaPair = std::pair<StateId, StateId>;
  using SoaEoaPairVec = std::vector<SoaEoaPair>;
  using DisamSymMap = unordered_map<Label, Label>;
  #define DISAMID_TO_PAIRID(d) ((d - disambig_sym_start_)/2) //disambig_sym_start_ is a even

  AFSTCombine(AfstCombineOptions opts) : afst_num_(0), hfstid_num_(0),
              disambig_sym_start_(opts.disambig_sym_start), 
              disambig_sym_end_(opts.disambig_sym_end), opts_(opts) {
    assert(disambig_sym_end_ >= disambig_sym_start_);
  }
  ~AFSTCombine() {
    if (hfst_) delete(hfst_);
    CleanupFstVec(afst_vec_);
  }

  int CombineMain() {
    MutableFst<Arc>* fst1 = hfst_;
    for ( auto it = hfstid2vecid_map_.cbegin(); 
      it != hfstid2vecid_map_.cend(); ++it ) {
      int32 vec_id =it->second;
      int32 afst_id = hfstid2afstid_vec_[vec_id];
      int32 afst_vec_id = afstid2vecid_map_[afst_id];
      int32 state_offset;
      CopyFst(hfst_, *afst_vec_[afst_vec_id], state_offset); //copy without connect
      SoaEoaPairVec& afst_pair_vec = afst_soa_eoa_pair_vec_vec_[afst_vec_id];
      SoaEoaPairVec& hfstid_pair_vec = hfstid_soa_eoa_pair_vec_vec_[vec_id];
      for (int i = 0; i < afst_pair_vec.size(); i++) {
        StateId hfst_soa_state_ = hfstid_pair_vec[i].first;
        StateId hfst_eoa_state_ = hfstid_pair_vec[i].second;
        StateId afst_soa_state_ = afst_pair_vec[i].first + state_offset;
        StateId afst_eoa_state_ = afst_pair_vec[i].second + state_offset;
        fst1->AddArc(hfst_eoa_state_, Arc(0, 0, Weight::One(), afst_soa_state_));
        fst1->AddArc(afst_eoa_state_, Arc(0, 0, Weight::One(), hfst_soa_state_));
      }
    }

    if (opts_.connect) Connect(fst1);
    return 0;
  }
  
  const void WriteCombineResult(const std::string& fst_name) const { 
    WriteFstKaldi(*hfst_, fst_name);
  }

  int InitHfst(const std::string& fst_name, const std::string& map_name) {
    if (LoadFstWithSymMap(fst_name, map_name, &hfst_, hfst_disam_map_)) return 1;
    InitHfstSoaEoaPair();
    return 0;
  }
  
  int InitSingleAfst(const std::string& fst_name, const Label& afst_id, 
    const std::string& map_name) {
    afstid2vecid_map_[afst_id] = afst_num_;
    afst_disam_map_vec_.emplace_back();
    afst_vec_.emplace_back();
    if (LoadFstWithSymMap(fst_name, map_name, &afst_vec_[afst_num_], 
                      afst_disam_map_vec_[afst_num_])) return 1;

    afst_soa_eoa_pair_vec_vec_.emplace_back();
    InitAfstSoaEoaPair(*afst_vec_[afst_num_], afst_disam_map_vec_[afst_num_],
                        afst_soa_eoa_pair_vec_vec_[afst_num_]);
    afst_num_++;
    return 0;
  }

private:
  void CopyFst(MutableFst<Arc> *fst1, const Fst<Arc>& fst2, int32& state_offset) {
    const auto props1 = fst1->Properties(kFstProperties, false);
    const auto props2 = fst2.Properties(kFstProperties, false);
    const auto start2 = fst2.Start();
    const auto numstates1 = fst1->NumStates();
    if (fst2.Properties(kExpanded, false)) {
      fst1->ReserveStates(numstates1 + CountStates(fst2));
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
      fst1->SetFinal(s1, Weight::Zero());
      fst1->ReserveArcs(s1, fst2.NumArcs(s2));
      for (ArcIterator<Fst<Arc>> aiter(fst2, s2); !aiter.Done(); aiter.Next()) {
        auto arc = aiter.Value();
        arc.nextstate += numstates1;
        fst1->AddArc(s1, arc);
      }
    }
    state_offset = numstates1;
  }  
  // NOTICE: actually, this information can be saved in previous processing
  void InitHfstSoaEoaPair() {
    const auto numstates1 = hfst_->NumStates();
    for (StateId s1 = 0; s1 < numstates1; ++s1) {
      for (MutableArcIterator<VectorFst<Arc>> aiter(hfst_, s1); !aiter.Done(); aiter.Next()) {
        auto arc = aiter.Value();
        if (hfst_disam_map_.count(arc.ilabel) == 0) continue;
        //it's disambig
        Label ilabel = hfst_disam_map_[arc.ilabel];
        uint16 hfst_id, afst_id, disam_id;
        afst::DecodeIlabel(ilabel, hfst_id, afst_id, disam_id);
        if (opts_.del_disambig_sym) {
          arc.ilabel = 0;
          aiter.SetValue(arc);
        }
        //it's WFST disambig_sym
        if (disam_id < disambig_sym_start_ || disam_id > disambig_sym_end_) continue;
        //it's HFST/AFST disambig_sym, to be connected to AFSTs
        assert(hfst_id != 0);
        if (hfstid2vecid_map_.count(hfst_id) == 0) {
          hfstid_soa_eoa_pair_vec_vec_.emplace_back();
          hfstid2afstid_vec_.emplace_back(afst_id);
          hfstid_soa_eoa_pair_vec_vec_[hfstid_num_].resize(
              DISAMID_TO_PAIRID(disambig_sym_end_) + 1);
          hfstid2vecid_map_[hfst_id] = hfstid_num_;
          hfstid_num_++;
        }
        SoaEoaPairVec& pair_vec = hfstid_soa_eoa_pair_vec_vec_[
                                  hfstid2vecid_map_[hfst_id]];
        //according to SOA or EOA, modify the pair.first or pair.second
        SoaEoaPair& pair = pair_vec[DISAMID_TO_PAIRID(disam_id)];
        if (disam_id%2 == 0) { // SOA HFST-right-E
          pair.first = arc.nextstate;
        } else { //EOA HFST-left-S
          pair.second = s1;
        }
      }
    }
  }
  // NOTICE: actually, this information can be saved in previous processing
  void InitAfstSoaEoaPair(MutableFst<Arc>& fst, DisamSymMap& map, SoaEoaPairVec& info_pair_vec) {
    info_pair_vec.resize(DISAMID_TO_PAIRID(disambig_sym_end_) + 1);
    for (StateIterator<Fst<Arc>> siter(fst); !siter.Done(); siter.Next()) {
      StateId s1 = siter.Value();
      //get which pair from map[arc.ilabel];
      //according to SOA or EOA, modify the pair.first or pair.second
      for (MutableArcIterator<MutableFst<Arc>> aiter(&fst, s1); !aiter.Done(); aiter.Next()) {
        auto arc = aiter.Value();
        if (map.count(arc.ilabel) == 0) continue; //it's normal arc
        //it's disambig
        Label disam_id = afst::GetDecodedIlabel(map[arc.ilabel]);
        if (opts_.del_disambig_sym) {
          arc.ilabel = 0;
          aiter.SetValue(arc);
        }
        //it's WFST disambig_sym
        if (disam_id < disambig_sym_start_ || disam_id > disambig_sym_end_) continue;
        SoaEoaPair& pair = info_pair_vec[DISAMID_TO_PAIRID(disam_id)];
        if (disam_id%2 == 0) { // SOA AFST-left-E
          pair.first = arc.nextstate;
        } else { //EOA AFST-right-S
          pair.second = s1;
        }
      }
    }
  }

  int LoadFstWithSymMap(const std::string& fst_name, const std::string& map_name,
                        VectorFst<Arc>** fst, DisamSymMap& map) {
    *fst = ReadFstKaldi(fst_name);
    if (*fst) {
      return 0;
    } else {
      KALDI_ERR << "read FST error in: " << fst_name;
      return 1;
    }
    std::vector<std::vector<Label>> vec;
    if (!kaldi::ReadIntegerVectorVectorSimple(map_name, &vec))
      KALDI_ERR << "Error reading label map from " << map_name;
    return 1;
    for (const auto &n : vec) {
      assert(n.size() == 2);
      map[n[0]] = n[1];
    }
    return 0;
  }
  void CleanupFstVec(std::vector<VectorFst<Arc>*> fst_vec) {
    for (const auto &n : fst_vec) if (n) delete(n);
    fst_vec.clear();
  }

  std::unordered_map<int32, int32> afstid2vecid_map_;
  std::vector<DisamSymMap> afst_disam_map_vec_;
  //each afst needs a vec of pairs
  std::vector<SoaEoaPairVec> afst_soa_eoa_pair_vec_vec_;
  std::vector<VectorFst<Arc>*> afst_vec_;
  int32 afst_num_;

  std::unordered_map<int32, int32> hfstid2vecid_map_;
  std::vector<int32> hfstid2afstid_vec_;
  //each hfstid needs a vec of pairs
  std::vector<SoaEoaPairVec> hfstid_soa_eoa_pair_vec_vec_; 
  DisamSymMap hfst_disam_map_;
  VectorFst<Arc>* hfst_;
  int32 hfstid_num_;

  int32 disambig_sym_start_;
  int32 disambig_sym_end_;

  AfstCombineOptions opts_;
};

}  // namespace fst

#endif  // FST_CONCAT_H_
