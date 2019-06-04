// bin/lazylm.h

// Copyright 2018 Zhehuai Chen

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

#ifndef KALDI_LAZYLM_H_
#define KALDI_LAZYLM_H_

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "base/timer.h"

namespace fst {
using namespace kaldi;

struct LazyLMConfig {
  int otf_mode;
  int preinit_mode;
  BaseFloat preinit_para;
  std::string statetable_out_filename;
  std::string statetable_in_filename;
  int32 debug_level;

  LazyLMConfig(): 
  otf_mode(1), preinit_mode(0), preinit_para(-1),
  statetable_out_filename(""), statetable_in_filename(""),
  debug_level(0){ }
  void Register(OptionsItf *opts) {
    opts->Register("otf-mode", &otf_mode,
                "on-the-fly comp. mode.");
    opts->Register("preinit-mode", &preinit_mode,
                "preinit mode.");
    opts->Register("preinit-para", &preinit_para,
                "preinit mode.");
    opts->Register("statetable-out-filename", &statetable_out_filename,
                "statetable out filename.");
    opts->Register("statetable-in-filename", &statetable_in_filename,
                "statetable in filename.");
    opts->Register("debug-level", &debug_level,
                "debug mode.");

  }
  void Check() const {
    KALDI_ASSERT(1);
  }
};


template <typename Arc, typename FilterState, typename StateTuple =
                DefaultComposeStateTuple<typename Arc::StateId, FilterState>>
class PreinitComposeStateTable: public GenericComposeStateTable<Arc, FilterState, StateTuple> {
  using StateId = typename Arc::StateId;
  using GenericComposeStateTable<Arc, FilterState, StateTuple>::Size;
  using GenericComposeStateTable<Arc, FilterState, StateTuple>::FindEntry;
 public:
  StateId FindId(const StateTuple &entry, bool insert = true) {
    all_entry_++;
    return GenericComposeStateTable<Arc, FilterState, StateTuple>::FindId(entry);
  }

  PreinitComposeStateTable(const Fst<Arc>& ifst1, const Fst<Arc>& ifst2) : GenericComposeStateTable<Arc, FilterState, StateTuple>(ifst1, ifst2), acc_(false) { }
  ~PreinitComposeStateTable() {
    // Notably: not_acc could be larger than all_entry since some state could be found for more than once
    uint64 max_ent=std::max(all_entry_, state_counter_.size());
    KALDI_VLOG(0) << "tot_acc tot_acc/all_entry (tot_acc+not_acc)/all_entry "<< tot_acc_ << " " <<1.0f*tot_acc_/max_ent << " "
      << 1.0f*(tot_acc_+not_acc_)/max_ent;
  }
  typedef std::vector<std::pair<StateTuple, int32>> MapHolder;
  StateId FindState(const StateTuple &tuple) {
    StateId ret = GenericComposeStateTable<Arc, FilterState, StateTuple>::FindState(tuple);
    //printf("%i:%i:%i ", tuple.StateId1(), tuple.StateId2(), ret);
    if (ret>=state_counter_.size()) state_counter_.resize(ret+1, 0);
    if (acc_) {
      state_counter_[ret]++;
      tot_acc_++;
    } else not_acc_++;
    return ret;
  }
  void GetId2Entry(MapHolder& id2entry) {
    for (int32 i = 0; i<Size(); i++) 
      id2entry.emplace_back(FindEntry(i), state_counter_[i]);
    return;
  }
  void SetAcc(bool acc) { acc_ = acc; }
  uint64 tot_acc_;
  uint64 not_acc_;
  uint64 all_entry_;
 private:
  std::vector<int32> state_counter_;
  // we assume the initialization stage acc_=false; 
  // the decoding stage acc_=true
  bool acc_; 
};

template <typename Weight, typename StateTuple, typename FilterState>
void LoadStateTableFunc(std::vector<std::pair<StateTuple, int32>>& holder, std::string statetable_in_filename) {
  typedef std::vector<std::pair<StateTuple, int32>> MapHolder;
  bool binary;
  Input it(statetable_in_filename, &binary);
  std::istream &is = it.Stream();
  int32 sz;
  ReadBasicType(is, binary, &sz);
  for (int i = 0; i<sz; i++) {
    std::vector<int32> v;
    ReadIntegerVector(is, binary, &v);
    FilterState fs(
          PairFilterState<
            IntegerFilterState<signed char>,
            WeightFilterState<Weight>>(
              IntegerFilterState<signed char>(v[3]),
              WeightFilterState<Weight>(*(BaseFloat*)&v[4])),
          IntegerFilterState<int>(v[5]));
    holder.emplace_back(StateTuple(v[0], v[1], fs), v[2]);
  }
  KALDI_ASSERT(holder.size() == sz);
  KALDI_LOG<<statetable_in_filename;
}
template <typename StateTuple>
void DumpStateTableFunc(std::vector<std::pair<StateTuple, int32>>& holder, std::string statetable_out_filename, bool binary) {
  typedef std::vector<std::pair<StateTuple, int32>> MapHolder;
  Output ot(statetable_out_filename, binary);
  std::ostream& os = ot.Stream();
  int32 sz = holder.size();
  WriteBasicType(os, binary, sz);
  for (const auto &i:holder) {
    // dump id_pair & acc
    const auto &fs = i.first.GetFilterState();
    std::vector<int32> v = {
      (int32)i.first.StateId1(), 
      (int32)i.first.StateId2(), 
      (int32)i.second,
      (int32)fs.GetState1().GetState1().GetState(),
      *(int32*)&fs.GetState1().GetState2().GetWeight().Value(), // TODO: keep BaseFloat
      (int32)fs.GetState2().GetState() 
    };
    if (v[2]<0) v[2] = std::numeric_limits<int32>::max(); // overflow
    WriteIntegerVector(os, binary, v);
    //printf("%li ", i.second);
  }
  KALDI_LOG<<statetable_out_filename;
}
template <class A, class CacheStore = DefaultCacheStore<A> >
class LazyLMComposeFst {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using Store = CacheStore;
  using State = typename CacheStore::State;

  typedef DefaultLookAhead<Arc, MATCH_OUTPUT> LA;
  using FstMatcher = typename LA::FstMatcher;
  using ComposeFilter = typename LA::ComposeFilter;
  typedef PreinitComposeStateTable<Arc, typename ComposeFilter::FilterState> LazyLMStateTable;
  typedef typename LazyLMStateTable::StateTuple StateTuple;
  typedef std::vector<std::pair<StateTuple, int32>> MapHolder;

  ComposeFst<A, CacheStore> *decode_fst_;

  LazyLMComposeFst(
    const Fst<Arc> &ifst1, const Fst<Arc> &ifst2,
    const LazyLMConfig& lazylm_config,
    const CacheOptions& cache_opts = CacheOptions()) : 
    decode_fst_(NULL), config_(lazylm_config), state_table_(NULL) {
      const int otf_mode = config_.otf_mode;
      switch (otf_mode) {
        case 3: {
              ComposeFstOptions<Arc, FstMatcher, ComposeFilter> opts(cache_opts);
              decode_fst_ = new ComposeFst<Arc, CacheStore>(ifst1, ifst2, opts);
              break;
                }
        case 13: {
              ComposeFstImplOptions<FstMatcher, FstMatcher, ComposeFilter, LazyLMStateTable> opts(cache_opts);
              KALDI_ASSERT(!state_table_);
              state_table_ = new LazyLMStateTable(ifst1, ifst2);
              opts.state_table = state_table_;
              opts.own_state_table = false; // TODO: delete it manually
              decode_fst_ = new ComposeFst<Arc, CacheStore>(ifst1, ifst2, opts);
              break;
                }
        default: {
              KALDI_WARN << "otf_mode undefined: " << otf_mode;
              decode_fst_ = new ComposeFst<Arc, CacheStore>(ifst1, ifst2, cache_opts);
              break;
                 }
      }
  }
  ~LazyLMComposeFst() {
    if (config_.statetable_out_filename != "") DumpStateTable();
    if (state_table_) delete state_table_;
    if (decode_fst_) delete decode_fst_;
  }

    void PreInitFST() {
    Fst<Arc> &decode_fst = *decode_fst_;
    int32 preinit_mode = config_.preinit_mode;
    BaseFloat preinit_para = config_.preinit_para;
    if (state_table_) state_table_->SetAcc(false); // start acc after init state_table
    switch (preinit_mode) {
      case 1:
        PreInitFSTByBFS(decode_fst, (int32)preinit_para);
        break;
      case 2:
        PreInitFSTByIter(decode_fst, (int32)preinit_para);
        break;
      case 3:
        PreInitFSTBySortedBFS(decode_fst, (int32)preinit_para);
        break;
      case 4:
        PreInitFSTByAcc(decode_fst, preinit_para);
        break;
      default:
        break;
    }
    if (state_table_) state_table_->SetAcc(true); // start acc after init state_table
    return;
  }


 private:
  LazyLMConfig config_; 
  LazyLMStateTable* state_table_;

void PreInitFSTByIter(Fst<Arc> &fst, int32 depth = -1) {
  using namespace kaldi;
  using StateId = typename Arc::StateId;
  
  Timer timer;
  std::set<StateId> visited;
  std::queue<StateId> queue;
  queue.push(fst.Start());
  int32 cnt=0, cnt2=0;
  std::unordered_map<StateId, int32> levels;
  levels[fst.Start()]=0;
  visited.insert(fst.Start());
  for (StateIterator<Fst<Arc>> siter(fst); !siter.Done(); siter.Next()) {
    cnt++;
    if (cnt > depth && depth!=-1)
      break;
    StateId state=siter.Value();
    fst::ArcIterator<Fst<Arc>> aiter(fst, state);
    for (; !aiter.Done(); aiter.Next()) {
      cnt2++;
    }
  }
  KALDI_VLOG(0) << "preinit_info: " << timer.Elapsed() << " " << depth << " "<<cnt<< " " << cnt2;
}

void PreInitFSTBySortedBFS(Fst<Arc> &fst, int32 depth = -1) {
  using namespace kaldi;
  using StateId = typename Arc::StateId;
  
  Timer timer;
  std::unordered_set<StateId> visited;
  std::vector<StateId> next_lev;
  std::vector<StateId> cur_lev;
  int32 cnt=0, cnt2=0;
  visited.insert(fst.Start());
  int32 lev_cnt = 0;
  next_lev.push_back(fst.Start());
  while (next_lev.size() && (depth<0 || lev_cnt++ < depth)) {
    std::swap(next_lev, cur_lev);
    next_lev.clear();
    for (auto state:cur_lev) {
      cnt++;
      fst::ArcIterator<Fst<Arc>> aiter(fst, state);
      for (; !aiter.Done(); aiter.Next()) {
        cnt2++;
        const auto& arc = aiter.Value();
        StateId nextstate = arc.nextstate;
        if (visited.find(nextstate) == visited.end()) {
          visited.insert(nextstate);
          next_lev.push_back(nextstate);
        }
      }
    } // for auto
    std::sort(next_lev.begin(), next_lev.end());
  }
  KALDI_VLOG(0) << "preinit_info: " << timer.Elapsed() << " " << depth << " "<<cnt<< " " << cnt2;
}

void PreInitFSTByBFS(Fst<Arc> &fst, int32 depth = -1) {
  using namespace kaldi;
  using StateId = typename Arc::StateId;
  
  Timer timer;
  std::set<StateId> visited;
  std::queue<StateId> queue;
  queue.push(fst.Start());
  int32 cnt=0, cnt2=0;
  std::unordered_map<StateId, int32> levels;
  levels[fst.Start()]=0;
  visited.insert(fst.Start());
  while (!queue.empty()) {
    cnt++;
    StateId state=queue.front();
    queue.pop();
    fst::ArcIterator<Fst<Arc>> aiter(fst, state);
    for (; !aiter.Done(); aiter.Next()) {
      cnt2++;
      const auto& arc = aiter.Value();
      StateId nextstate = arc.nextstate;
      if (visited.find(nextstate) == visited.end()) {
        int32 new_lev = levels[state] + 1;
        levels[nextstate] = new_lev;
        visited.insert(nextstate);
        if (new_lev <= depth || depth == -1) {
          queue.push(nextstate);
        }
      }
    }
  }
  KALDI_VLOG(0) << "preinit_info: " << timer.Elapsed() << " " << depth << " "<<cnt<< " " << cnt2;
}
// Notably: this func is based on how many times we call FindState, but not the real accumulation of states
void PreInitFSTByAcc(Fst<Arc>& fst, int32 thres) {
  Timer timer;

  // init state_table_
  MapHolder holder;
  LoadStateTable(holder);
  for (auto &i:holder) {
    state_table_->FindId(i.first);
  }
  double statetable_time = timer.Elapsed();
  // init decode_fst_
  int32 s=0, cnt=0, cnt2=0;
  for (auto &i:holder) {
    if (i.second >= thres) {
      cnt++;
      fst::ArcIterator<Fst<Arc>> aiter(fst, s);
      for (; !aiter.Done(); aiter.Next()) {
        cnt2++;
      }
    }
    i.second = 0; // init for next accumulation
    s++;
  }
  KALDI_VLOG(0) << "preinit_info: " << statetable_time << " " << timer.Elapsed() << " " << thres << " "<<cnt<< " " << cnt2;
}

void LoadStateTable(MapHolder& holder) {
  KALDI_ASSERT(config_.statetable_in_filename != "");
  KALDI_ASSERT(state_table_);
  LoadStateTableFunc<Weight, StateTuple, typename ComposeFilter::FilterState>(holder, config_.statetable_in_filename);
}

void DumpStateTable() {
  KALDI_ASSERT(config_.statetable_out_filename != "");
  KALDI_ASSERT(state_table_);
  MapHolder holder;
  bool binary = config_.debug_level<=0;
  state_table_->GetId2Entry(holder);
  DumpStateTableFunc<StateTuple>(holder, config_.statetable_out_filename, binary);
}

};
}  // namespace fst

#endif  // KALDI_LAZYLM_H_
