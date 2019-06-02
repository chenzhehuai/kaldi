// decoder/lattice-incremental-fact-decoder.cc

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

#include "lattice-incremental-fact-decoder.h"
#include "base/timer.h"
#include "lat/lattice-functions.h"

namespace kaldi {

// instantiate this class once for each thing you have to decode.
template <typename FST, std::size_t kStatePerPhone, typename Token>
LatticeIncrementalFactDecoderTpl<FST, kStatePerPhone, Token>::
    LatticeIncrementalFactDecoderTpl(const LatticeIncrementalDecoderConfig &config,
                                     const TransitionModel *trans_model,
                                     const std::string fst_in_str)
    : LatticeIncrementalDecoderTpl<FST, Token>(config, trans_model) {
  arc_toks_.SetSize(1000); // just so on the first frame we do something reasonable.
  if (fst_in_str != "")
    LoadHTransducers(fst_in_str);
  else
    assert(0);
}

template <typename FST, std::size_t kStatePerPhone, typename Token>
LatticeIncrementalFactDecoderTpl<FST, kStatePerPhone, Token>::
    LatticeIncrementalFactDecoderTpl(const LatticeIncrementalDecoderConfig &config,
                                     FST *fst, const TransitionModel *trans_model,
                                     const std::string fst_in_str)
    : LatticeIncrementalDecoderTpl<FST, Token>(config, fst, trans_model) {
  arc_toks_.SetSize(1000); // just so on the first frame we do something reasonable.
  if (fst_in_str != "")
    LoadHTransducers(fst_in_str);
  else
    assert(0);
}

template <typename FST, std::size_t kStatePerPhone, typename Token>
LatticeIncrementalFactDecoderTpl<FST, kStatePerPhone,
                                 Token>::~LatticeIncrementalFactDecoderTpl() {
  DeleteElemArcs(arc_toks_.Clear());
}

template <typename FST, std::size_t kStatePerPhone, typename Token>
void LatticeIncrementalFactDecoderTpl<FST, kStatePerPhone, Token>::LoadHTransducers(
    std::string fst_in_str) {
  SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_in_str);
  int count = 0;
  h_transducers_.clear();
  for (; !fst_reader.Done(); fst_reader.Next(), count++) {
    h_transducers_.emplace_back(fst_reader.Value());
  }
  KALDI_VLOG(1) << "Total ilabels: " << count;
}

template <typename FST, std::size_t kStatePerPhone, typename Token>
void LatticeIncrementalFactDecoderTpl<FST, kStatePerPhone, Token>::InitDecoding(
    FST *fst, bool keep_context) {
  // clean up from last time:
  DeleteElems(toks_.Clear());
  cost_offsets_.clear();
  ClearActiveTokens();
  warned_ = false;
  num_toks_ = 0;
  emit_tok_num_ = 0;
  nemit_tok_num_ = 0;
  conf_ = -1;
  decoding_finalized_ = false;
  final_costs_.clear();
  active_toks_.resize(1);
  if (token_allocator_) {
    delete token_allocator_;
    token_allocator_ = NULL;
  }
  if (link_allocator_) {
    delete link_allocator_;
    link_allocator_ = NULL;
  }
  if (arc_token_allocator_) {
    delete arc_token_allocator_;
    arc_token_allocator_ = NULL;
  }
  if (fst_ == NULL) return;

  if (fst_->Properties(fst::kILabelSorted, true) == 0) {
    KALDI_VLOG(2)
        << "The FST is not ilabel sorted. "
        << "If it is a standard Fst, do the following please:\n"
        << " cat OLD.fst | fstarcsort |fstconvert  --fst_type=const > NEW.fst";
    fst_sorted_ = false;
  } else {
    fst_sorted_ = true;
  }

  token_allocator_ = new fst::PoolAllocator<Token>(1024);
  link_allocator_ = new fst::PoolAllocator<BackwardLinkT>(2048);
  arc_token_allocator_ = new fst::PoolAllocator<ArcToken>(1024);
  StateId start_state = fst_->Start();
  KALDI_ASSERT(start_state != fst::kNoStateId);
  if (!keep_context) {
    last_frame_nonfinal_states_.clear();
  }
  last_frame_nonfinal_states_.push_back(start_state);
  std::vector<Token *> last_frame_nonfinal_tokens;
  Token *start_tok = NULL;
  for (auto i : last_frame_nonfinal_states_) {
    Token *&toks = active_toks_[0].toks;
    Token *tok = token_allocator_->allocate(1);
    token_allocator_->construct(tok, 0.0, std::numeric_limits<BaseFloat>::infinity(),
                                nullptr, toks);
    toks = tok;
    toks_.Insert(i, tok);
    queue_.push_back(QueueElem(i, tok));
    num_toks_++;
    if (i == start_state)
      start_tok = tok;
    else
      last_frame_nonfinal_tokens.push_back(tok);
  }
  // connect state 0 and above states
  for (auto i : last_frame_nonfinal_tokens) {
    BackwardLinkT *t_link = i->links;
    i->links = link_allocator_->allocate(1);
    link_allocator_->construct(i->links, start_tok, nullptr, 0, t_link);
  }

  last_get_lattice_frame_ = 0;
  token_label_map_.clear();
  token_label_map_.reserve(std::min((int32)1e5, config_.max_active));
  token_label_available_idx_ = config_.max_word_id + 1;
  token_label_final_cost_.clear();
  determinizer_.Init();

  DeleteElemArcs(arc_toks_.Clear());
  last_best_arc_tok_ = NULL;
  last_cutoff_ = config_.beam; // initialize it by config_.beam

  ProcessNonemitting(config_.beam);
}

// Returns true if any kind of traceback is available (not necessarily from
// a final state).  It should only very rarely return false; this indicates
// an unusual search error.
template <typename FST, std::size_t kStatePerPhone, typename Token>
bool LatticeIncrementalFactDecoderTpl<FST, kStatePerPhone, Token>::Decode(
    DecodableInterface *decodable) {
  InitDecoding();

  // We use 1-based indexing for frames in this decoder (if you view it in
  // terms of features), but note that the decodable object uses zero-based
  // numbering, which we have to correct for when we call it.

  while (!decodable->IsLastFrame(NumFramesDecoded() - 1)) {
    if (NumFramesDecoded() % config_.prune_interval == 0) {
      PruneActiveTokens(config_.lattice_beam * config_.prune_scale);
    }

    // We always incrementally determinize the lattice after lattice pruning in
    // PruneActiveTokens() since we need extra_cost as the weights
    // of final arcs to denote the "future" information of final states (Tokens)
    // Moreover, the delay on GetLattice to do determinization
    // make it process more skinny lattices which reduces the computation overheads.
    int32 frame_det_most = NumFramesDecoded() - config_.determinize_delay;
    // The minimum length of chunk is config_.determinize_chunk_size.
    if (frame_det_most % config_.determinize_chunk_size == 0) {
      int32 frame_det_least =
          last_get_lattice_frame_ + config_.determinize_chunk_size;
      // To adaptively decide the length of chunk, we further compare the number of
      // tokens in each frame and a pre-defined threshold.
      // If the number of tokens in a certain frame is less than
      // config_.determinize_max_active, the lattice can be determinized up to this
      // frame. And we try to determinize as most frames as possible so we check
      // numbers from frame_det_most to frame_det_least
      for (int32 f = frame_det_most; f >= frame_det_least; f--) {
        if (config_.determinize_max_active == std::numeric_limits<int32>::max() ||
            GetNumToksForFrame(f) < config_.determinize_max_active) {
          KALDI_VLOG(2) << "Frame: " << NumFramesDecoded()
                        << " incremental determinization up to " << f;
          GetLattice(false, f);
          break;
        }
      }
    }

    // step 0
    active_toks_.resize(active_toks_.size() + 1);
    // step 1
    SetEntryTokenForArcTokens(decodable, last_cutoff_,
                              config_.beam); // use adaptive_beam
    // step 2
    BaseFloat cost_cutoff = ProcessEmitting(decodable);
    last_cutoff_ = cost_cutoff;

    // step 3
    ExpandArcTokensToNextState(cost_cutoff);
    int32 num;
    if (g_kaldi_verbose_level > 1) {
      num = GetNumToksForFrame(NumFramesDecoded()); // it takes time
      KALDI_VLOG(6) << "e " << num;
      emit_tok_num_ += num;
    }

    // step 4
    ProcessNonemitting(cost_cutoff);

    if (g_kaldi_verbose_level > 1) {
      num = GetNumToksForFrame(NumFramesDecoded()); // it takes time
      KALDI_VLOG(6) << "ne " << num;
      nemit_tok_num_ += num;
    }
  }
  FinalizeDecoding();

  // Returns true if we have any kind of traceback available (not necessarily
  // to the end state; query ReachedFinal() for that).
  return !active_toks_.empty() && active_toks_.back().toks != NULL;
}

template <typename FST, std::size_t kStatePerPhone, typename Token>
void LatticeIncrementalFactDecoderTpl<FST, kStatePerPhone, Token>::AdvanceDecoding(
    DecodableInterface *decodable, int32 max_num_frames) {
  KALDI_ASSERT(!active_toks_.empty() && !decoding_finalized_ &&
               "You must call InitDecoding() before AdvanceDecoding");
  int32 num_frames_ready = decodable->NumFramesReady();
  // num_frames_ready must be >= num_frames_decoded, or else
  // the number of frames ready must have decreased (which doesn't
  // make sense) or the decodable object changed between calls
  // (which isn't allowed).
  KALDI_ASSERT(num_frames_ready >= NumFramesDecoded());
  int32 target_frames_decoded = num_frames_ready;
  if (max_num_frames >= 0)
    target_frames_decoded =
        std::min(target_frames_decoded, NumFramesDecoded() + max_num_frames);
  while (NumFramesDecoded() < target_frames_decoded) {
    if (NumFramesDecoded() % config_.prune_interval == 0) {
      PruneActiveTokens(config_.lattice_beam * config_.prune_scale);
    }
    // We always incrementally determinize the lattice after lattice pruning in
    // PruneActiveTokens() since we need extra_cost as the weights
    // of final arcs to denote the "future" information of final states (Tokens)
    // Moreover, the delay on GetLattice to do determinization
    // make it process more skinny lattices which reduces the computation overheads.
    int32 frame_det_most = NumFramesDecoded() - config_.determinize_delay;
    // The minimum length of chunk is config_.determinize_chunk_size.
    if (frame_det_most % config_.determinize_chunk_size == 0) {
      int32 frame_det_least =
          last_get_lattice_frame_ + config_.determinize_chunk_size;
      // To adaptively decide the length of chunk, we further compare the number of
      // tokens in each frame and a pre-defined threshold.
      // If the number of tokens in a certain frame is less than
      // config_.determinize_max_active, the lattice can be determinized up to this
      // frame. And we try to determinize as most frames as possible so we check
      // numbers from frame_det_most to frame_det_least
      for (int32 f = frame_det_most; f >= frame_det_least; f--) {
        if (config_.determinize_max_active == std::numeric_limits<int32>::max() ||
            GetNumToksForFrame(f) < config_.determinize_max_active) {
          KALDI_VLOG(2) << "Frame: " << NumFramesDecoded()
                        << " incremental determinization up to " << f;
          GetLattice(false, f);
          break;
        }
      }
    }

    // step 0
    active_toks_.resize(active_toks_.size() + 1);
    // step 1
    SetEntryTokenForArcTokens(decodable, last_cutoff_,
                              config_.beam); // use adaptive_beam
    // step 2
    BaseFloat cost_cutoff = ProcessEmitting(decodable);
    last_cutoff_ = cost_cutoff;

    // step 3
    ExpandArcTokensToNextState(cost_cutoff);
    int32 num;
    if (g_kaldi_verbose_level > 1) {
      num = GetNumToksForFrame(NumFramesDecoded()); // it takes time
      KALDI_VLOG(6) << "e " << num;
      emit_tok_num_ += num;
    }

    // step 4
    ProcessNonemitting(cost_cutoff);

    if (g_kaldi_verbose_level > 1) {
      num = GetNumToksForFrame(NumFramesDecoded()); // it takes time
      KALDI_VLOG(6) << "ne " << num;
      nemit_tok_num_ += num;
    }
  }
}

/// Gets the weight cutoff.  Also counts the active tokens.
/// Here we take all the tokens in arc_toks_ into account. It includes:
/// i) tokens of the last frame ii) the new tokens created by
/// SetEntryTokenForArcTokens(). Notably, the latter part is belonging to the current
/// frame. Nevertheless, we does not include the acoustic and transition cost in it.
/// Hence it is comparable to the tokens of the last frame
/// We do GetCutoff() after SetEntryTokenForArcTokens() because GetCutoff() is used to
/// control the computation of ProcessEmitting() which should take tokens from
/// ArcIterator of SetEntryTokenForArcTokens() into account
template <typename FST, std::size_t kStatePerPhone, typename Token>
BaseFloat LatticeIncrementalFactDecoderTpl<FST, kStatePerPhone, Token>::GetCutoff(
    ElemArc *list_head, size_t *tok_count, BaseFloat *adaptive_beam,
    ElemArc **best_elem, BaseFloat &best_weight) {
  best_weight = std::numeric_limits<BaseFloat>::infinity();
  // positive == high cost == bad.
  size_t count = 0;
  int max_active = config_.max_active;
  if (max_active >= std::numeric_limits<int32>::max() && config_.min_active == 0) {
    for (auto *e = list_head; e != NULL; e = e->tail, count++) {
      for (int i = 0; i < kStatePerPhone; i++) {
        auto *tok = e->val->tokens[i];
        if (tok) {
          BaseFloat w = static_cast<BaseFloat>(tok->tot_cost);
          if (w < best_weight) {
            best_weight = w;
            if (best_elem) *best_elem = e;
          }
        }
      }
    }
    if (tok_count != NULL) *tok_count = count;
    if (adaptive_beam != NULL) *adaptive_beam = config_.beam;
    return best_weight + config_.beam;
  } else {
    tmp_array_.clear();
    for (auto *e = list_head; e != NULL; e = e->tail, count++) {
      for (int i = 0; i < kStatePerPhone; i++) {
        auto *tok = e->val->tokens[i];
        if (tok) {
          BaseFloat w = static_cast<BaseFloat>(tok->tot_cost);
          tmp_array_.push_back(w);
          if (w < best_weight) {
            best_weight = w;
            if (best_elem) {
              *best_elem = e;
            }
          }
        }
      }
    }

    if (tok_count != NULL) *tok_count = count;

    BaseFloat beam_cutoff = best_weight + config_.beam,
              min_active_cutoff = std::numeric_limits<BaseFloat>::infinity(),
              max_active_cutoff = std::numeric_limits<BaseFloat>::infinity();

    KALDI_VLOG(6) << "Number of tokens active on frame " << NumFramesDecoded()
                  << " is " << tmp_array_.size();

    if (tmp_array_.size() > static_cast<size_t>(max_active)) {
      std::nth_element(tmp_array_.begin(), tmp_array_.begin() + max_active,
                       tmp_array_.end());
      max_active_cutoff = tmp_array_[max_active];
    }
    if (max_active_cutoff < beam_cutoff) { // max_active is tighter than beam.
      if (adaptive_beam)
        *adaptive_beam = max_active_cutoff - best_weight + config_.beam_delta;
      return max_active_cutoff;
    }
    if (tmp_array_.size() > static_cast<size_t>(config_.min_active)) {
      if (config_.min_active == 0)
        min_active_cutoff = best_weight;
      else {
        std::nth_element(tmp_array_.begin(), tmp_array_.begin() + config_.min_active,
                         tmp_array_.size() > static_cast<size_t>(max_active)
                             ? tmp_array_.begin() + max_active
                             : tmp_array_.end());
        // prune by tot_cost
        min_active_cutoff = tmp_array_[config_.min_active];
      }
    }
    if (min_active_cutoff > beam_cutoff) { // min_active is looser than beam.
      if (adaptive_beam)
        *adaptive_beam = min_active_cutoff - best_weight + config_.beam_delta;
      return min_active_cutoff;
    } else {
      *adaptive_beam = config_.beam;
      return beam_cutoff;
    }
  }
}

// We do GetCutoff() after SetEntryTokenForArcTokens() because GetCutoff() is used to
// control the computation of ProcessEmitting() which should take tokens from
// ArcIterator of SetEntryTokenForArcTokens() into account
template <typename FST, std::size_t kStatePerPhone, typename Token>
BaseFloat
LatticeIncrementalFactDecoderTpl<FST, kStatePerPhone, Token>::ProcessEmitting(
    DecodableInterface *decodable) {
  using namespace fst;
  KALDI_ASSERT(active_toks_.size() > 0);
  int32 frame = NumFramesDecoded() - 1;
  BaseFloat adaptive_beam, best_weight;
  size_t tok_cnt;
  ElemArc *best_elem = NULL;
  auto *final_toks =
      arc_toks_.Clear(); // analogous to swapping prev_toks_ / cur_toks_
  KALDI_ASSERT(final_toks);
  BaseFloat cur_cutoff =
      GetCutoff(final_toks, &tok_cnt, &adaptive_beam, &best_elem, best_weight);

  Token *&toks = active_toks_[frame + 1].toks;
  const BaseFloat extra_cost = std::numeric_limits<BaseFloat>::infinity();

  PossiblyResizeHash(tok_cnt); // This makes sure the hash is always big enough.

  BaseFloat next_best_cost = std::numeric_limits<BaseFloat>::infinity();
  BaseFloat next_cutoff = std::numeric_limits<BaseFloat>::infinity();
  // pruning "online" before having seen all tokens

  KALDI_ASSERT(best_elem);
  BaseFloat cost_offset = -best_weight;
  // Store the offset on the acoustic likelihoods that we're applying.
  // Could just do cost_offsets_.push_back(cost_offset), but we
  // do it this way as it's more robust to future code changes.
  cost_offsets_.resize(frame + 1, 0.0);
  cost_offsets_[frame] = cost_offset;

  ElemArc best_elem_head = *best_elem;
  best_elem_head.tail = final_toks;
  final_toks = &best_elem_head;
  size_t count = 0;
  Token *tok_buf[kStatePerPhone];
  for (ElemArc *e = final_toks, *e_tail; e != NULL; e = e_tail, count++) {
    e_tail = e->tail;
    bool remain_elem_arc = false;
    auto *arc_tok = e->val;
    memcpy(tok_buf, arc_tok->tokens, sizeof(Token *) * kStatePerPhone);
    auto &hmm_fst = h_transducers_[arc_tok->ilabel];
    KALDI_ASSERT(hmm_fst.NumStates() == kStatePerPhone);
    int start_state = 0;
    if (e !=
        &best_elem_head) { // just to get a reasonable cutoff from the best element
      memset(arc_tok->tokens, 0, sizeof(Token *) * kStatePerPhone);
      if (tok_buf[0]) {
        Token *&tok = arc_tok->tokens[1];
        Token *prev_tok = tok_buf[0];
        // the token is contructed in SetEntryTokenForArcTokens() and the
        // acoustic_cost and tot_cost are updated here
        auto *link = prev_tok->links;
        while (link) {
          // Add the cost_offset we obtain here
          link->acoustic_cost += cost_offset;
          link = link->next;
        }
        // We include the final ac_trans_cost to the tot_cost of the token. After
        // that, this token has been processed and all weights included
        BaseFloat ac_trans_cost = prev_tok->links->acoustic_cost;
        prev_tok->tot_cost += ac_trans_cost;
        // check whether to remain this token
        if (prev_tok->tot_cost < next_cutoff) {
          // Move the token from state 0 to state 1
          // Meanwhile, we have construct BackwardLinkT for it in
          // SetEntryTokenForArcTokens()
          tok = prev_tok;
          remain_elem_arc = true;
        }
      }
      start_state = 1; // state 0 has been processed above
    }
    for (int j = start_state; j < kStatePerPhone; j++) {
      auto *prev_tok = tok_buf[j];
      if (!prev_tok || prev_tok->tot_cost > cur_cutoff) continue;
      fst::ArcIterator<ConstFst<StdArc>> hmm_aiter(hmm_fst, j);
      for (; !hmm_aiter.Done(); hmm_aiter.Next()) {
        auto &hmm_arc = hmm_aiter.Value();
        int i = hmm_arc.nextstate;
        auto *&tok = arc_tok->tokens[i];
        // We apply a cur_cutoff on prev_tok
        Label tid = hmm_arc.ilabel;
        BaseFloat ac_cost =
            tid ? cost_offset - decodable->LogLikelihood(frame, tid) : 0;
        BaseFloat infinity = std::numeric_limits<BaseFloat>::infinity();
        BaseFloat trans_cost = hmm_arc.weight.Value();
        BaseFloat tot_cost =
            prev_tok ? prev_tok->tot_cost + ac_cost + trans_cost : infinity;
        // We apply a next_cutoff on the current tok
        if (tot_cost > next_cutoff)
          continue;
        else if (tot_cost + adaptive_beam < next_cutoff)
          next_cutoff = tot_cost + adaptive_beam; // prune by best current token

        if (e == &best_elem_head)
          continue; // just to get a reasonable cutoff from the best element

        // set cost for tok
        if (!tok) {
          tok = token_allocator_->allocate(1);
          token_allocator_->construct(tok, tot_cost, extra_cost, nullptr, toks);
          toks = tok;
          num_toks_++;
        } else if (tok->tot_cost > tot_cost)
          tok->tot_cost = tot_cost;
        // construct BackwardLink
        BackwardLinkT *t_link = tok->links;
        tok->links = link_allocator_->allocate(1);
        // the graph_cost is included in SetEntryTokenForArcTokens()
        Arc arc(tid, 0, 0, kNoStateId);
        link_allocator_->construct(tok->links, prev_tok, &arc, ac_cost + trans_cost,
                                   t_link);
        KALDI_ASSERT(tok != prev_tok);
        remain_elem_arc |=
            tok !=
            nullptr; // we will remain this elem arc if at least one new_tok exists
        if (tok && tok->tot_cost < next_best_cost) {
          next_best_cost = tok->tot_cost;
          last_best_arc_tok_ = arc_tok;
        }
      }
    }
    if (e == &best_elem_head) {
      KALDI_VLOG(6) << "Adaptive beam on frame " << NumFramesDecoded() << " is "
                    << adaptive_beam << " " << cur_cutoff << " " << next_cutoff << " "
                    << best_weight;
      continue;
    } else {
      arc_toks_.Delete(e);
      if (remain_elem_arc)
        arc_toks_.Insert(e->key, e->val); // TODO better method
      else
        arc_token_allocator_->deallocate(arc_tok, 1);
    }
  }

  return next_cutoff;
}

template <typename FST, std::size_t kStatePerPhone, typename Token>
inline
    typename LatticeIncrementalFactDecoderTpl<FST, kStatePerPhone, Token>::ArcToken *
    LatticeIncrementalFactDecoderTpl<FST, kStatePerPhone, Token>::FindOrAddArcToken(
        PairId state, const Arc &arc, int32 frame_plus_one, BaseFloat tot_cost,
        bool *changed) {
  KALDI_ASSERT(frame_plus_one < active_toks_.size());
  Token *&toks = active_toks_[frame_plus_one].toks;

  ElemArc *e_found = arc_toks_.Find(state);
  const BaseFloat extra_cost = std::numeric_limits<BaseFloat>::infinity();
  if (e_found == NULL) { // no such token presently.
    // construct first token for the following ArcToken
    Token *new_tok = token_allocator_->allocate(1);
    token_allocator_->construct(new_tok, tot_cost, extra_cost, nullptr, toks);
    toks = new_tok;
    num_toks_++;

    auto *new_arc_tok = arc_token_allocator_->allocate(1);
    arc_token_allocator_->construct(new_arc_tok, new_tok, arc.ilabel, arc.olabel,
                                    arc.nextstate);

    arc_toks_.Insert(state, new_arc_tok);
    if (changed) *changed = true;
    return new_arc_tok;
  } else {
    auto *arc_tok = e_found->val; // There is an existing Token for this state.
    auto *&tok = arc_tok->tokens[0];
    if (!tok) {
      Token *new_tok = token_allocator_->allocate(1);
      token_allocator_->construct(new_tok, tot_cost, extra_cost, nullptr, toks);
      toks = new_tok;
      num_toks_++;
      tok = new_tok;
      if (changed) *changed = true;
    } else if (tok->tot_cost > tot_cost) { // replace old token
      tok->tot_cost = tot_cost;
      if (changed) *changed = true;
    } else {
      if (changed) *changed = false;
    }
    return arc_tok;
  }
}

// Do both set entry and DeleteElems(toks_.Clear());
// pruning current frame based on cur_cutoff
// pruning the next frame based on last_best_arc_tok_ and the best_elem in the
// following
template <typename FST, std::size_t kStatePerPhone, typename Token>
void LatticeIncrementalFactDecoderTpl<FST, kStatePerPhone, Token>::
    SetEntryTokenForArcTokens(DecodableInterface *decodable, BaseFloat cur_cutoff,
                              BaseFloat adaptive_beam) {
  using namespace fst;
  int frame = NumFramesDecoded() - 1;
  Elem *final_toks = toks_.Clear();
  // the following cutoff includes a fake ac_cost which aims to help pruning
  BaseFloat next_cutoff_fake = std::numeric_limits<BaseFloat>::infinity();
  KALDI_ASSERT(final_toks);
  Elem best_elem;
  memset(&best_elem, 0, sizeof(Elem));
  // use last_best_arc_tok_ to help pruning
  if (last_best_arc_tok_) {
    auto *arc_tok = last_best_arc_tok_;
    auto &hmm_fst = h_transducers_[arc_tok->ilabel];
    KALDI_ASSERT(hmm_fst.NumStates() == kStatePerPhone);
    for (int i = 0; i < kStatePerPhone; i++) {
      fst::ArcIterator<ConstFst<StdArc>> hmm_aiter(hmm_fst, i);
      for (; !hmm_aiter.Done(); hmm_aiter.Next()) {
        auto &hmm_arc = hmm_aiter.Value();
        auto *prev_tok = arc_tok->tokens[i];
        if (prev_tok) {
          Label tid = hmm_arc.ilabel;
          BaseFloat ac_cost = tid ? -decodable->LogLikelihood(frame, tid) : 0;
          BaseFloat infinity = std::numeric_limits<BaseFloat>::infinity();
          BaseFloat trans_cost = hmm_arc.weight.Value();
          BaseFloat tot_cost =
              prev_tok ? prev_tok->tot_cost + ac_cost + trans_cost : infinity;
          if (tot_cost + adaptive_beam < next_cutoff_fake)
            next_cutoff_fake = tot_cost + adaptive_beam;
        }
      }
    }
  }
  // get the best_elem from toks_
  for (Elem *e = final_toks; e != NULL; e = e->tail) {
    Token *tok = e->val;
    StateId state = e->key;
    if (!best_elem.val || best_elem.val->tot_cost > tok->tot_cost) {
      best_elem = *e;
      best_pair = QueueElem(state, tok);
    }
  }
  // make the best_elem as the first one to process, which is to obtain a reasonable
  // cutoff at first
  best_elem.tail = final_toks;
  final_toks = &best_elem;
  for (Elem *e = final_toks, *e_tail; e != NULL; e = e_tail) {
    e_tail = e->tail;
    StateId state = e->key;
    Token *tok = e->val;
    fst::ArcIterator<FST> aiter(*fst_, state);
    if (fst_sorted_) aiter.Seek(fst_->NumInputEpsilons(state));
    for (int arc_idx = 0; !aiter.Done(); aiter.Next(), arc_idx++) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0) { // propagate..
        // Get the ac_cost of the current decoding step,
        // This ac_cost is only for helping pruning, we will include it later in
        // ProcessEmitting Hence we call it "fake"
        auto &hmm_fst = h_transducers_[arc.ilabel];
        fst::ArcIterator<ConstFst<StdArc>> hmm_aiter(hmm_fst, 0); // first state
        auto &hmm_arc = hmm_aiter.Value();
        auto tid = hmm_arc.ilabel;
        BaseFloat ac_cost_fake = tid ? -decodable->LogLikelihood(frame, tid) : 0,
                  trans_cost_fake = hmm_arc.weight.Value(),
                  graph_cost = arc.weight.Value(), cur_cost = tok->tot_cost,
                  tot_cost =
                      cur_cost +
                      graph_cost, // the real tot_cost does not include ac_cost_fake
            tot_cost_fake = cur_cost + ac_cost_fake + trans_cost_fake + graph_cost;
        // This token is still in current frame, so we prune it using cur_cutoff
        if (cur_cost + graph_cost > cur_cutoff)
          continue;
        else if (tot_cost_fake > next_cutoff_fake)
          continue;
        else if (tot_cost_fake + adaptive_beam < next_cutoff_fake)
          next_cutoff_fake = tot_cost_fake + adaptive_beam;
        // to help pruning by best current token
        // we process best_elem in the very first just to get a reasonable
        // next_cutoff_fake . the real processing (add BackwardLinkT) will be later.
        if (e == &best_elem) continue;

        // we do not use ArcIterator::Position() because GrammarFst does not have this
        // implementation
        PairId next_pair = ConstructPair(state, arc_idx);
        bool changed;
        // findoradd ArcToken and findoradd ArcToken.tokens[0] together
        auto *next_arc_tok = FindOrAddArcToken(
            next_pair, arc,
            frame + 1, // this token will be moved to the state 1 of h_transducers_,
                       // and include ac_cost in ProcessEmitting(), so we make it
                       // (frame+1)
            tot_cost,  // we didn't have the cost_offset of this frame yet. Hence we
                       // didn't include the ac_cost_fake and trans_cost_fake into the
                       // cost here. Notably, the cost here will be included into
                       // GetCutoff() later
            &changed);

        auto *next_tok = next_arc_tok->tokens[0];
        // Add BackwardLinkT from tok to next_tok (put on head of list tok->links)
        BackwardLinkT *t_link = next_tok->links;
        next_tok->links = link_allocator_->allocate(1);
        Arc arc_copy(hmm_arc.ilabel, 0, graph_cost,
                     kNoStateId); // we will include the olabel in another new arc to
                                  // arc.nextstate
        link_allocator_->construct(next_tok->links, tok, &arc_copy,
                                   ac_cost_fake + trans_cost_fake, t_link);
        // the weights here are fake weights since the ac_cost is without cost_offset.
        // We obtain the cost_offset in ProcessEmitting() and transform these "fake"
        // weights into the final weight
        KALDI_ASSERT(next_tok != tok);

      } else if (fst_sorted_)
        break;
    }
    if (e != &best_elem) toks_.Delete(e);
  }
}

template <typename FST, std::size_t kStatePerPhone, typename Token>
void LatticeIncrementalFactDecoderTpl<
    FST, kStatePerPhone, Token>::ExpandArcTokensToNextState(BaseFloat next_cutoff) {
  using namespace fst;
  int frame = NumFramesDecoded();
  auto *final_toks =
      arc_toks_.GetList(); // analogous to swapping prev_toks_ / cur_toks_
  size_t count = 0;
  expand_num_ = 0;
  for (const ElemArc *e = final_toks, *e_tail; e != NULL; e = e_tail, count++) {
    e_tail = e->tail;
    auto *arc_tok = e->val;
    bool changed = false;
    Token *next_tok = NULL;

    for (int i = 0; i < kStatePerPhone; i++) {
      auto *&tok = arc_tok->tokens[i];
      if (tok) {
        auto &hmm_fst = h_transducers_[arc_tok->ilabel];
        // if state i is not a final state, the cost will be infinite and cannot pass
        // the next_cutoff threshold
        BaseFloat expand_cost =
            tok->tot_cost + hmm_fst.Final(i).Value(); // add trans weight
        if (expand_cost > next_cutoff) continue;

        if (!next_tok)
          next_tok =
              FindOrAddToken(arc_tok->nextstate, frame, expand_cost,
                             &changed); // TODO: do it once for kStatePerPhone states
        else if (expand_cost < next_tok->tot_cost) {
          next_tok->tot_cost = expand_cost;
          changed = true;
        }
        BackwardLinkT *t_link = next_tok->links;
        next_tok->links = link_allocator_->allocate(1);
        // record arc_tok->olabel in this BackwardLink
        Arc arc(0, arc_tok->olabel, expand_cost - tok->tot_cost, kNoStateId);
        link_allocator_->construct(next_tok->links, tok, &arc, 0, t_link);
        KALDI_ASSERT(next_tok != tok);
      }
    }
    // prep for ProcessNonemitting
    if (changed) {
      queue_.push_back(QueueElem(arc_tok->nextstate, next_tok));
      expand_num_++;
    }
  }
}

template <typename FST, std::size_t kStatePerPhone, typename Token>
void LatticeIncrementalFactDecoderTpl<FST, kStatePerPhone, Token>::DeleteElemArcs(
    typename LatticeIncrementalFactDecoderTpl<FST, kStatePerPhone, Token>::ElemArc
        *list) {
  for (ElemArc *e = list, *e_tail; e != NULL; e = e_tail) {
    e_tail = e->tail;
    arc_toks_.Delete(e);
  }
}

// Instantiate the template for the combination of token types and FST types
// that we'll need.
template class LatticeIncrementalFactDecoderTpl<fst::Fst<fst::StdArc>, 2>;
template class LatticeIncrementalFactDecoderTpl<fst::Fst<fst::StdArc>, 3>;
template class LatticeIncrementalFactDecoderTpl<fst::VectorFst<fst::StdArc>, 2>;
template class LatticeIncrementalFactDecoderTpl<fst::VectorFst<fst::StdArc>, 3>;
template class LatticeIncrementalFactDecoderTpl<fst::ConstFst<fst::StdArc>, 2>;
template class LatticeIncrementalFactDecoderTpl<fst::ConstFst<fst::StdArc>, 3>;
template class LatticeIncrementalFactDecoderTpl<fst::GrammarFst, 2>;
template class LatticeIncrementalFactDecoderTpl<fst::GrammarFst, 3>;

} // end namespace kaldi.
