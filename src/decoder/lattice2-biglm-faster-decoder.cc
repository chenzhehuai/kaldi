// decoder/lattice2-biglm-faster-decoder.h

// Copyright      2018  Hang Lyu  Zhehuai Chen

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

#include "decoder/lattice2-biglm-faster-decoder.h"

namespace kaldi {

Lattice2BiglmFasterDecoder::Lattice2BiglmFasterDecoder(
    const fst::Fst<fst::StdArc> &fst,
    const Lattice2BiglmFasterDecoderConfig &config,
    fst::DeterministicOnDemandFst<fst::StdArc> *lm_diff_fst):
    fst_(fst), lm_diff_fst_(lm_diff_fst), config_(config),
  warned_noarc_(false), num_toks_(0) {
  config.Check();
  KALDI_ASSERT(fst.Start() != fst::kNoStateId &&
               lm_diff_fst->Start() != fst::kNoStateId);
  toks_.SetSize(1000);  // just so on the first frame we do something reasonable.
  for (int i = 0; i < 2; i++) toks_shadowing_[i].SetSize(1000);  // just so on the first frame we do something reasonable.
    toks_backfill_hclg_.resize(0);
  }


bool Lattice2BiglmFasterDecoder::Decode(DecodableInterface *decodable) {
    
  InitDecoding();
  // We use 1-based indexing for frames in this decoder (if you view it in
  // terms of features), but note that the decodable object uses zero-based
  // numbering, which we have to correct for when we call it.
  int32 last_expand_frame=0;
  for (int32 frame = 1; !decodable->IsLastFrame(frame-2); frame++) {
    active_toks_.resize(frame+1); // new column

    ProcessEmitting(decodable, frame);

    ProcessNonemitting(frame);

    if (frame % config_.prune_interval == 0) {
      PruneActiveTokens(frame, config_.lattice_beam * 0.1); // use larger delta.
    }
    int32 t = frame-config_.prune_interval-config_.explore_interval;
    if (t >= 0 && (frame-config_.explore_interval) % config_.prune_interval == 0) {
      KALDI_ASSERT(t==last_expand_frame);
      for (; t<=frame; t++)
        ExpandShadowTokens(t, frame-config_.explore_interval-1, decodable, t==last_expand_frame);
      last_expand_frame=frame-config_.explore_interval;
    }
    

    // We could add another config option to decide the gap between state passing
    // and lm passing.
  }

  PruneActiveTokens(NumFramesDecoded(), config_.lattice_beam * 0.1);

  for (int32 t=last_expand_frame; t<=NumFramesDecoded(); t++)
    ExpandShadowTokens(t, NumFramesDecoded(), decodable, t==last_expand_frame);

  // Process the last few frames lm passing
  PruneActiveTokensFinal(NumFramesDecoded(), true); // with sanity check
  KALDI_VLOG(1) << "propage_lm_num_: " << propage_lm_expand_num_ << " " << propage_lm_num_;

  // Returns true if we have any kind of traceback available (not necessarily
  // to the end state; query ReachedFinal() for that).
  return !final_costs_.empty();
}

bool Lattice2BiglmFasterDecoder::Decode(DecodableInterface *decodable,
                                        const Vector<BaseFloat> &cutoff) {
  // //initial cutoff_
  if (cutoff.Dim()) {
    cutoff_.Resize(cutoff.Dim());
    cutoff_ = cutoff;
  } else {
    cutoff_.Resize(1);
    cutoff_.Data()[0] = std::numeric_limits<int32>::max();
  }
  return Decode(decodable);
}



void Lattice2BiglmFasterDecoder::ExpandShadowTokens(int32 cur_frame, int32 frame_stop_expand, DecodableInterface *decodable, bool first) {
  Timer timer;
  
  expanding_=true;
  bool is_last = cur_frame <= frame_stop_expand; // the last time we do expand in this frame
  KALDI_ASSERT(cur_frame >= 0);
  auto& cur_q = GetExpandQueue(cur_frame);
  auto& cur_h = GetBackfillMap(cur_frame);
  if (cur_frame > frame_stop_expand && !cur_q.size()) {
    expanding_=false;
    return;
  }

  if (first) BuildBackfillMap(cur_frame, frame_stop_expand, first);
  if ( (cur_frame + 1) < active_toks_.size()) {
    BuildBackfillMap(cur_frame+1, frame_stop_expand, true);
  }

  while (!cur_q.empty()) {
    auto q_elem= cur_q.front();
    cur_q.pop();
    Token* tok = q_elem.first;
    tok->in_queue=false;
    bool cur_better_hclg = q_elem.second;
    int32 frame = cur_frame;
    BaseFloat cur_cutoff = (frame+1 < cutoff_.Dim())?
cutoff_(frame+1) : std::numeric_limits<BaseFloat>::infinity();
   
    if (tok->tot_cost > cur_cutoff) {
      tok->shadowing_tok = NULL;  // already expand
      continue;
    }

    ForwardLink *link=NULL, *links_to_clear=NULL;
    if (tok->shadowing_tok == NULL) {
      // if we need to update a shadowing token itself
      link=tok->links;
      tok->links=NULL; // we firstly un-hook it from shadowing token
      links_to_clear=link; // we will reconstruct it and delete the original one later
    } else {
      // otherwise, we are updating a shadowed token
      // we obtain template links from shadowing token
      // sanity check:
      // KALDI_ASSERT(toks_backfill_hclg_[frame]->find(tok->hclg_state)->second==tok->shadowing_tok);
      KALDI_ASSERT(!tok->links);
      Token* shadowing_tok = tok;
      while (shadowing_tok->shadowing_tok && !shadowing_tok->links) shadowing_tok = shadowing_tok->shadowing_tok;
      // Update toks_shadowing_mod for better_hclg here
      // Notice that we only update if it reaches NumFramesDecoded(), since it will affect explore
      if (frame == NumFramesDecoded() && *tok > *shadowing_tok) {
        HashList<StateId, Token*> &toks_shadowing_mod=toks_shadowing_[frame%2];
        ElemShadow *elem = toks_shadowing_mod.Find(tok->hclg_state);
        if (elem) {
          if (*tok < *elem->val) {
            elem->val = tok;
            // sanity check
            // KALDI_ASSERT(!(*toks_backfill_hclg_[frame]).find(tok->hclg_state)->second->shadowing_tok);
          }
        } else // from better_hclg
          toks_shadowing_mod.Insert(tok->hclg_state, tok);
      }
      link = shadowing_tok->links;
      if (!link) {
        // for the end of decoding, we need to expand all
        if (is_last)
          tok->shadowing_tok = NULL;
        else if (frame == NumFramesDecoded()) {
          auto iter = *toks_backfill_hclg_[frame]->find(tok->hclg_state);
          KALDI_ASSERT(iter.second != tok);
          if (*iter.second > *tok) // better_hclg
            tok->shadowing_tok = NULL;
          else
            KALDI_ASSERT(tok->shadowing_tok == iter.second); 
        } // for normal shadowed token is_last==false, we process it later
        continue; 
      }
    }

    if (cur_better_hclg && config_.better_hclg==2) {
      for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, tok->hclg_state);
           !aiter.Done();
           aiter.Next()) {
        Arc arc = aiter.Value();
        StateId ilabel = arc.ilabel;
        int32 new_frame_index = ilabel ? frame+1 : frame; 
        if (new_frame_index > NumFramesDecoded()) continue;
        BaseFloat graph_cost_ori = arc.weight.Value();
        StateId new_hclg_state = arc.nextstate;
        Arc new_arc(arc);
        StateId new_lm_state = PropagateLm(tok->lm_state, &arc); // may affect "arc.weight".
        BaseFloat ac_cost = ilabel ? -decodable->LogLikelihood(frame, ilabel) : 0,
                  graph_cost = new_arc.weight.Value(),
                  cur_cost = tok->tot_cost,
                  tot_cost = cur_cost + ac_cost + graph_cost;
  
        BaseFloat extra_cost = tok->extra_cost, // TODO
                  backward_cost = tok->backward_cost;
  
        // prepare to store a new token in the current / next frame
        if (new_frame_index+1 < cutoff_.Dim() &&
            tot_cost > cutoff_(new_frame_index+1)) continue;
        if (extra_cost > config_.lattice_beam) continue;
        Token* new_tok = ExpandShadowTokensSub(ilabel, new_hclg_state, new_lm_state, frame, new_frame_index, tot_cost, extra_cost, backward_cost, is_last);
        // create lattice arc
        tok->links = new ForwardLink(new_tok, arc.ilabel, arc.olabel, 
                                     graph_cost, ac_cost, tok->links,
                                     graph_cost_ori);
      }
    } else {
      // There will be four kinds of links need to be processed.
      // 1. Go to next frame and the corresponding "next_tok" is shadowed
      // 2. Go to next frame and the corresponding "next_tok" is the processed
      // (Under most circumstances, it is the best one and processed in
      // explore step)
      // 3. Still in current frame and the corresponding "next_tok" is
      // shadowed
      // 4. Still in current frame and the corresponding "next_tok" is
      // processed.(Under most circumstances, it is the best one and processed
      // in explore step)
      // However, the way to deal with them is similar.
      for (; link != NULL; link = link->next) {
        Token *next_tok = link->next_tok;
        while (next_tok->shadowing_tok && !next_tok->links) next_tok=next_tok->shadowing_tok;
        StateId ilabel = link->ilabel;
        int32 new_frame_index = ilabel ? frame+1 : frame; 
        if (new_frame_index<NumFramesDecoded() && !next_tok->links) continue; // this link should be pruned
        
        Arc arc(ilabel, link->olabel, link->graph_cost_ori, 0);
        BaseFloat graph_cost_ori = link->graph_cost_ori; // TODO
        StateId new_hclg_state = next_tok->hclg_state;
        StateId new_lm_state = PropagateLm(tok->lm_state, &arc); // may affect "arc.weight".
        BaseFloat ac_cost = link->acoustic_cost,
                  graph_cost = arc.weight.Value(),
                  cur_cost = tok->tot_cost,
                  tot_cost = cur_cost + ac_cost + graph_cost;
  
        // The extra_cost and backward_cost are temporary. They are inherited from
        // "next_tok" which is the destation of "shadowing_token". So they are
        // estimated rather than exact. They will be used to initialize a new
        // token and help to decide the new token will be expanded or not
        BaseFloat extra_cost = next_tok->extra_cost + tot_cost - next_tok->tot_cost, // inherit backward cost, use its own tot_cost
                  backward_cost = next_tok->backward_cost;
  
        // prepare to store a new token in the current / next frame
        if (new_frame_index+1 < cutoff_.Dim() &&
            tot_cost > cutoff_(new_frame_index+1)) continue;
        if (extra_cost > config_.lattice_beam) continue;
        Token* new_tok = ExpandShadowTokensSub(ilabel, new_hclg_state, new_lm_state, frame, new_frame_index, tot_cost, extra_cost, backward_cost, is_last);
        // create lattice arc
        tok->links = new ForwardLink(new_tok, arc.ilabel, arc.olabel, 
                                     graph_cost, ac_cost, tok->links,
                                     graph_cost_ori);
      }  // end of for loop
    }
    while (links_to_clear) {
      ForwardLink* l=links_to_clear->next;
      delete links_to_clear;
      links_to_clear = l;
    }
    tok->shadowing_tok = NULL;  // already expand
  }

  // Clean the backfill map
  cur_h.clear();
  if (is_last) toks_backfill_hclg_[cur_frame]->clear();
    
  KALDI_VLOG(2) << "expand fr num: " << cur_frame << " " << is_last << " " << GetExpandQueue(cur_frame+1).size() << " " << ToksNum(cur_frame);
  expanding_=false;
  expand_time_ += timer.Elapsed();
}

 
bool Lattice2BiglmFasterDecoder::GetRawLattice(
    fst::MutableFst<LatticeArc> *ofst,
    bool use_final_probs) const {
  typedef LatticeArc Arc;
  typedef Arc::StateId StateId;
  // A PairId will be constructed as: (StateId in fst) + (StateId in lm_diff_fst) << 32;
  typedef uint64 PairId;
  typedef Arc::Weight Weight;
  typedef Arc::Label Label;
  ofst->DeleteStates();
  // num-frames plus one (since frames are one-based, and we have
  // an extra frame for the start-state).
  int32 num_frames = active_toks_.size() - 1;
  KALDI_ASSERT(num_frames > 0);
  unordered_map<Token*, StateId> tok_map(num_toks_/2 + 3); // bucket count
  // First create all states.
  for (int32 f = 0; f <= num_frames; f++) {
    if (active_toks_[f].toks == NULL) {
      KALDI_WARN << "GetRawLattice: no tokens active on frame " << f
                 << ": not producing lattice.\n";
      return false;
    }
    for (Token *tok = active_toks_[f].toks; tok != NULL; tok = tok->next)
      tok_map[tok] = ofst->AddState();
    // The next statement sets the start state of the output FST.
    // Because we always add new states to the head of the list
    // active_toks_[f].toks, and the start state was the first one
    // added, it will be the last one added to ofst.
    if (f == 0 && ofst->NumStates() > 0)
      ofst->SetStart(ofst->NumStates()-1);
  }
  KALDI_VLOG(3) << "init:" << num_toks_/2 + 3 << " buckets:" 
                << tok_map.bucket_count() << " load:" << tok_map.load_factor() 
                << " max:" << tok_map.max_load_factor();
  // Now create all arcs.
  StateId cur_state = 0; // we rely on the fact that we numbered these
  // consecutively (AddState() returns the numbers in order..)
  for (int32 f = 0; f <= num_frames; f++) {
    for (Token *tok = active_toks_[f].toks; tok != NULL; tok = tok->next,
         cur_state++) {
      for (ForwardLink *l = tok->links;
           l != NULL;
           l = l->next) {
        unordered_map<Token*, StateId>::const_iterator iter =
          tok_map.find(l->next_tok);
        // the tok has been pruned, so the arc do not need to exist
        // TODO better way to solve it in pruning arc
        if (iter == tok_map.end()) continue; 
        StateId nextstate = iter->second;
        KALDI_ASSERT(iter != tok_map.end());
        Arc arc(l->ilabel, l->olabel,
                Weight(l->graph_cost, l->acoustic_cost),
                nextstate);
        ofst->AddArc(cur_state, arc);
      }
      if (f == num_frames) {
        if (use_final_probs && !final_costs_.empty()) {
          std::map<Token*, BaseFloat>::const_iterator iter =
            final_costs_.find(tok);
          if (iter != final_costs_.end())
            ofst->SetFinal(cur_state, LatticeWeight(iter->second, 0));
        } else {
          ofst->SetFinal(cur_state, LatticeWeight::One());
        }
      }
    }
  }
  KALDI_ASSERT(cur_state == ofst->NumStates());
  return (cur_state != 0);
}



bool Lattice2BiglmFasterDecoder::GetLattice(
    fst::MutableFst<CompactLatticeArc> *ofst,
    bool use_final_probs) const {
  Lattice raw_fst;
  if (!GetRawLattice(&raw_fst, use_final_probs)) return false;
  Invert(&raw_fst); // make it so word labels are on the input.
  if (!TopSort(&raw_fst)) // topological sort makes lattice-determinization more efficient
    KALDI_WARN << "Topological sorting of state-level lattice failed "
        "(probably your lexicon has empty words or your LM has epsilon cycles; this "
        " is a bad idea.)";
  // (in phase where we get backward-costs).
  fst::ILabelCompare<LatticeArc> ilabel_comp;
  ArcSort(&raw_fst, ilabel_comp); // sort on ilabel; makes
  // lattice-determinization more efficient.
    
  fst::DeterminizeLatticePrunedOptions lat_opts;
  lat_opts.max_mem = config_.det_opts.max_mem;
    
  DeterminizeLatticePruned(raw_fst, config_.lattice_beam, ofst, lat_opts);
  raw_fst.DeleteStates(); // Free memory-- raw_fst no longer needed.
  Connect(ofst); // Remove unreachable states... there might be
  // a small number of these, in some cases.
  return true;
}


void Lattice2BiglmFasterDecoder::PruneForwardLinks(int32 frame,
                                                   bool *extra_costs_changed,
                                                   bool *links_pruned,
                                                   BaseFloat delta, bool is_expand=false) {
  // delta is the amount by which the extra_costs must change
  // If delta is larger,  we'll tend to go back less far
  //    toward the beginning of the file.
  // extra_costs_changed is set to true if extra_cost was changed for any token
  // links_pruned is set to true if any link in any token was pruned

  *extra_costs_changed = false;
  *links_pruned = false;
  KALDI_ASSERT(frame >= 0 && frame < active_toks_.size());
  if (active_toks_[frame].toks == NULL ) { // empty list; should not happen.
    if (!warned_) {
      KALDI_WARN << "No tokens alive [doing pruning].. warning first "
          "time only for each utterance\n";
      warned_ = true;
    }
  }
    
  // We have to iterate until there is no more change, because the links
  // are not guaranteed to be in topological order.
  bool changed = true; // difference new minus old extra cost >= delta ?
  while (changed) {
    changed = false;
    for (Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
      ForwardLink *link, *prev_link=NULL;
      // will recompute tok_extra_cost for tok.
      BaseFloat tok_extra_cost = std::numeric_limits<BaseFloat>::infinity();
      // tok_extra_cost is the best (min) of link_extra_cost of outgoing links
      if (tok->shadowing_tok && tok->links) { // has been expanded
        if (*tok > *tok->shadowing_tok) tok->DeleteForwardLinks();
        else tok->shadowing_tok = NULL;
      }
      for (link = tok->links; link != NULL; ) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        BaseFloat link_extra_cost = 0.0;
        if (is_expand && next_tok->shadowing_tok) {
          KALDI_ASSERT(!next_tok->links);
          next_tok->shadowing_tok=NULL; // hasn't pruned but it should do
        } 
        if (next_tok->shadowing_tok) {
          Token* s=next_tok->shadowing_tok;
          while (s->shadowing_tok) s=s->shadowing_tok;
          link_extra_cost = s->extra_cost +
            ((tok->tot_cost + link->acoustic_cost + link->graph_cost)
             - s->tot_cost);
        } else {
          link_extra_cost = next_tok->extra_cost +
            ((tok->tot_cost + link->acoustic_cost + link->graph_cost)
             - next_tok->tot_cost); // difference in brackets is >= 0
        }
        // link_exta_cost is the difference in score between the best paths
        // through link source state and through link destination state
        KALDI_ASSERT(link_extra_cost == link_extra_cost); // check for NaN
        if (link_extra_cost > config_.lattice_beam) { // excise link
          ForwardLink *next_link = link->next;
          if (prev_link != NULL) prev_link->next = next_link;
          else tok->links = next_link;
          delete link;
          link = next_link;  // advance link but leave prev_link the same.
          *links_pruned = true;
        } else {   // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) {  // this is just a precaution.
            if (link_extra_cost < -0.01)
              //KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
            link_extra_cost = 0.0;
          }
          if (link_extra_cost < tok_extra_cost)
            tok_extra_cost = link_extra_cost;
          prev_link = link;  // move to next link
          link = link->next;
        }
      }  // for all outgoing links
      if (fabs(tok_extra_cost - tok->extra_cost) > delta)
        changed = true;   // difference new minus old is bigger than delta
      tok->extra_cost = tok_extra_cost;
      // will be +infinity or <= lattice_beam_.
      // infinity indicates, that no forward link survived pruning
    }  // for all Token on active_toks_[frame]
    if (changed) *extra_costs_changed = true;

    // Note: it's theoretically possible that aggressive compiler
    // optimizations could cause an infinite loop here for small delta and
    // high-dynamic-range scores.
  } // while changed
}


void Lattice2BiglmFasterDecoder::PruneForwardLinksFinal(int32 frame) {
  KALDI_ASSERT(static_cast<size_t>(frame+1) == active_toks_.size());
  if (active_toks_[frame].toks == NULL ) // empty list; should not happen.
    KALDI_WARN << "No tokens alive at end of file\n";

  // First go through, working out the best token (do it in parallel
  // including final-probs and not including final-probs; we'll take
  // the one with final-probs if it's valid).
  const BaseFloat infinity = std::numeric_limits<BaseFloat>::infinity();
  BaseFloat best_cost_final = infinity,
      best_cost_nofinal = infinity;
  unordered_map<Token*, BaseFloat> tok_to_final_cost;
  Elem *cur_toks = toks_.Clear(); // swapping prev_toks_ / cur_toks_
  for (Elem *e = cur_toks, *e_tail; e != NULL;  e = e_tail) {
    PairId state_pair = e->key;
    StateId state = PairToState(state_pair),
         lm_state = PairToLmState(state_pair);
    Token *tok = e->val;
    BaseFloat final_cost = fst_.Final(state).Value() +
        lm_diff_fst_->Final(lm_state).Value();
    tok_to_final_cost[tok] = final_cost;
    best_cost_final = std::min(best_cost_final, tok->tot_cost + final_cost);
    best_cost_nofinal = std::min(best_cost_nofinal, tok->tot_cost);
    e_tail = e->tail;
    toks_.Delete(e);
  }
  final_active_ = (best_cost_final != infinity);
    
  // Now go through tokens on this frame, pruning forward links...  may have
  // to iterate a few times until there is no more change, because the list is
  // not in topological order.

  bool changed = true;
  BaseFloat delta = 1.0e-05;
  while (changed) {
    changed = false;
    for (Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
      ForwardLink *link, *prev_link=NULL;
      // will recompute tok_extra_cost.  It has a term in it that corresponds
      // to the "final-prob", so instead of initializing tok_extra_cost to infinity
      // below we set it to the difference between the (score+final_prob) of this token,
      // and the best such (score+final_prob).
      BaseFloat tok_extra_cost;
      if (final_active_) {
        BaseFloat final_cost = tok_to_final_cost[tok];
        tok_extra_cost = (tok->tot_cost + final_cost) - best_cost_final;
      } else 
        tok_extra_cost = tok->tot_cost - best_cost_nofinal;
    
      for (link = tok->links; link != NULL; ) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        KALDI_ASSERT(!next_tok->shadowing_tok);
        BaseFloat link_extra_cost = next_tok->extra_cost +
            ((tok->tot_cost + link->acoustic_cost + link->graph_cost)
             - next_tok->tot_cost);
        if (link_extra_cost > config_.lattice_beam) { // excise link
          ForwardLink *next_link = link->next;
          if (prev_link != NULL) prev_link->next = next_link;
          else tok->links = next_link;
          delete link;
          link = next_link; // advance link but leave prev_link the same.
        } else { // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) { // this is just a precaution.
            if (link_extra_cost < -0.01)
              //KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
            link_extra_cost = 0.0;
          }
          if (link_extra_cost < tok_extra_cost)
            tok_extra_cost = link_extra_cost;
          prev_link = link;
          link = link->next;
        }
      }
      // prune away tokens worse than lattice_beam above best path.  This step
      // was not necessary in the non-final case because then, this case
      // showed up as having no forward links.  Here, the tok_extra_cost has
      // an extra component relating to the final-prob.
      if (tok_extra_cost > config_.lattice_beam)
        tok_extra_cost = infinity;
      // to be pruned in PruneTokensForFrame
      if (!ApproxEqual(tok->extra_cost, tok_extra_cost, delta))
        changed = true;
      tok->extra_cost = tok_extra_cost; // will be +infinity or <= lattice_beam_.
    }
  } // while changed

  // Now put surviving Tokens in the final_costs_ hash, which is a class
  // member (unlike tok_to_final_costs).
  for (Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {    
    if (tok->extra_cost != infinity) {
      // If the token was not pruned away, 
      if (final_active_) {
        BaseFloat final_cost = tok_to_final_cost[tok];         
        if (final_cost != infinity)
          final_costs_[tok] = final_cost;
      } else {
        final_costs_[tok] = 0;
      }
    }
  }
}


void Lattice2BiglmFasterDecoder::PruneTokensForFrame(int32 frame, bool is_expand=false) {
  KALDI_ASSERT(frame >= 0 && frame < active_toks_.size());
  Token *&toks = active_toks_[frame].toks;
  if (toks == NULL)
    KALDI_WARN << "No tokens alive [doing pruning]\n";
  Token *tok, *next_tok, *prev_tok = NULL;
  // proc shadowed token at first as it needs info from shadowing token
  for (tok = toks; tok != NULL; tok = next_tok) {
    next_tok = tok->next;
    if (is_expand && tok->shadowing_tok) {
      KALDI_ASSERT(!tok->links);
      tok->shadowing_tok=NULL; // hasn't pruned but it should do
    }
    if (tok->shadowing_tok) {// shadowed token
      if (tok->shadowing_tok->extra_cost == std::numeric_limits<BaseFloat>::infinity()) {
        // token is unreachable from end of graph; (no forward links survived)
        // excise tok from list and delete tok.
        if (prev_tok != NULL) prev_tok->next = tok->next;
        else toks = tok->next;
        delete tok;
        num_toks_--;
      } else { // fetch next Token
        prev_tok = tok;
        // KALDI_ASSERT(tok->shadowing_tok->tot_cost <= tok->tot_cost);
        // After expanding, sometimes the tok->tot_cost better than shadowing's.
        Token* s=tok->shadowing_tok;
        while (s->shadowing_tok) s=s->shadowing_tok;
        tok->extra_cost = s->extra_cost +
                          s->tot_cost - tok->tot_cost;
      }
    } else {
      prev_tok = tok;
    }
  }
  prev_tok = NULL;
  for (tok = toks; tok != NULL; tok = next_tok) {
    next_tok = tok->next;
    if (!tok->shadowing_tok) { // shadowing token
      if (tok->extra_cost == std::numeric_limits<BaseFloat>::infinity()) { 
        // token is unreachable from end of graph; (no forward links survived)
        // excise tok from list and delete tok.
        if (toks_backfill_hclg_.size()>frame && frame >= NumFramesDecoded()-config_.prune_interval) { // the map has been built 
          if (toks_backfill_hclg_[frame]->erase(tok->hclg_state))
            ; //for (Token* t=toks; t; t=t->next) KALDI_ASSERT(t->shadowing_tok!=tok); // sanity check
        }
        if (prev_tok != NULL) prev_tok->next = tok->next;
        else toks = tok->next;
        delete tok;
        num_toks_--;
      } else { // fetch next Token
        prev_tok = tok;
      }
    } else {
      prev_tok = tok;
    }
  }
}
 

void Lattice2BiglmFasterDecoder::PruneActiveTokens(int32 cur_frame, 
                                                   BaseFloat delta) {
  int32 num_toks_begin = num_toks_;
  for (int32 frame = cur_frame-1; frame >= 0; frame--) {
    // Reason why we need to prune forward links in this situation:
    // (1) we have never pruned them (new TokenList)
    // (2) we have not yet pruned the forward links to the next frame,
    // after any of those tokens have changed their extra_cost.
    if (active_toks_[frame].must_prune_forward_links) {
      bool extra_costs_changed = false, links_pruned = false;
      PruneForwardLinks(frame, &extra_costs_changed, &links_pruned, delta);
      if (extra_costs_changed && frame > 0) // any token has changed extra_cost
        active_toks_[frame-1].must_prune_forward_links = true;
      if (links_pruned) // any link was pruned
        active_toks_[frame].must_prune_tokens = true;
      active_toks_[frame].must_prune_forward_links = false; // job done
    }
    if (frame+1 < cur_frame &&      // except for last frame (no forward links)
       active_toks_[frame+1].must_prune_tokens) {
      PruneTokensForFrame(frame+1);
      active_toks_[frame+1].must_prune_tokens = false;
    }
  }
  KALDI_VLOG(3) << "PruneActiveTokens: pruned tokens from " << num_toks_begin
                << " to " << num_toks_;
}


void Lattice2BiglmFasterDecoder::PruneActiveTokensFinal(int32 cur_frame, bool is_expand) {
  // returns true if there were final states active
  // else returns false and treats all states as final while doing the pruning
  // (this can be useful if you want partial lattice output,
  // although it can be dangerous, depending what you want the lattices for).
  // final_active_ and final_probs_ (a hash) are set internally
  // by PruneForwardLinksFinal
  int32 num_toks_begin = num_toks_;
  PruneForwardLinksFinal(cur_frame); // prune final frame (with final-probs)
  // sets final_active_ and final_probs_
  for (int32 frame = cur_frame-1; frame >= 0; frame--) {
    bool b1, b2; // values not used.
    BaseFloat dontcare = 0.0; // delta of zero means we must always update
    PruneForwardLinks(frame, &b1, &b2, dontcare, is_expand);
    PruneTokensForFrame(frame+1, is_expand);
  }
  PruneTokensForFrame(0, is_expand);
  KALDI_VLOG(3) << "PruneActiveTokensFinal: pruned tokens from " << num_toks_begin
                << " to " << num_toks_;
}
  
  /// Gets the weight cutoff.  Also counts the active tokens.
BaseFloat Lattice2BiglmFasterDecoder::GetCutoff(Elem *list_head, 
                                                size_t *tok_count,
                                                BaseFloat *adaptive_beam,
                                                Elem **best_elem) {
  BaseFloat best_weight = std::numeric_limits<BaseFloat>::infinity();
  // positive == high cost == bad.
  size_t count = 0;
  if (config_.max_active == std::numeric_limits<int32>::max()) {
    for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
      BaseFloat w = static_cast<BaseFloat>(e->val->tot_cost);
      if (w < best_weight) {
        best_weight = w;
        if (best_elem) *best_elem = e;
      }
    }
    if (tok_count != NULL) *tok_count = count;
    if (adaptive_beam != NULL) *adaptive_beam = config_.beam;
    return best_weight + config_.beam;
  } else {
    tmp_array_.clear();
    for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
      BaseFloat w = e->val->tot_cost;
      tmp_array_.push_back(w);
      if (w < best_weight) {
        best_weight = w;
        if (best_elem) *best_elem = e;
      }
    }
    if (tok_count != NULL) *tok_count = count;
    KALDI_VLOG(6) << "Number of tokens active on frame " << active_toks_.size()
                  << " is " << tmp_array_.size();
    if (tmp_array_.size() <= static_cast<size_t>(config_.max_active)) {
      if (adaptive_beam) *adaptive_beam = config_.beam;
        return best_weight + config_.beam;
    } else {
      // the lowest elements (lowest costs, highest likes)
      // will be put in the left part of tmp_array.
      std::nth_element(tmp_array_.begin(),
                       tmp_array_.begin()+config_.max_active,
                       tmp_array_.end());
      // return the tighter of the two beams.
      BaseFloat ans = std::min(best_weight + config_.beam,
                               *(tmp_array_.begin()+config_.max_active));
      if (adaptive_beam)
        *adaptive_beam = std::min(config_.beam,
                                  ans - best_weight + config_.beam_delta);
      return ans;
    }
  }
}

Lattice2BiglmFasterDecoder::Token* Lattice2BiglmFasterDecoder::ExpandShadowTokensSub(StateId ilabel, 
    StateId new_hclg_state, StateId new_lm_state, int32 frame, 
    int32 new_frame_index, BaseFloat tot_cost, BaseFloat extra_cost, BaseFloat backward_cost,
    bool is_last) {
  Token *&toks = ilabel ? active_toks_[frame+1].toks : active_toks_[frame].toks;
  assert(toks);

  Token *tok_found=NULL;
  PairId new_pair = ConstructPair(new_hclg_state, new_lm_state);
  auto& next_h = GetBackfillMap(new_frame_index);
  auto& next_q = GetExpandQueue(new_frame_index);
  auto iter = next_h.find(new_pair);
  if (iter != next_h.end()) 
    tok_found = iter->second;

  Token* new_tok;
  bool update_tok=false;
  if (!tok_found) {  // A new token.
    // Construct the new token.
    new_tok = new Token(tot_cost, extra_cost, NULL, toks, new_lm_state,
                        new_hclg_state, backward_cost);
    toks = new_tok;
    num_toks_++;

    // Add the new token to "backfill" map.
    next_h[new_pair] = new_tok;
    update_tok=true;
  } else {  // An existing token
    new_tok = tok_found;
    if (new_tok->tot_cost > tot_cost) {
      new_tok->tot_cost = tot_cost;
      new_tok->backward_cost = backward_cost;
      new_tok->extra_cost = extra_cost;
      update_tok=true;
    }
  }

  bool better_hclg=false;
  KALDI_ASSERT(toks_backfill_hclg_.size() > new_frame_index);
  auto iter_hclg = (*toks_backfill_hclg_[new_frame_index]).find(new_hclg_state);
  if (iter_hclg != (*toks_backfill_hclg_[new_frame_index]).end()) {
    if (tot_cost < iter_hclg->second->tot_cost) 
      better_hclg=true; // search: "Update toks_shadowing_mod for better_hclg" 
      // although it is better hclg, we still keep its shadowing token for expanding in the next iter
  } else {
    (*toks_backfill_hclg_[new_frame_index])[new_hclg_state] = new_tok;
    iter_hclg = (*toks_backfill_hclg_[new_frame_index]).find(new_hclg_state);
  }

 
  if (update_tok && !new_tok->in_queue) {
    new_tok->shadowing_tok = iter_hclg->second; // by default
    if (new_tok->shadowing_tok == new_tok) { 
      // if new_tok is the shadowing token
      // search the comments above regarding to:
      // "we need to update a shadowing token itself"
      new_tok->shadowing_tok = NULL; 
    }
    if (is_last || better_hclg || new_frame_index == frame) {
      if (new_tok->shadowing_tok) { // prepare for forwardlinks updating
        // sanity check
        // KALDI_ASSERT(!new_tok->shadowing_tok->shadowing_tok || new_tok->shadowing_tok->shadowing_tok != new_tok);
        new_tok->DeleteForwardLinks();
      } 
      next_q.push(QElem(new_tok, better_hclg));
      new_tok->in_queue=true;
    }
  }
  return new_tok;
}
void Lattice2BiglmFasterDecoder::ProcessEmitting(DecodableInterface *decodable,
                                                 int32 frame) {
  Timer timer;
  // Processes emitting arcs for one frame.  Propagates from prev_toks_ to cur_toks_.
  HashList<StateId, Token*> &toks_shadowing_check=toks_shadowing_[(frame-1)%2];
  HashList<StateId, Token*> &toks_shadowing_mod=toks_shadowing_[frame%2];
  DeleteElemsShadow(toks_shadowing_mod);

  Elem *last_toks = toks_.Clear(); // swapping prev_toks_ / cur_toks_
  Elem *best_elem = NULL;
  BaseFloat adaptive_beam;
  size_t tok_cnt;
  BaseFloat cur_cutoff = GetCutoff(last_toks, &tok_cnt, &adaptive_beam, &best_elem);
  PossiblyResizeHash(tok_cnt);  // This makes sure the hash is always big enough.
  // TODO PossiblyResizeHash for toks_shadowing_
  KALDI_VLOG(6) << "Adaptive beam on frame " << frame << "\t" << NumFramesDecoded() << " is "
                << adaptive_beam << "\t" << cur_cutoff;
  if (cutoff_.Dim()<=frame) {
    cutoff_.Resize(frame+1,kCopyData);
    cutoff_.Data()[frame]=cur_cutoff;
  }

  
  BaseFloat next_cutoff = std::numeric_limits<BaseFloat>::infinity();
  // pruning "online" before having seen all tokens

  // First process the best token to get a hopefully
  // reasonably tight bound on the next cutoff.
  if (best_elem) {
    PairId state_pair = best_elem->key;
    StateId state = PairToState(state_pair), // state in "fst"
         lm_state = PairToLmState(state_pair);
    Token *tok = best_elem->val;
    for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
         !aiter.Done();
         aiter.Next()) {
      Arc arc = aiter.Value();
      if (arc.ilabel != 0) {  // propagate..
        PropagateLm(lm_state ,&arc); // may affect "arc.weight".
        // We don't need the return value (the new LM state).
        arc.weight = Times(arc.weight,
                           Weight(-decodable->LogLikelihood(frame-1, arc.ilabel)));
        BaseFloat new_weight = arc.weight.Value() + tok->tot_cost;
        if (new_weight + adaptive_beam < next_cutoff)
          next_cutoff = new_weight + adaptive_beam;
      }
    }
  }
    
  // the tokens are now owned here, in last_toks, and the hash is empty.
  // 'owned' is a complex thing here; the point is we need to call DeleteElem
  // on each elem 'e' to let toks_ know we're done with them.
  //
  // Process all previous frame tokens. Only the tokens which are stored in
  // toks_shadowing_check will move forward. Others will wait to expand.
  for (Elem *e = last_toks, *e_tail; e != NULL; e = e_tail) {
    // loop this way because we delete "e" as we go.
    PairId state_pair = e->key;
    StateId state = PairToState(state_pair),
         lm_state = PairToLmState(state_pair);
    Token *tok = e->val;
    if (tok->tot_cost <  cur_cutoff) {
      ElemShadow *elem = toks_shadowing_check.Find(state);
      assert(elem);
      if (elem->val == tok || // explore
         !tok->shadowing_tok ||
         *tok < *tok->shadowing_tok) { // it is generated by better_hclg; otherwise tok->shadowing_tok should be set by pne in the last frame
        tok->shadowing_tok = NULL;
        for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
             !aiter.Done();
             aiter.Next()) {
          const Arc &arc_ref = aiter.Value();
          if (arc_ref.ilabel != 0) {  // propagate..
            Arc arc(arc_ref);
            BaseFloat graph_cost_ori = arc.weight.Value();
            StateId next_lm_state = PropagateLm(lm_state, &arc);
            BaseFloat ac_cost = -decodable->LogLikelihood(frame-1, arc.ilabel),
                   graph_cost = arc.weight.Value(),
                     cur_cost = tok->tot_cost,
                     tot_cost = cur_cost + ac_cost + graph_cost;
            if (tot_cost > next_cutoff) continue;
            else if (tot_cost + config_.beam < next_cutoff)
              next_cutoff = tot_cost + config_.beam; // prune by best current token

            PairId next_pair = ConstructPair(arc.nextstate, next_lm_state);
            Token *next_tok = FindOrAddToken(next_pair, frame, tot_cost, true, NULL);
            // true: emitting, NULL: no change indicator needed
            ElemShadow *elem = toks_shadowing_mod.Find(arc.nextstate);
            if (elem) {
              if ((*elem->val) > *next_tok) 
                elem->val = next_tok; // update it
            } else {
              toks_shadowing_mod.Insert(arc.nextstate, next_tok);
            }

            // Add ForwardLink from tok to next_tok (put on head of list tok->links)
            tok->links = new ForwardLink(next_tok, arc.ilabel, arc.olabel, 
                                         graph_cost, ac_cost, tok->links, graph_cost_ori);
          }
        } // for all arcs
        KALDI_ASSERT(tok->shadowing_tok == NULL); // it's shadowing token
      } else {
        KALDI_ASSERT(tok->shadowing_tok && *tok > *tok->shadowing_tok);
      }
    }
    e_tail = e->tail;
    toks_.Delete(e); // delete Elem
  }
  ta_+=timer.Elapsed();
}


void Lattice2BiglmFasterDecoder::ProcessNonemitting(int32 frame) {
  Timer timer;
  // note: "frame" is the same as emitting states just processed.
  
  // Processes nonemitting arcs for one frame.  Propagates within toks_.
  // Note-- this queue structure is is not very optimal as
  // it may cause us to process states unnecessarily (e.g. more than once),
  // but in the baseline code, turning this vector into a set to fix this
  // problem did not improve overall speed.

  KALDI_ASSERT(queue_.empty());
  HashList<StateId, Token*> &toks_shadowing_check=toks_shadowing_[frame%2];
  HashList<StateId, Token*> &toks_shadowing_mod=toks_shadowing_[frame%2];

  BaseFloat best_cost = std::numeric_limits<BaseFloat>::infinity();
  for (const Elem *e = toks_.GetList(); e != NULL;  e = e->tail) {
    queue_.push_back(e->key);
    // for pruning with current best token
    best_cost = std::min(best_cost, static_cast<BaseFloat>(e->val->tot_cost));
  }
  if (queue_.empty()) {
    if (!warned_) {
      KALDI_ERR << "Error in ProcessNonemitting: no surviving tokens: frame is "
                << frame;
      warned_ = true;
    }
  }
  BaseFloat cutoff = best_cost + config_.beam;
    
  while (!queue_.empty()) {
    PairId state_pair = queue_.back();
    queue_.pop_back();
    Token *tok = toks_.Find(state_pair)->val;  // would segfault if state not in
                                               // toks_ but this can't happen.
    BaseFloat cur_cost = tok->tot_cost;
    if (cur_cost > cutoff) // Don't bother processing successors.
      continue;
    StateId state = PairToState(state_pair),
         lm_state = PairToLmState(state_pair);
    // If "tok" has any existing forward links, delete them,
    // because we're about to regenerate them.  This is a kind
    // of non-optimality (remember, this is the simple decoder),
    // but since most states are emitting it's not a huge issue.
    ElemShadow *elem = toks_shadowing_check.Find(state);
    assert(elem);
    if (elem->val == tok) { // Explore the best token in certain HCLG state
      tok->DeleteForwardLinks(); // necessary when re-visiting
      tok->links = NULL;
      for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
          !aiter.Done();
          aiter.Next()) {
        const Arc &arc_ref = aiter.Value();
        if (arc_ref.ilabel == 0) {  // propagate nonemitting only...
          Arc arc(arc_ref);
          BaseFloat graph_cost_ori = arc.weight.Value();
          StateId next_lm_state = PropagateLm(lm_state, &arc);
          BaseFloat graph_cost = arc.weight.Value(),
              tot_cost = cur_cost + graph_cost;
          if (tot_cost < cutoff) {
            bool changed;
            PairId next_pair = ConstructPair(arc.nextstate, next_lm_state);
            Token *new_tok = FindOrAddToken(next_pair, frame, tot_cost,
                                            false, &changed); // false: non-emit
            ElemShadow *elem = toks_shadowing_mod.Find(arc.nextstate);
            if (elem) {
              if ((*elem->val) > *new_tok) 
                elem->val = new_tok; // update it
            } else {
              toks_shadowing_mod.Insert(arc.nextstate, new_tok);
            }
            tok->links = new ForwardLink(new_tok, 0, arc.olabel,
                                         graph_cost, 0, tok->links, graph_cost_ori);
              
            // "changed" tells us whether the new token has a different
            // cost from before, or is new [if so, add into queue].
            if (changed) queue_.push_back(next_pair);
          }
        }
      }  // for all arcs
    } // if
  }  // while queue not empty

  // Make the "shadowing_tok" pointer point to the best one with the same
  // HCLG state id.
  for (const Elem *e = toks_.GetList(); e != NULL;  e = e->tail) {
    Token *cur_tok = e->val;
    // toks_shadowing_mod stores the best record in each HCLG state
    ElemShadow *elem = toks_shadowing_mod.Find(cur_tok->hclg_state);
    assert(elem);
    if (cur_tok == elem->val){
      cur_tok->shadowing_tok = NULL;
    } else {
      cur_tok->shadowing_tok = elem->val;
      // sanity check
      KALDI_ASSERT(!cur_tok->shadowing_tok->shadowing_tok || cur_tok->shadowing_tok->shadowing_tok != cur_tok);
      cur_tok->extra_cost=std::numeric_limits<BaseFloat>::infinity();
      cur_tok->DeleteForwardLinks(); // since some tok could be shadowed after exploring in the same decoding step
    }
  }
  BuildHCLGMapFromHash(frame); // do it here to make it consistent
  tb_+=timer.Elapsed();
}

void Lattice2BiglmFasterDecoder::BuildHCLGMapFromHash(int32 frame, bool append) {
  if (!append) KALDI_ASSERT(toks_backfill_hclg_.size() > frame);
  HashList<StateId, Token*> &toks_shadowing_mod=toks_shadowing_[frame%2];
  StateHash *hclg_map =
  new StateHash();
  hclg_map->reserve(toks_shadowing_mod.Size());

  for (const ElemShadow *e = toks_shadowing_mod.GetList(); e != NULL;  e = e->tail) {
    (*hclg_map)[e->key] = e->val;
  }

  if (append) {
    toks_backfill_hclg_.push_back(hclg_map);
  } else {
    std::swap(toks_backfill_hclg_[frame], hclg_map);
    delete hclg_map;
  }

  // sanity check
  // for (auto i:(*hclg_map)) {
  //   KALDI_ASSERT(!i.second->shadowing_tok);
  //   //i.second->links is possible to be NULL since it is possible hasnt been pruned
  // }
}
void Lattice2BiglmFasterDecoder::InitDecoding() {
  for (int i=0; i<2; i++) {
    toks_backfill_pair_[i].clear();
    expand_current_frame_queue_[i] = std::queue<QElem>();
  }
  expanding_ = false;
  // clean up from last time.
  DeleteElems(toks_.Clear());
  for (int i = 0; i < 2; i++) DeleteElemsShadow(toks_shadowing_[i]);
  ClearActiveTokens();

  cutoff_.Resize(1);
  cutoff_.Data()[0] = std::numeric_limits<int32>::max();

  // clean up private members
  warned_noarc_ = false;
  warned_ = false;
  final_active_ = false;
  final_costs_.clear();
  num_toks_ = 0;

  // At the beginning of an utterance, initialize.
  toks_backfill_hclg_.resize(0);
  PairId start_pair = ConstructPair(fst_.Start(), lm_diff_fst_->Start());
  active_toks_.resize(1);
  Token *start_tok = new Token(0.0, 0.0, NULL, NULL, lm_diff_fst_->Start(), fst_.Start());
  active_toks_[0].toks = start_tok;
  toks_.Insert(start_pair, start_tok);
  toks_shadowing_[NumFramesDecoded()%2].Insert(fst_.Start(), start_tok);
  num_toks_++;
  propage_lm_num_=0;
  propage_lm_expand_num_=0;
  ProcessNonemitting(0);
}
void Lattice2BiglmFasterDecoder::BuildBackfillMap(int32 frame, int32 frame_stop_expand, bool clear) {

  PairHash *pair_map = &GetBackfillMap(frame);
  std::queue<QElem>& q = GetExpandQueue(frame);
  if (clear) 
    pair_map->clear();

  BaseFloat cur_cutoff = (frame+1 < cutoff_.Dim())?
cutoff_(frame+1) : std::numeric_limits<BaseFloat>::infinity();

  for(Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
    if (tok->tot_cost > cur_cutoff) {
      tok->shadowing_tok=NULL;
      tok->in_queue=false;
      continue;
    }
    PairId cur_pair_id = ConstructPair(tok->hclg_state, tok->lm_state);

    bool ok = pair_map->insert({cur_pair_id, tok}).second;
    if (frame <= frame_stop_expand) { // need to expand
      if (ok) { // without this tok before
        if (tok->shadowing_tok) {
          q.push(QElem(tok, false));
          tok->in_queue=true;
        } else tok->in_queue = false;
      } else KALDI_ASSERT(tok->in_queue); // tok has been pushed by ExpandShadowTokens
    }
  }
}

}
