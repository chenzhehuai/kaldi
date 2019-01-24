// decoder/lattice2-biglm-faster-decoder.h

// Copyright      2018  Johns Hopkins University (Author: Daniel Povey)
//                      Hang Lyu

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
    toks_backfill_pair_.resize(0);
    toks_backfill_hclg_.resize(0);
  }


bool Lattice2BiglmFasterDecoder::Decode(DecodableInterface *decodable) {
  // clean up from last time.
  DeleteElems(toks_.Clear());
  for (int i = 0; i < 2; i++) DeleteElemsShadow(toks_shadowing_[i]);
  ClearActiveTokens();

  // clean up private members
  warned_noarc_ = false;
  warned_ = false;
  final_active_ = false;
  final_costs_.clear();
  num_toks_ = 0;

  // At the beginning of an utterance, initialize.
  toks_backfill_pair_.resize(0);
  toks_backfill_hclg_.resize(0);
  PairId start_pair = ConstructPair(fst_.Start(), lm_diff_fst_->Start());
  active_toks_.resize(1);
  Token *start_tok = new Token(0.0, 0.0, NULL, NULL, lm_diff_fst_->Start(), fst_.Start());
  active_toks_[0].toks = start_tok;
  toks_.Insert(start_pair, start_tok);
  toks_shadowing_[NumFramesDecoded()%2].Insert(fst_.Start(), start_tok);
  num_toks_++;
  ProcessNonemitting(0);
    
  // We use 1-based indexing for frames in this decoder (if you view it in
  // terms of features), but note that the decodable object uses zero-based
  // numbering, which we have to correct for when we call it.
  for (int32 frame = 1; !decodable->IsLastFrame(frame-2); frame++) {
    active_toks_.resize(frame+1); // new column

    ProcessEmitting(decodable, frame);

    ProcessNonemitting(frame);

    // Update the backward-cost of each token
    if (frame % 5 == 0) UpdateBackwardCost(frame, config_.lattice_beam * 0.1);

    //if (decodable->IsLastFrame(frame-1))
    //  PruneActiveTokensFinal(frame);
    //else if (frame % config_.prune_interval == 0)
    //  PruneActiveTokens(frame, config_.lattice_beam * 0.1); // use larger delta.
    if (frame % config_.prune_interval == 0)
      PruneActiveTokens(frame, config_.lattice_beam * 0.1); // use larger delta.
    

    // We could add another config option to decide the gap between state passing
    // and lm passing.
    if (frame-config_.prune_interval >= 0) ExpandShadowTokens(frame-config_.prune_interval);
  }

  // Process the last few frames lm passing
  for (int32 frame = std::max(0, NumFramesDecoded() - config_.prune_interval + 1); // final expand
       frame < NumFramesDecoded() + 1; frame++) {
    if (frame % 5 == 0) UpdateBackwardCost(frame, config_.lattice_beam * 0.1);
    ExpandShadowTokens(frame);
  }
    
  PruneActiveTokensFinal(NumFramesDecoded());

  // Returns true if we have any kind of traceback available (not necessarily
  // to the end state; query ReachedFinal() for that).
  return !final_costs_.empty();
}

bool Lattice2BiglmFasterDecoder::Decode(DecodableInterface *decodable,
                                        const Vector<BaseFloat> &cutoff) {
  // clean up from last time.
  DeleteElems(toks_.Clear());
  for (int i = 0; i < 2; i++) DeleteElemsShadow(toks_shadowing_[i]);
  ClearActiveTokens();

  // initial cutoff_
  cutoff_.Resize(cutoff.Dim());
  cutoff_ = cutoff;

  // clean up private members
  warned_noarc_ = false;
  warned_ = false;
  final_active_ = false;
  final_costs_.clear();
  num_toks_ = 0;

  // At the beginning of an utterance, initialize.
  toks_backfill_pair_.resize(0);
  toks_backfill_hclg_.resize(0);
  PairId start_pair = ConstructPair(fst_.Start(), lm_diff_fst_->Start());
  active_toks_.resize(1);
  Token *start_tok = new Token(0.0, 0.0, NULL, NULL, lm_diff_fst_->Start(), fst_.Start());
  active_toks_[0].toks = start_tok;
  toks_.Insert(start_pair, start_tok);
  toks_shadowing_[NumFramesDecoded()%2].Insert(fst_.Start(), start_tok);
  num_toks_++;
  ProcessNonemitting(0);
    
  // We use 1-based indexing for frames in this decoder (if you view it in
  // terms of features), but note that the decodable object uses zero-based
  // numbering, which we have to correct for when we call it.
  for (int32 frame = 1; !decodable->IsLastFrame(frame-2); frame++) {
    active_toks_.resize(frame+1); // new column

    ProcessEmitting(decodable, frame);

    ProcessNonemitting(frame);

    // Update the backward-cost of each token
    if (frame % 5 == 0) UpdateBackwardCost(frame, config_.lattice_beam * 0.1);

    //if (decodable->IsLastFrame(frame-1))
    //  PruneActiveTokensFinal(frame);
    //else if (frame % config_.prune_interval == 0)
    //  PruneActiveTokens(frame, config_.lattice_beam * 0.1); // use larger delta.
    if (frame % config_.prune_interval == 0)
      PruneActiveTokens(frame, config_.lattice_beam * 0.1); // use larger delta.
    

    // We could add another config option to decide the gap between state passing
    // and lm passing.
    if (frame-config_.prune_interval >= 0) ExpandShadowTokens(frame-config_.prune_interval);
  }

  // Process the last few frames lm passing
  for (int32 frame = std::max(0, NumFramesDecoded() - config_.prune_interval + 1); // final expand
       frame < NumFramesDecoded() + 1; frame++) {
    if (frame % 5 == 0) UpdateBackwardCost(frame, config_.lattice_beam * 0.1);
    ExpandShadowTokens(frame);
  }
    
  PruneActiveTokensFinal(NumFramesDecoded());

  // Returns true if we have any kind of traceback available (not necessarily
  // to the end state; query ReachedFinal() for that).
  return !final_costs_.empty();
}



void Lattice2BiglmFasterDecoder::ExpandShadowTokens(int32 frame) {
  assert(frame >= 0);
    
  BuildBackfillMap(frame);
  if ( (frame + 1) <= active_toks_.size()) {
    BuildBackfillMap(frame+1);
  }

  if (active_toks_[frame].toks == NULL) {
    KALDI_WARN << "ExpandShadowTokens: no tokens active on frame " << frame;
  }

  BaseFloat cur_cutoff = cutoff_(frame);

  /*
  // Find the minimum backward cost in this frame
  BaseFloat best_backward_cost = std::numeric_limits<BaseFloat>::infinity();
  BaseFloat worst_backward_cost = std::numeric_limits<BaseFloat>::lowest();
  for (Token* tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
    best_backward_cost = std::min(best_backward_cost, tok->backward_cost);
    if (tok->backward_cost != std::numeric_limits<BaseFloat>::infinity())
      worst_backward_cost = std::max(worst_backward_cost, tok->backward_cost);
  }
  std::cout << "In frame " << frame
            << " Best backward cost is " << best_backward_cost
            << " Worst backward cost is " << worst_backward_cost << std::endl;
  */
  /*
  BaseFloat best_forward_cost = std::numeric_limits<BaseFloat>::infinity();
  BaseFloat worst_forward_cost = std::numeric_limits<BaseFloat>::lowest();
  for (Token* tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
    best_forward_cost = std::min(best_forward_cost, tok->tot_cost);
    worst_forward_cost = std::max(worst_forward_cost, tok->tot_cost);
  }
  std::cout << "In frame " << frame
            << " Best forward cost is " << best_forward_cost
            << " Worst forward cost is " << worst_forward_cost << std::endl;
  */

  /*
  BaseFloat best_fb_cost = std::numeric_limits<BaseFloat>::infinity();
  for (Token* tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
    best_fb_cost = std::min(best_fb_cost, tok->tot_cost + tok->backward_cost);
  }
  */

  /*
    std::cout << "Show pair map." << std::endl;
    for(unordered_map<PairId, Token*>::iterator iter = toks_backfill_pair_[frame]->begin();
        iter != toks_backfill_pair_[frame]->end(); iter++) {
      Token* tok = iter->second;
      if (tok->shadowing_tok) {
        std::cout << "Token is (" << tok->hclg_state << "," << tok->lm_state << "). And Shadowing token is ("
                << tok->shadowing_tok->hclg_state << "," << tok->shadowing_tok->lm_state << ")" << std::endl;
      } else {
        std::cout << "Token is (" << tok->hclg_state << "," << tok->lm_state << ")" << std::endl;
      }
    }
  */

  // When we expand, if the arc.ilabel == 0, there maybe a new token is
  // created in current frame. The queue "expand_current_frame_queue" is used
  // to deal with all tokens in current frame in loop.
  std::queue<PairId> expand_current_frame_queue;
  // Initialize the "expand_current_frame_queue" with current tokens.
  for (Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
    if (!tok->shadowing_tok) continue; // shadowing token
                                       // If shadowing_tok == NULL, it means
                                       // this token has been processed. Or it
                                       // is the best one in this HCLG state
                                       // in the frame. Skip it.
    // Decide which token should be expanded
    if (tok->tot_cost <= cur_cutoff) {
      expand_current_frame_queue.push(ConstructPair(tok->hclg_state,
                                                  tok->lm_state));
    }
  }

  while (!expand_current_frame_queue.empty()) {
    PairId cur_id = expand_current_frame_queue.front();
    expand_current_frame_queue.pop();

    KALDI_ASSERT(toks_backfill_pair_[frame]->find(cur_id) !=
                 toks_backfill_pair_[frame]->end());
  
    Token* tok = (*toks_backfill_pair_[frame])[cur_id];
    //if (tok->tot_cost > best_forward_cost + config_.expand_beam) continue;
    if (tok->tot_cost > cur_cutoff)
      continue;

    Token* shadowing_tok = tok->shadowing_tok;

    /*
      std::cout << "Expanding: Frame " << frame
                << " cur_token HCLG_id is " << tok->hclg_state
                << " LM_id is " << tok->lm_state
                << " .And shadowing_token HCLG_id is " << shadowing_tok->hclg_state
                << " LM_id is " << shadowing_tok->lm_state << std::endl;
    */
    for (ForwardLink *link = shadowing_tok->links; link != NULL; link = link->next) {
      Token *next_tok = link->next_tok;
      
      Arc arc(link->ilabel, link->olabel, link->graph_cost_ori, 0);
      StateId new_hclg_state = next_tok->hclg_state;
      StateId new_lm_state = PropagateLm(tok->lm_state, &arc); // may affect "arc.weight".
      PairId new_pair = ConstructPair(new_hclg_state, new_lm_state);
      BaseFloat ac_cost = link->acoustic_cost,
                graph_cost = arc.weight.Value(),
                cur_cost = tok->tot_cost,
                tot_cost = cur_cost + ac_cost + graph_cost;
      // The extra_cost and backward_cost are temporary. They are inherited from
      // "next_tok" which is the destation of "shadowing_token". So they are
      // estimated rather than exact. They will be used to initialize a new
      // token and help to decide the new token will be expanded or not, as the
      // backward_cost value will be updated periodly rather than frame-by-frame.
      BaseFloat extra_cost = next_tok->extra_cost,
                backward_cost = next_tok->backward_cost;

      // prepare to store a new token in the current / next frame
      int32 new_frame_index = link->ilabel ? frame+1 : frame; 
      Token *&toks = link->ilabel ? active_toks_[frame+1].toks : active_toks_[frame].toks;
      assert(toks);

      /*
        std::cout << "ilabel is " << link->ilabel
                  << " Next token HCLG_id is " << next_tok->hclg_state
                  << " LM_id is " << next_tok->lm_state
                  << " extra_cost is " << next_tok->extra_cost
                  << " New token HCLG_id is " << new_hclg_state
                  << " LM_id is " << new_lm_state
                  << " extra_cost is " << extra_cost << std::endl;
     */    

      bool exist_flag = false;
      if (toks_backfill_pair_[new_frame_index]->find(new_pair) !=
          toks_backfill_pair_[new_frame_index]->end()) {
        exist_flag = true;
      }

      // Special case: An arc that we expand in backfill reaches an existing
      // state but it gives that state a better forward cost than before.
      /*
      if (exist_flag && tot_cost <
          (*toks_backfill_pair_[new_frame_index])[new_pair]->tot_cost) {
        // Update the destination token
        ProcessBetterExistingToken(new_frame_index, new_pair, tot_cost);
      }
      */
      if (exist_flag && tot_cost <
          (*toks_backfill_pair_[new_frame_index])[new_pair]->tot_cost) {
        // Update the destination token
        (*toks_backfill_pair_[new_frame_index])[new_pair]->tot_cost = tot_cost;
      }

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
      Token* new_tok;
      if (!exist_flag) {  // A new token.
        // Construct the new token.
        new_tok = new Token(tot_cost, extra_cost, NULL, toks, new_lm_state,
                            new_hclg_state, backward_cost);
        toks = new_tok;
        num_toks_++;

        // Add the new token to "backfill" map.
        (*toks_backfill_pair_[new_frame_index])[new_pair] = new_tok;

        new_tok->shadowing_tok =
          (*toks_backfill_hclg_[new_frame_index])[new_hclg_state];

        // Still a shadowed token. Push into queue.
        if (link->ilabel == 0 && new_tok->shadowing_tok) {
          expand_current_frame_queue.push(new_pair);
        }
      } else {  // An existing token
        new_tok = (*toks_backfill_pair_[new_frame_index])[new_pair];
      }

      // create lattice arc
      tok->links = new ForwardLink(new_tok, arc.ilabel, arc.olabel, 
                                   graph_cost, ac_cost, tok->links,
                                   link->graph_cost_ori);


      
      // Special case: A previously unseen state was created that has a higher
      // probability than an existing copy of the same HCLG.fst state.
      /*
      if (tot_cost <
          (*toks_backfill_hclg_[new_frame_index])[new_hclg_state]->tot_cost) {
        ProcessBetterHCLGToken(new_frame_index, new_tok);
        // The token has been processed along previous-best-HCLG token
        new_tok->shadowing_tok = NULL;
      } else {
        if (!exist_flag) {
          new_tok->shadowing_tok =
            (*toks_backfill_hclg_[new_frame_index])[new_hclg_state];
        }
      }
      */
    }  // end of for loop
    tok->shadowing_tok = NULL;  // already expand
  }

  // Clean the backfill map
  toks_backfill_pair_[frame]->clear();
  toks_backfill_hclg_[frame]->clear();
    
  /*
    std::cout << "CHeck the shadowing token in ExpandShadowTokens() on frame "
              << frame << std::endl;
    for (int32 frame_idx = 0; frame_idx <= active_toks_.size(); frame_idx++) {
      for (Token *tok = active_toks_[frame_idx].toks; tok != NULL; tok = tok->next) {
        if (tok->shadowing_tok != NULL) {  // shadowed token
          for (ForwardLink *link = tok->links; link != NULL;
              link = link->next) {
            if (link->ilabel != 0) {
              std::cout << "The bug is in frame " << frame_idx << std::endl;
              std::cout << "The error token HCLG_id is " << tok->hclg_state
                        << " LM_id is " << tok->lm_state
                        << " .And shadowing token HCLG_id is " << tok->shadowing_tok->hclg_state
                        << " LM_id is " << tok->shadowing_tok->lm_state << std::endl;
            }
            KALDI_ASSERT(link->ilabel == 0);
          }
        } else {  // exploring token

        }
      }
    }
  */
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
                                                   BaseFloat delta) {
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
      for (link = tok->links; link != NULL; ) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        BaseFloat link_extra_cost = 0.0;
        if (next_tok->shadowing_tok) {
          link_extra_cost = next_tok->shadowing_tok->extra_cost +
            ((tok->tot_cost + link->acoustic_cost + link->graph_cost)
             - next_tok->shadowing_tok->tot_cost);
        } else {
          link_extra_cost = next_tok->extra_cost +
            ((tok->tot_cost + link->acoustic_cost + link->graph_cost)
             - next_tok->tot_cost); // difference in brackets is >= 0
        }
        /*
          Token *next_tok = link->next_tok;
          if (next_tok->shadowing_tok) {  // excise this link depend on the
                                          // extra_cost of the shadowing token
            if (next_tok->shadowing_tok->extra_cost ==
                std::numeric_limits<BaseFloat>::infinity()) {  // The next_tok
                                                               // will be deleted
              ForwardLink *next_link = link->next;
              if (prev_link != NULL) prev_link->next = link->next;
              else tok->links = link->next;
              delete link;
              link = next_link; // advance link but leave prev_link the same.
              *links_pruned = true;
              continue;
            } else {
              prev_link = link; // move to next link
              link = link->next;
              continue;
            }
          }
          
          BaseFloat link_extra_cost = next_tok->extra_cost +
              ((tok->tot_cost + link->acoustic_cost + link->graph_cost)
               - next_tok->tot_cost); // difference in brackets is >= 0
        */
        // link_exta_cost is the difference in score between the best paths
        // through link source state and through link destination state
        KALDI_ASSERT(link_extra_cost == link_extra_cost); // check for NaN
        if (link_extra_cost > config_.lattice_beam) { // excise link
          ForwardLink *next_link = link->next;
          if (prev_link != NULL) prev_link->next = next_link;
          else tok->links = next_link;
          delete link;
          link = next_link; // advance link but leave prev_link the same.
          *links_pruned = true;
        } else { // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) { // this is just a precaution.
            if (link_extra_cost < -0.01)
              KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
            link_extra_cost = 0.0;
          }
          if (link_extra_cost < tok_extra_cost)
            tok_extra_cost = link_extra_cost;
          prev_link = link; // move to next link
          link = link->next;
        }
      } // for all outgoing links
      if (fabs(tok_extra_cost - tok->extra_cost) > delta)
        changed = true;  // difference new minus old is bigger than delta
      tok->extra_cost = tok_extra_cost;
      // will be +infinity or <= lattice_beam_.
      // infinity indicates, that no forward link survived pruning
    } // for all Token on active_toks_[frame]
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
        if (next_tok->shadowing_tok) { // TODO
          prev_link = link; // move to next link
          link = link->next;
          continue;
        }
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
              KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
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


void Lattice2BiglmFasterDecoder::PruneTokensForFrame(int32 frame) {
  KALDI_ASSERT(frame >= 0 && frame < active_toks_.size());
  Token *&toks = active_toks_[frame].toks;
  if (toks == NULL)
    KALDI_WARN << "No tokens alive [doing pruning]\n";
  Token *tok, *next_tok, *prev_tok = NULL;
  // proc shadowed token at first as it needs info from shadowing token
  for (tok = toks; tok != NULL; tok = next_tok) {
    next_tok = tok->next;
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
        tok->extra_cost = tok->shadowing_tok->extra_cost +
                          tok->shadowing_tok->tot_cost - tok->tot_cost;
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
      // Check
      for (Token *tok = active_toks_[frame+1].toks; tok != NULL; tok = tok->next) {
        KALDI_ASSERT(tok->extra_cost != std::numeric_limits<BaseFloat>::infinity());
      }
      active_toks_[frame+1].must_prune_tokens = false;
    }
  }
  KALDI_VLOG(3) << "PruneActiveTokens: pruned tokens from " << num_toks_begin
                << " to " << num_toks_;
}


void Lattice2BiglmFasterDecoder::PruneActiveTokensFinal(int32 cur_frame) {
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
    PruneForwardLinks(frame, &b1, &b2, dontcare);
    PruneTokensForFrame(frame+1);
  }
  PruneTokensForFrame(0); 
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


void Lattice2BiglmFasterDecoder::ProcessEmitting(DecodableInterface *decodable,
                                                 int32 frame) {
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
        PropagateLm(lm_state, &arc); // may affect "arc.weight".
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
    if (tok->tot_cost <=  cur_cutoff) {
      ElemShadow *elem = toks_shadowing_check.Find(state);
      assert(elem);
      if (elem->val == tok) { // explore
        KALDI_ASSERT(tok->shadowing_tok == NULL);
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
            if (elem && elem->val->tot_cost > tot_cost) {
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
        KALDI_ASSERT(tok->shadowing_tok != NULL);
      }
    }
    e_tail = e->tail;
    toks_.Delete(e); // delete Elem
  }
}


void Lattice2BiglmFasterDecoder::ProcessNonemitting(int32 frame) {
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
            if (elem && elem->val->tot_cost > tot_cost) {
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
    }  // end of if condition
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
    }
  }
}


void Lattice2BiglmFasterDecoder::BuildBackfillMap(int32 frame) {
  KALDI_ASSERT(toks_backfill_pair_.size() == toks_backfill_hclg_.size());
  // We have already construct the map when we previous special case.
  if (toks_backfill_pair_.size() != frame ||
      active_toks_.size() == toks_backfill_pair_.size()) { return; }
  if (active_toks_[frame].toks == NULL) {
    KALDI_WARN << "BuildBackfillMap: no tokens active on frame " << frame;
  }

  // Initialize
  std::unordered_map<PairId, Token*> *pair_map =
    new std::unordered_map<PairId, Token*>();
  std::unordered_map<StateId, Token*> *hclg_map =
    new std::unordered_map<StateId, Token*>();

  /*
    for(Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
      PairId cur_pair_id = ConstructPair(tok->hclg_state, tok->lm_state);
      StateId cur_hclg_id = tok->hclg_state;

      if (pair_map->find(cur_pair_id) == pair_map->end()) {
        (*pair_map)[cur_pair_id] = tok;
      }
      if (hclg_map->find(cur_hclg_id) != hclg_map->end()) {  // already exist
        if (tok->tot_cost < (*hclg_map)[cur_hclg_id]->tot_cost) {  // better
          (*hclg_map)[cur_hclg_id] = tok;
        }
      } else {  // new
        (*hclg_map)[cur_hclg_id] = tok;
      }
    }

    toks_backfill_pair_.push_back(pair_map);
    toks_backfill_hclg_.push_back(hclg_map);
  */

  for (Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
    const PairId cur_pair_id = ConstructPair(tok->hclg_state, tok->lm_state);
    const StateId cur_hclg_id = tok->hclg_state;

    if (pair_map->find(cur_pair_id) == pair_map->end()) {
      (*pair_map)[cur_pair_id] = tok;
    }
    // Check this token is an explored token or not
    bool is_explored_tok = false;
    for(ForwardLink *link = tok->links; link != NULL; link = link->next) {
      if (link->ilabel != 0) {
        is_explored_tok = true;
        break;
      }
    }
    if (is_explored_tok) {
      KALDI_ASSERT(hclg_map->find(cur_hclg_id) == hclg_map->end());
      (*hclg_map)[cur_hclg_id] = tok;
    }
  }

  toks_backfill_pair_.push_back(pair_map);
  toks_backfill_hclg_.push_back(hclg_map);
}


void Lattice2BiglmFasterDecoder::ProcessBetterExistingToken(
  int32 cur_frame,
  PairId new_pair_id,
  BaseFloat new_tot_cost) {
  if (cur_frame > active_toks_.size() - 1) {  // have already expand to newest
                                              // frame
    return;
  }
    

  BuildBackfillMap(cur_frame);

  KALDI_ASSERT((*toks_backfill_pair_[cur_frame])[new_pair_id]);

  Token *ori_tok = (*toks_backfill_pair_[cur_frame])[new_pair_id];
  BaseFloat diff = ori_tok->tot_cost - new_tot_cost;
  if (diff <= 0) {
    return;
  }

  // Update the token
  ori_tok->tot_cost = new_tot_cost;
  BaseFloat new_extra_cost = ori_tok->extra_cost - diff;
  ori_tok->extra_cost = new_extra_cost ? new_extra_cost : 0;
  // The "ori_tok" could be a shadowed token or exploring token.
  // If shadowing token is NULL, it means the token has already been expanded.
  // So we only need to expand the change of tot_cost along each arc.
  // If shadowing token isn't NULL, it means the token is shadowed. We will
  // expand it. In this circumstance, the "links" maybe empty or not, which
  // can be caused by ProcessNonemitting() [The token was the best token in
  // certain HCLG state--"A" on this frame. But the processing of other token
  // create a better token in this HCLG state--"A", so that both of the two
  // tokens did the non-emit expand. The former one is shadowed token with
  // link, the latter one was processed in "exploration" step.]
  if (ori_tok->shadowing_tok == NULL) {  // The token was expanded
    // Expand the change of tot_cost along each arc.
    for (ForwardLink *link = ori_tok->links; link != NULL;
         link = link->next) {
      Token *next_tok = link->next_tok;
      int32 new_frame_index = link->ilabel ? cur_frame + 1 : cur_frame;
      PairId next_pair_id = ConstructPair(next_tok->hclg_state,
                                          next_tok->lm_state);
      BaseFloat new_tot_cost = next_tok->tot_cost - diff;
      ProcessBetterExistingToken(new_frame_index, next_pair_id, new_tot_cost);
    }
    // Check the token. It is the new best HCLG token or not
    if (ori_tok->tot_cost <
        (*toks_backfill_hclg_[cur_frame])[ori_tok->hclg_state]->tot_cost) {
      // Get the Best HCLG token
      Token* pre_best_hclg_tok =
        (*toks_backfill_hclg_[cur_frame])[ori_tok->hclg_state];

      // Update the hclg list
      (*toks_backfill_hclg_[cur_frame])[ori_tok->hclg_state] = ori_tok;

      // Update the exploring list
      if (cur_frame == active_toks_.size() - 1) {
        HashList<StateId, Token*> &toks_shadowing_mod =
          toks_shadowing_[cur_frame % 2];
        StateId state = PairToState(new_pair_id);
        ElemShadow *elem = toks_shadowing_mod.Find(state);
        KALDI_ASSERT(elem);
        KALDI_ASSERT(elem->val == pre_best_hclg_tok);
        elem->val = ori_tok;
        pre_best_hclg_tok->shadowing_tok = ori_tok;
      }
    }
  } else {  // The token hasn't been expanded
    if (ori_tok->links == NULL) {
      // If the token is the new best HCLG token, go on expanding.
      if (ori_tok->tot_cost <
          (*toks_backfill_hclg_[cur_frame])[ori_tok->hclg_state]->tot_cost) {
        // Inside, the ori_tok->shadowing_tok will be set to NULL.
        ProcessBetterHCLGToken(cur_frame, ori_tok);
      }
    } else {  // An unexpanded token with links
      // Handle the links
      for (ForwardLink *link = ori_tok->links; link != NULL;
           link = link->next) {
        Token *next_tok = link->next_tok;
        KALDI_ASSERT(link->ilabel == 0);  // These links should be generated
                                          // from ProcessNonemitting().
        PairId next_pair_id = ConstructPair(next_tok->hclg_state,
                                            next_tok->lm_state);
        BaseFloat new_tot_cost = next_tok->tot_cost - diff;
        ProcessBetterExistingToken(cur_frame, next_pair_id,
                                   new_tot_cost);
      }
      // Check the token.
      if (ori_tok->tot_cost <
          (*toks_backfill_hclg_[cur_frame])[ori_tok->hclg_state]->tot_cost) {
        // Inside, the ori_tok->shadowing_tok will be set to NULL.
        ProcessBetterHCLGToken(cur_frame, ori_tok);
      }
    }
  }
}


void Lattice2BiglmFasterDecoder::ProcessBetterHCLGToken(int32 cur_frame,
                                                        Token *better_token) {
  if (cur_frame > active_toks_.size() - 1) {  // have already expand to newest
                                              // frame
    return;
  }

  BuildBackfillMap(cur_frame);
  BuildBackfillMap(cur_frame+1);

  // Get previous best one
  Token *pre_best_hclg_tok =
    (*toks_backfill_hclg_[cur_frame])[better_token->hclg_state];

  if (better_token == pre_best_hclg_tok) {
    return;
  }
  // iterator all links in previous best token
  for (ForwardLink *link = pre_best_hclg_tok->links; link != NULL;
       link = link->next) {
    Token *next_tok = link->next_tok;
    Arc arc(link->ilabel, link->olabel, link->graph_cost_ori, 0);
    StateId new_hclg_state = next_tok->hclg_state;
    StateId new_lm_state = PropagateLm(better_token->lm_state, &arc); // may affect "arc.weight".
    PairId new_pair = ConstructPair(new_hclg_state, new_lm_state);
    BaseFloat ac_cost = link->acoustic_cost,
              graph_cost = arc.weight.Value(),
              cur_cost = better_token->tot_cost,
              tot_cost = cur_cost + ac_cost + graph_cost;
    // Use the extra_cost of pre_best_hlcg_tok to prune the new token
    const BaseFloat ref_extra_cost = next_tok->extra_cost;
    BaseFloat extra_cost = ref_extra_cost +
                           (tot_cost - next_tok->tot_cost);
    if (extra_cost > config_.beam) {  // skip this link
      continue;
    }
    // prepare to store a new token in the current / next frame
    int32 new_frame_index = link->ilabel ? cur_frame+1 : cur_frame; 
    Token *&toks = active_toks_[new_frame_index].toks;
    assert(toks);
      
    bool exist_flag = false;
    if (toks_backfill_pair_[new_frame_index]->find(new_pair) !=
        toks_backfill_pair_[new_frame_index]->end()) {
      exist_flag = true;
    }
    // Special case: An arc that we expand in backfill reaches an existing
    // state，but it gives that state a better forward cost than before.
    if (exist_flag && tot_cost <
        (*toks_backfill_pair_[new_frame_index])[new_pair]->tot_cost) {
      // Update the destination token
      ProcessBetterExistingToken(new_frame_index, new_pair, tot_cost);
    }
    Token* new_tok;
    if (!exist_flag) {  // A new token.
      // Construct the new token.
      new_tok = new Token(tot_cost, extra_cost, NULL, toks, new_lm_state,
                          new_hclg_state);
      new_tok->shadowing_tok =
        (*toks_backfill_hclg_[new_frame_index])[new_hclg_state];
      toks = new_tok;
      num_toks_++;
      // Add the new token to "backfill" map.
      (*toks_backfill_pair_[new_frame_index])[new_pair] = new_tok;
    } else {  // An existing token
      new_tok = (*toks_backfill_pair_[new_frame_index])[new_pair];
    }
    // create lattice arc
    better_token->links = new ForwardLink(new_tok, arc.ilabel, arc.olabel, 
                                          graph_cost, ac_cost, better_token->links,
                                          link->graph_cost_ori);

    // Special case: A previously unseen state was created that has a higher
    // probability than an existing copy of the same HCLG.fst state.
    if (tot_cost <
      (*toks_backfill_hclg_[new_frame_index])[new_hclg_state]->tot_cost) {
        ProcessBetterHCLGToken(new_frame_index, new_tok);
    } else {
      if (!exist_flag) {
        new_tok->shadowing_tok =
          (*toks_backfill_hclg_[new_frame_index])[new_hclg_state];
      }
    }
  }  // end of for loop
  // As the expanding process may cause "best_hclg_tok" change, so we
  // reacquire it.
  pre_best_hclg_tok =
    (*toks_backfill_hclg_[cur_frame])[better_token->hclg_state];

  // Update the exploring list
  if (cur_frame == active_toks_.size() - 1) {
    HashList<StateId, Token*> &toks_shadowing_mod =
      toks_shadowing_[cur_frame % 2];
    StateId state = better_token->hclg_state;
    ElemShadow *elem = toks_shadowing_mod.Find(state);
    KALDI_ASSERT(elem);
    KALDI_ASSERT(elem->val == pre_best_hclg_tok);
    elem->val = better_token;
    pre_best_hclg_tok->shadowing_tok = better_token;
  }

  // Update
  (*toks_backfill_hclg_[cur_frame])[better_token->hclg_state] = better_token;
  better_token->shadowing_tok = NULL;  // already expand
}


void Lattice2BiglmFasterDecoder::UpdateBackwardCost(int32 cur_frame,
                                                    BaseFloat delta) {
  // Set the backward-cost of current frame tokens to zero
  for(Token* tok = active_toks_[cur_frame].toks; tok != NULL; tok = tok->next) {
    tok->backward_cost = 0;
  }

  for (int32 frame = cur_frame-1;
       frame >= 0 && frame > cur_frame - config_.prune_interval; frame--) {
    if (active_toks_[frame].toks == NULL ) { // empty list; should not happen.
      KALDI_WARN << "No tokens alive when compute backward cost\n";
    }
  
    // We have to iterate until there is no more change, because the links
    // are not guaranteed to be in topological order.
    bool changed = true; // difference new minus old extra cost >= delta ?
    while (changed) {
      changed = false;
      for (Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
        ForwardLink *link;
        // will recompute tok_extra_cost for tok.
        BaseFloat tok_backward_cost = std::numeric_limits<BaseFloat>::infinity();
        // tok_backward_cost is the best (min) of link_extra_cost of outgoing links
        for (link = tok->links; link != NULL; link = link->next) {
          Token *next_tok = link->next_tok;
          BaseFloat link_backward_cost =
            std::numeric_limits<BaseFloat>::infinity();
          if (next_tok->shadowing_tok) {
            link_backward_cost = next_tok->shadowing_tok->backward_cost + 
                                 link->acoustic_cost + link->graph_cost;
          } else {
            link_backward_cost = next_tok->backward_cost + link->acoustic_cost +
                                 link->graph_cost;
          }
          KALDI_ASSERT(link_backward_cost == link_backward_cost); // check for NaN
          tok_backward_cost = std::min(tok_backward_cost, link_backward_cost);
        }
        if (fabs(tok_backward_cost - tok->backward_cost) > delta)
          changed = true;  // difference new minus old is bigger than delta
        tok->backward_cost = tok_backward_cost;
      } // for all Token on active_toks_[frame]
    }
  }
  // Set the backward-cost of current frame tokens back to infinity
  for (Token* tok = active_toks_[cur_frame].toks; tok != NULL; tok = tok->next) {
    tok->backward_cost = std::numeric_limits<BaseFloat>::infinity();
  }
} 

}
