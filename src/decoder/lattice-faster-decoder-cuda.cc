// decoder/lattice-faster-decoder-cuda.cc

// Copyright      2018  Zhehuai Chen

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

#include <nvToolsExt.h>
#include "base/timer.h"
#include "cuda-lattice-decoder.h"
#include "decoder/lattice-faster-decoder-cuda.h"
#include "lat/lattice-functions.h"

namespace kaldi {



// instantiate this class once for each thing you have to decode.
LatticeFasterDecoderCuda::LatticeFasterDecoderCuda(const CudaFst &fst,
                                           const CudaLatticeDecoderConfig &config):
    fst_(fst), delete_fst_(false), config_(config), num_toks_(0), 
    decoder_(fst, config_){
  //toks_.SetSize(1000);  // just so on the first frame we do something reasonable.
  //decoder_.InitDecoding();
  //InitDecoding();
}

LatticeFasterDecoderCuda::~LatticeFasterDecoderCuda() {
  //DeleteElems(toks_.Clear());
  ClearActiveTokens();
  if (delete_fst_) delete &(fst_);
}

void LatticeFasterDecoderCuda::InitDecoding() {
  // clean up from last time:
  //DeleteElems(toks_.Clear());
  cost_offsets_.clear();
  ClearActiveTokens();
  warned_ = false;
  num_toks_ = 0;
  decoding_finalized_ = false;
  final_costs_.clear();
  StateId start_state = fst_.Start();
  KALDI_ASSERT(start_state != fst::kNoStateId);
  num_frames_decoded_=0;
  //change to init in ProcessLattices()
  //active_toks_.resize(1);
  //Token *start_tok = new Token(0.0, 0.0, NULL, NULL);
  //active_toks_[0].toks = start_tok;

  //toks_.Insert(start_state, start_tok);
  //num_toks_++;
  //ProcessNonemittingWrapper(config_.beam);
}
//inline
void LatticeFasterDecoderCuda::CreateTokAndRegister(BaseFloat cost, 
  Token *&toks) {
    Token *new_tok = new Token (cost, 0, NULL, toks);
    toks = new_tok; //add into active_toks_;
}
void LatticeFasterDecoderCuda::dbg(cuToken *i) {
    active_toks_map_[i] = active_toks_[num_frames_decoded_].toks;
}
int LatticeFasterDecoderCuda::AddLatticeArcs(cuTokenVector& cur_toks_,
      LatLinkVector*& cur_arcs_) {
  //proc t-1,t-1; t-1,t; leave t,t and t,t+1 in the next call
  //copy lat_arcs_sub_vec_ to lat_arcs_vec_
  const CudaFst& cu_fst = decoder_.fst_;
  int num_arcs=0;
  for (int i = 0; i < config_.sub_vec_num; i++) {
    for ( int j = 0; j < cur_arcs_[i].size(); j++) {
      LatLink& arc_d_h = cur_arcs_[i][j];
      assert(arc_d_h.arc_id<cu_fst.arc_count);
      BaseFloat graph_cost=cu_fst.arc_weights_h[arc_d_h.arc_id];
      int32 ilabel=cu_fst.arc_ilabels_h[arc_d_h.arc_id];
      int32 olabel=cu_fst.arc_olabels_h[arc_d_h.arc_id];
      Token* next_tok=active_toks_map_.at(arc_d_h.next_tok);
      Token* prev_tok=active_toks_map_.at(arc_d_h.prev_tok);
      prev_tok->links = new ForwardLink(next_tok, ilabel, olabel, graph_cost, 
                                        arc_d_h.acoustic_cost, prev_tok->links);
      num_arcs++;
    }
  }
  return num_arcs;
}
BaseFloat LatticeFasterDecoderCuda::get_cost(int i) {
  return (*cur_toks_)[i].token->cost_;
}
LatticeFasterDecoderCuda::cuToken*  LatticeFasterDecoderCuda::get_cutok(int i) {
  return (*cur_toks_)[i].token;
}
void LatticeFasterDecoderCuda::ProcessLattices(cuTokenVector& cur_toks_,
  cuTokenVector& prev_toks_, LatLinkVector*& cur_arcs_) {
  //KALDI_ASSERT(num_frames_decoded_);
  //active_toks_map_.clear();

//  if (num_frames_decoded_==1) {//add prev
//    active_toks_.resize(1);
//    for (int i=0;i<prev_toks_.size();i++) { //always add into active_toks_map_, the newer key will replace the older
//      assert(prev_toks_[i].token);
//      CreateTokAndRegister(*prev_toks_[i].token, active_toks_[num_frames_decoded_-1].toks);
//      num_toks_++;
//    }    
//  }
  //add current
  active_toks_.resize(active_toks_.size() + 1);
  for (int i=0;i<cur_toks_.size();i++) { //always add into active_toks_map_, the newer key will replace the older
    assert(get_cutok(i));
    CreateTokAndRegister(get_cost(i), active_toks_[num_frames_decoded_].toks);
    dbg(get_cutok(i));
    num_toks_++;
  }
  //ERR: proc t-1,t-1; t-1,t; leave t,t and t,t+1 in the next call
  //TODO: change to preproc 0,0; proc t-1,t and t,t
  int num_arcs=0, num_arcs2=0;
  num_arcs+=AddLatticeArcs(prev_toks_, cur_arcs_);
  for (int i=0; i<config_.sub_vec_num; i++)  num_arcs2+=cur_arcs_[i].size();
  assert(num_arcs==num_arcs2);
  //call prune
  if (NumFramesDecoded() % config_.prune_interval == 0)
    PruneActiveTokens(config_.lattice_beam * config_.prune_scale);
}
// Returns true if any kind of traceback is available (not necessarily from
// a final state).  It should only very rarely return false; this indicates
// an unusual search error.
bool LatticeFasterDecoderCuda::Decode(DecodableInterface *decodable) {
  //decoder_.Decode(decodable);

  nvtxRangePushA("CudaLatticeDecoder::Decode");

  Timer timer;
  timer.Reset();

  InitDecoding();
  decoder_.InitDecoding();
  decoder_.PreProcessLattices(&cur_toks_, &prev_toks_, &cur_arcs_);
  ProcessLattices(*cur_toks_, *prev_toks_, cur_arcs_);

  decoder_.ComputeLogLikelihoods(decodable);
  num_frames_decoded_++;

  double pre_time = timer.Elapsed();

  double cpu_proc_time=0, gpu_proc_time=0;
  while( !decodable->IsLastFrame(NumFramesDecoded() - 1)) {
    double t1 = timer.Elapsed();
    decoder_.PreProcessTokens();
    decoder_.ProcessTokens();
    decoder_.PreProcessLattices(&cur_toks_, &prev_toks_, &cur_arcs_);
    double t2 = timer.Elapsed();
    ProcessLattices(*cur_toks_, *prev_toks_, cur_arcs_);
    double t3 = timer.Elapsed();
    //active_toks_.resize(active_toks_.size()+1);
    decoder_.PostProcessTokens();
    if (decodable->IsLastFrame(NumFramesDecoded() - 1)) break;
    //computes log likelihoods for the next frame
    decoder_.ComputeLogLikelihoods(decodable);
    num_frames_decoded_++;
    double t4 = timer.Elapsed();
    
    cpu_proc_time += t3-t2;
    gpu_proc_time += t2-t1+t4-t3;
  }

  nvtxRangePop();

  double t5 = timer.Elapsed();
  decoder_.PreFinalizeDecoding();
  FinalizeDecoding();
  double t6 = timer.Elapsed();
  double cpu_proc_time_f = t6-t5;
  KALDI_VLOG(3)<<"pre_time,cpu_proc_time,cpu_proc_time_f,gpu_proc_time: "<<pre_time<<" "<<cpu_proc_time<<" "<<cpu_proc_time_f<<" "<<gpu_proc_time;
  // Returns true if we have any kind of traceback available (not necessarily
  // to the end state; query ReachedFinal() for that).
  return !active_toks_.empty() && active_toks_.back().toks != NULL;
}

// FinalizeDecoding() is a version of PruneActiveTokens that we call
// (optionally) on the final frame.  Takes into account the final-prob of
// tokens.  This function used to be called PruneActiveTokensFinal().
void LatticeFasterDecoderCuda::FinalizeDecoding() {
  int32 final_frame_plus_one = NumFramesDecoded();
  int32 num_toks_begin = num_toks_;
  // PruneForwardLinksFinal() prunes final frame (with final-probs), and
  // sets decoding_finalized_.
  PruneForwardLinksFinal();
  for (int32 f = final_frame_plus_one - 1; f >= 0; f--) {
    bool b1, b2; // values not used.
    BaseFloat dontcare = 0.0; // delta of zero means we must always update
    PruneForwardLinks(f, &b1, &b2, dontcare);
    PruneTokensForFrame(f + 1);
  }
  PruneTokensForFrame(0);
  KALDI_VLOG(4) << "pruned tokens from " << num_toks_begin
                << " to " << num_toks_;
}



// Outputs an FST corresponding to the single best path through the lattice.
bool LatticeFasterDecoderCuda::GetBestPath(Lattice *olat,
                                       bool use_final_probs) const {
  Lattice raw_lat;
  GetRawLattice(&raw_lat, use_final_probs);
  ShortestPath(raw_lat, olat);
  //decoder_.GetBestPath(best_path, true);
  return (olat->NumStates() != 0);
}

// Outputs an FST corresponding to the raw, state-level
// tracebacks. :
bool LatticeFasterDecoderCuda::GetRawLattice(Lattice *ofst,
                                         bool use_final_probs) const {
  typedef LatticeArc Arc;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;
  typedef Arc::Label Label;

  // Note: you can't use the old interface (Decode()) if you want to
  // get the lattice with use_final_probs = false.  You'd have to do
  // InitDecoding() and then AdvanceDecoding().
  if (decoding_finalized_ && !use_final_probs)
    KALDI_ERR << "You cannot call FinalizeDecoding() and then call "
              << "GetRawLattice() with use_final_probs == false";

  unordered_map<Token*, BaseFloat> final_costs_local;

  const unordered_map<Token*, BaseFloat> &final_costs =
      (decoding_finalized_ ? final_costs_ : final_costs_local);
  if (!decoding_finalized_ && use_final_probs)
    ComputeFinalCosts(&final_costs_local, NULL, NULL);

  ofst->DeleteStates();
  // num-frames plus one (since frames are one-based, and we have
  // an extra frame for the start-state).
  int32 num_frames = active_toks_.size() - 1;
  KALDI_ASSERT(num_frames > 0);
  const int32 bucket_count = num_toks_/2 + 3;
  unordered_map<Token*, StateId> tok_map(bucket_count);
  // First create all states.
  std::vector<Token*> token_list;
  for (int32 f = 0; f <= num_frames; f++) {
    if (active_toks_[f].toks == NULL) {
      KALDI_WARN << "GetRawLattice: no tokens active on frame " << f
                 << ": not producing lattice.\n";
      return false;
    }
    TopSortTokens(active_toks_[f].toks, &token_list);
    for (size_t i = 0; i < token_list.size(); i++)
      if (token_list[i] != NULL)
        tok_map[token_list[i]] = ofst->AddState();
  }
  // The next statement sets the start state of the output FST.  Because we
  // topologically sorted the tokens, state zero must be the start-state.
  ofst->SetStart(0);

  KALDI_VLOG(4) << "init:" << num_toks_/2 + 3 << " buckets:"
                << tok_map.bucket_count() << " load:" << tok_map.load_factor()
                << " max:" << tok_map.max_load_factor();
  // Now create all arcs.
  for (int32 f = 0; f <= num_frames; f++) {
    for (Token *tok = active_toks_[f].toks; tok != NULL; tok = tok->next) {
      StateId cur_state = tok_map[tok];
      for (ForwardLink *l = tok->links;
           l != NULL;
           l = l->next) {
        unordered_map<Token*, StateId>::const_iterator iter =
            tok_map.find(l->next_tok);
        StateId nextstate = iter->second;
        KALDI_ASSERT(iter != tok_map.end());
        BaseFloat cost_offset = 0.0;
        if (l->ilabel != 0) {  // emitting..
          //KALDI_ASSERT(f >= 0 && f < cost_offsets_.size());
          cost_offset = 0; //cost_offsets_[f];
        }
        Arc arc(l->ilabel, l->olabel,
                Weight(l->graph_cost, l->acoustic_cost - cost_offset),
                nextstate);
        ofst->AddArc(cur_state, arc);
      }
      if (f == num_frames) {
        if (use_final_probs && !final_costs.empty()) {
          unordered_map<Token*, BaseFloat>::const_iterator iter =
              final_costs.find(tok);
          if (iter != final_costs.end())
            ofst->SetFinal(cur_state, LatticeWeight(iter->second, 0));
        } else {
          ofst->SetFinal(cur_state, LatticeWeight::One());
        }
      }
    }
  }
  return (ofst->NumStates() > 0);
}


// prunes outgoing links for all tokens in active_toks_[frame]
// it's called by PruneActiveTokens
// all links, that have link_extra_cost > lattice_beam are pruned
void LatticeFasterDecoderCuda::PruneForwardLinks(
    int32 frame_plus_one, bool *extra_costs_changed,
    bool *links_pruned, BaseFloat delta) {
  // delta is the amount by which the extra_costs must change
  // If delta is larger,  we'll tend to go back less far
  //    toward the beginning of the file.
  // extra_costs_changed is set to true if extra_cost was changed for any token
  // links_pruned is set to true if any link in any token was pruned

  *extra_costs_changed = false;
  *links_pruned = false;
  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_toks_.size());
  if (active_toks_[frame_plus_one].toks == NULL) {  // empty list; should not happen.
    if (!warned_) {
      KALDI_WARN << "No tokens alive [doing pruning].. warning first "
          "time only for each utterance\n";
      warned_ = true;
    }
  }

  // We have to iterate until there is no more change, because the links
  // are not guaranteed to be in topological order.
  bool changed = true;  // difference new minus old extra cost >= delta ?
  while (changed) {
    changed = false;
    int extra_flag=0;
    int extra_flag2=0;
    for (Token *tok = active_toks_[frame_plus_one].toks;
         tok != NULL; tok = tok->next) {
      ForwardLink *link, *prev_link = NULL;
      // will recompute tok_extra_cost for tok.
      BaseFloat tok_extra_cost = std::numeric_limits<BaseFloat>::infinity();
      // tok_extra_cost is the best (min) of link_extra_cost of outgoing links
      for (link = tok->links; link != NULL; ) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        BaseFloat link_extra_cost = next_tok->extra_cost +
            ((tok->tot_cost + link->acoustic_cost + link->graph_cost)
             - next_tok->tot_cost);  // difference in brackets is >= 0
        // link_exta_cost is the difference in score between the best paths
        // through link source state and through link destination state
        if (link_extra_cost == 0) extra_flag = 1;
        KALDI_ASSERT(link_extra_cost == link_extra_cost);  // check for NaN
        if (link_extra_cost > config_.lattice_beam) {  // excise link
          ForwardLink *next_link = link->next;
          if (prev_link != NULL) prev_link->next = next_link;
          else tok->links = next_link;
          delete link;
          link = next_link;  // advance link but leave prev_link the same.
          *links_pruned = true;
        } else {   // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) {  // this is just a precaution.
            if (link_extra_cost < -0.01)
              KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
            link_extra_cost = 0.0;
          }
          if (link_extra_cost < tok_extra_cost) {
            tok_extra_cost = link_extra_cost;
          }
          prev_link = link;  // move to next link
          link = link->next;
        }
      }  // for all outgoing links
      if (fabs(tok_extra_cost - tok->extra_cost) > delta)
        changed = true;   // difference new minus old is bigger than delta
      tok->extra_cost = tok_extra_cost;
      if (tok_extra_cost==0) extra_flag2=1;
      // will be +infinity or <= lattice_beam_.
      // infinity indicates, that no forward link survived pruning
    }  // for all Token on active_toks_[frame]
    if (!extra_flag) KALDI_VLOG(6)<<frame_plus_one<<" no link_extra_cost==0";
    if (!extra_flag2) KALDI_VLOG(6)<<frame_plus_one<<" no tok_extra_cost==0";
    if (changed) *extra_costs_changed = true;

    // Note: it's theoretically possible that aggressive compiler
    // optimizations could cause an infinite loop here for small delta and
    // high-dynamic-range scores.
  } // while changed
}

// PruneForwardLinksFinal is a version of PruneForwardLinks that we call
// on the final frame.  If there are final tokens active, it uses
// the final-probs for pruning, otherwise it treats all tokens as final.
void LatticeFasterDecoderCuda::PruneForwardLinksFinal() {
  KALDI_ASSERT(!active_toks_.empty());
  int32 frame_plus_one = active_toks_.size() - 1;

  if (active_toks_[frame_plus_one].toks == NULL)  // empty list; should not happen.
    KALDI_WARN << "No tokens alive at end of file";

  typedef unordered_map<Token*, BaseFloat>::const_iterator IterType;
  ComputeFinalCosts(&final_costs_, &final_relative_cost_, &final_best_cost_);
  decoding_finalized_ = true;
  // We call DeleteElems() as a nicety, not because it's really necessary;
  // otherwise there would be a time, after calling PruneTokensForFrame() on the
  // final frame, when toks_.GetList() or toks_.Clear() would contain pointers
  // to nonexistent tokens.
  //DeleteElems(toks_.Clear());

  // Now go through tokens on this frame, pruning forward links...  may have to
  // iterate a few times until there is no more change, because the list is not
  // in topological order.  This is a modified version of the code in
  // PruneForwardLinks, but here we also take account of the final-probs.
  bool changed = true;
  BaseFloat delta = 1.0e-05;
  while (changed) {
    changed = false;
    for (Token *tok = active_toks_[frame_plus_one].toks;
         tok != NULL; tok = tok->next) {
      ForwardLink *link, *prev_link = NULL;
      // will recompute tok_extra_cost.  It has a term in it that corresponds
      // to the "final-prob", so instead of initializing tok_extra_cost to infinity
      // below we set it to the difference between the (score+final_prob) of this token,
      // and the best such (score+final_prob).
      BaseFloat final_cost;
      if (final_costs_.empty()) {
        final_cost = 0.0;
      } else {
        IterType iter = final_costs_.find(tok);
        if (iter != final_costs_.end())
          final_cost = iter->second;
        else
          final_cost = std::numeric_limits<BaseFloat>::infinity();
      }
      BaseFloat tok_extra_cost = tok->tot_cost + final_cost - final_best_cost_;
      // tok_extra_cost will be a "min" over either directly being final, or
      // being indirectly final through other links, and the loop below may
      // decrease its value:
      for (link = tok->links; link != NULL; ) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        BaseFloat link_extra_cost = next_tok->extra_cost +
            ((tok->tot_cost + link->acoustic_cost + link->graph_cost)
             - next_tok->tot_cost);
        if (link_extra_cost > config_.lattice_beam) {  // excise link
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
        tok_extra_cost = std::numeric_limits<BaseFloat>::infinity();
      // to be pruned in PruneTokensForFrame

      if (!ApproxEqual(tok->extra_cost, tok_extra_cost, delta))
        changed = true;
      tok->extra_cost = tok_extra_cost; // will be +infinity or <= lattice_beam_.
    }
  } // while changed
}

BaseFloat LatticeFasterDecoderCuda::FinalRelativeCost() const {
  if (!decoding_finalized_) {
    BaseFloat relative_cost;
    ComputeFinalCosts(NULL, &relative_cost, NULL);
    return relative_cost;
  } else {
    // we're not allowed to call that function if FinalizeDecoding() has
    // been called; return a cached value.
    return final_relative_cost_;
  }
}


// Prune away any tokens on this frame that have no forward links.
// [we don't do this in PruneForwardLinks because it would give us
// a problem with dangling pointers].
// It's called by PruneActiveTokens if any forward links have been pruned
void LatticeFasterDecoderCuda::PruneTokensForFrame(int32 frame_plus_one) {
  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_toks_.size());
  int num_toks_s=num_toks_;
  Token *&toks = active_toks_[frame_plus_one].toks;
  if (toks == NULL)
    KALDI_WARN << "frame: "<<frame_plus_one<< " No tokens alive [doing pruning]";
  Token *tok, *next_tok, *prev_tok = NULL;
  for (tok = toks; tok != NULL; tok = next_tok) {
    next_tok = tok->next;
    if (tok->extra_cost == std::numeric_limits<BaseFloat>::infinity()) {
      // token is unreachable from end of graph; (no forward links survived)
      // excise tok from list and delete tok.
      if (prev_tok != NULL) prev_tok->next = tok->next;
      else toks = tok->next;
      delete tok;
      num_toks_--;
    } else {  // fetch next Token
      prev_tok = tok;
    }
  }
  KALDI_VLOG(5) << "PR: "<<frame_plus_one<<","<<num_toks_s-num_toks_;
}

// Go backwards through still-alive tokens, pruning them, starting not from
// the current frame (where we want to keep all tokens) but from the frame before
// that.  We go backwards through the frames and stop when we reach a point
// where the delta-costs are not changing (and the delta controls when we consider
// a cost to have "not changed").
void LatticeFasterDecoderCuda::PruneActiveTokens(BaseFloat delta) {
  int32 cur_frame_plus_one = NumFramesDecoded();
  int32 num_toks_begin = num_toks_;
  // The index "f" below represents a "frame plus one", i.e. you'd have to subtract
  // one to get the corresponding index for the decodable object.
  for (int32 f = cur_frame_plus_one - 1; f >= 0; f--) {
    // Reason why we need to prune forward links in this situation:
    // (1) we have never pruned them (new TokenList)
    // (2) we have not yet pruned the forward links to the next f,
    // after any of those tokens have changed their extra_cost.
    if (active_toks_[f].must_prune_forward_links) {
      bool extra_costs_changed = false, links_pruned = false;
      PruneForwardLinks(f, &extra_costs_changed, &links_pruned, delta);
      if (extra_costs_changed && f > 0) // any token has changed extra_cost
        active_toks_[f-1].must_prune_forward_links = true;
      if (links_pruned) // any link was pruned
        active_toks_[f].must_prune_tokens = true;
      active_toks_[f].must_prune_forward_links = false; // job done
    }
    if (f+1 < cur_frame_plus_one &&      // except for last f (no forward links)
        active_toks_[f+1].must_prune_tokens) {
      PruneTokensForFrame(f+1);
      active_toks_[f+1].must_prune_tokens = false;
    }
  }
  KALDI_VLOG(4) << "PruneActiveTokens: pruned tokens from " << num_toks_begin
                << " to " << num_toks_;
}

void LatticeFasterDecoderCuda::ComputeFinalCosts(
  unordered_map<Token*, BaseFloat> *final_costs,
  BaseFloat *final_relative_cost,
  BaseFloat *final_best_cost) const {
  KALDI_ASSERT(!decoding_finalized_);
//  *final_relative_cost=0;
//  *final_best_cost=0;
//  KALDI_WARN<<"unfinished here";
  if (final_costs != NULL)
    final_costs->clear();
  BaseFloat infinity = std::numeric_limits<BaseFloat>::infinity();
  BaseFloat best_cost = infinity, best_cost_with_final = infinity;

  cuTokenVector& cur_toks=*this->cur_toks_;
  for (int i=0;i<cur_toks.size();i++) {
    cuToken* tok_d_h = cur_toks[i].token;
    assert(active_toks_map_.count(tok_d_h));
    StateId state = cur_toks[i].state;
    Token* tok = active_toks_map_.at(tok_d_h);

    BaseFloat final_cost = fst_.Final(state);
    BaseFloat cost = tok->tot_cost,
        cost_with_final = cost + final_cost;
    best_cost = std::min(cost, best_cost);
    best_cost_with_final = std::min(cost_with_final, best_cost_with_final);
    if (final_costs != NULL && final_cost != infinity)
      (*final_costs)[tok] = final_cost;    
  }

  if (final_relative_cost != NULL) {
    if (best_cost == infinity && best_cost_with_final == infinity) {
      // Likely this will only happen if there are no tokens surviving.
      // This seems the least bad way to handle it.
      *final_relative_cost = infinity;
    } else {
      *final_relative_cost = best_cost_with_final - best_cost;
    }
  }
  if (final_best_cost != NULL) {
    if (best_cost_with_final != infinity) { // final-state exists.
      *final_best_cost = best_cost_with_final;
    } else { // no final-state exists.
      *final_best_cost = best_cost;
    }
  }
}



void LatticeFasterDecoderCuda::ClearActiveTokens() { // a cleanup routine, at utt end/begin
  for (size_t i = 0; i < active_toks_.size(); i++) {
    // Delete all tokens alive on this frame, and any forward
    // links they may have.
    for (Token *tok = active_toks_[i].toks; tok != NULL; ) {
      tok->DeleteForwardLinks();
      Token *next_tok = tok->next;
      delete tok;
      num_toks_--;
      tok = next_tok;
    }
  }
  active_toks_.clear();
  active_toks_map_.clear();
  KALDI_ASSERT(num_toks_ == 0);
}

// static
void LatticeFasterDecoderCuda::TopSortTokens(Token *tok_list,
                                         std::vector<Token*> *topsorted_list) {
  unordered_map<Token*, int32> token2pos;
  typedef unordered_map<Token*, int32>::iterator IterType;
  int32 num_toks = 0;
  for (Token *tok = tok_list; tok != NULL; tok = tok->next)
    num_toks++;
  int32 cur_pos = 0;
  // We assign the tokens numbers num_toks - 1, ... , 2, 1, 0.
  // This is likely to be in closer to topological order than
  // if we had given them ascending order, because of the way
  // new tokens are put at the front of the list.
  for (Token *tok = tok_list; tok != NULL; tok = tok->next)
    token2pos[tok] = num_toks - ++cur_pos;

  unordered_set<Token*> reprocess;

  for (IterType iter = token2pos.begin(); iter != token2pos.end(); ++iter) {
    Token *tok = iter->first;
    int32 pos = iter->second;
    for (ForwardLink *link = tok->links; link != NULL; link = link->next) {
      if (link->ilabel == 0) {
        // We only need to consider epsilon links, since non-epsilon links
        // transition between frames and this function only needs to sort a list
        // of tokens from a single frame.
        IterType following_iter = token2pos.find(link->next_tok);
        if (following_iter != token2pos.end()) { // another token on this frame,
                                                 // so must consider it.
          int32 next_pos = following_iter->second;
          if (next_pos < pos) { // reassign the position of the next Token.
            following_iter->second = cur_pos++;
            reprocess.insert(link->next_tok);
          }
        }
      }
    }
    // In case we had previously assigned this token to be reprocessed, we can
    // erase it from that set because it's "happy now" (we just processed it).
    reprocess.erase(tok);
  }

  size_t max_loop = 1000000, loop_count; // max_loop is to detect epsilon cycles.
  for (loop_count = 0;
       !reprocess.empty() && loop_count < max_loop; ++loop_count) {
    std::vector<Token*> reprocess_vec;
    for (unordered_set<Token*>::iterator iter = reprocess.begin();
         iter != reprocess.end(); ++iter)
      reprocess_vec.push_back(*iter);
    reprocess.clear();
    for (std::vector<Token*>::iterator iter = reprocess_vec.begin();
         iter != reprocess_vec.end(); ++iter) {
      Token *tok = *iter;
      int32 pos = token2pos[tok];
      // Repeat the processing we did above (for comments, see above).
      for (ForwardLink *link = tok->links; link != NULL; link = link->next) {
        if (link->ilabel == 0) {
          IterType following_iter = token2pos.find(link->next_tok);
          if (following_iter != token2pos.end()) {
            int32 next_pos = following_iter->second;
            if (next_pos < pos) {
              following_iter->second = cur_pos++;
              reprocess.insert(link->next_tok);
            }
          }
        }
      }
    }
  }
  KALDI_ASSERT(loop_count < max_loop && "Epsilon loops exist in your decoding "
               "graph (this is not allowed!)");

  topsorted_list->clear();
  topsorted_list->resize(cur_pos, NULL);  // create a list with NULLs in between.
  for (IterType iter = token2pos.begin(); iter != token2pos.end(); ++iter)
    (*topsorted_list)[iter->second] = iter->first;
}

} // end namespace kaldi.
