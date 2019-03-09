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

#ifndef KALDI_DECODER_LATTICE2_BIGLM_FASTER_DECODER_H_
#define KALDI_DECODER_LATTICE2_BIGLM_FASTER_DECODER_H_


#include "util/stl-utils.h"
#include "util/hash-list.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "decoder/lattice-faster-decoder.h" // for options.
#include "base/timer.h"

namespace kaldi {

struct Lattice2BiglmFasterDecoderConfig{
  BaseFloat beam;
  int32 max_active;
  int32 min_active;
  BaseFloat lattice_beam;
  int32 prune_interval;
  bool determinize_lattice; // not inspected by this class... used in
                            // command-line program.
  BaseFloat beam_delta; // has nothing to do with beam_ratio
  BaseFloat hash_ratio;
  BaseFloat expand_beam;
  BaseFloat prune_scale;   // Note: we don't make this configurable on the command line,
                           // it's not a very important parameter.  It affects the
                           // algorithm that prunes the tokens as we go.
  // Most of the options inside det_opts are not actually queried by the
  // LatticeFasterDecoder class itself, but by the code that calls it, for
  // example in the function DecodeUtteranceLatticeFaster.
  fst::DeterminizeLatticePhonePrunedOptions det_opts;
  int better_hclg;
  int explore_interval;

  Lattice2BiglmFasterDecoderConfig(): beam(16.0),
                                max_active(std::numeric_limits<int32>::max()),
                                min_active(200),
                                lattice_beam(10.0),
                                prune_interval(25),
                                determinize_lattice(true),
                                beam_delta(0.5),
                                hash_ratio(2.0),
                                expand_beam(16.0),
                                prune_scale(0.1),
  better_hclg(false), explore_interval(0) { }
  void Register(OptionsItf *opts) {
    det_opts.Register(opts);
    opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
    opts->Register("max-active", &max_active, "Decoder max active states.  Larger->slower; "
                   "more accurate");
    opts->Register("min-active", &min_active, "Decoder minimum #active states.");
    opts->Register("lattice-beam", &lattice_beam, "Lattice generation beam.  Larger->slower, "
                   "and deeper lattices");
    opts->Register("prune-interval", &prune_interval, "Interval (in frames) at "
                   "which to prune tokens");
    opts->Register("determinize-lattice", &determinize_lattice, "If true, "
                   "determinize the lattice (lattice-determinization, keeping only "
                   "best pdf-sequence for each word-sequence).");
    opts->Register("beam-delta", &beam_delta, "Increment used in decoding-- this "
                   "parameter is obscure and relates to a speedup in the way the "
                   "max-active constraint is applied.  Larger is more accurate.");
    opts->Register("hash-ratio", &hash_ratio, "Setting used in decoder to "
                   "control hash behavior");
    opts->Register("expand-beam", &expand_beam, "Expanding beam.");
    opts->Register("better-hclg", &better_hclg, "Expanding better HCLG states.");
    opts->Register("explore-interval", &explore_interval, "the interval between explore and expand.");
  }
  void Check() const {
    KALDI_ASSERT(beam > 0.0 && max_active > 1 && lattice_beam > 0.0
                 && prune_interval > 0 && beam_delta > 0.0 && hash_ratio >= 1.0
                 && prune_scale > 0.0 && prune_scale < 1.0);
  }
};



/** This is as LatticeFasterDecoder, but does online composition between
    HCLG and the "difference language model", which is a deterministic
    FST that represents the difference between the language model you want
    and the language model you compiled HCLG with.  The class
    DeterministicOnDemandFst follows through the epsilons in G for you
    (assuming G is a standard backoff language model) and makes it look
    like a determinized FST.
*/

class Lattice2BiglmFasterDecoder {
 public:
  typedef fst::StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  // A PairId will be constructed as: (StateId in fst) + (StateId in lm_diff_fst) << 32;
  typedef uint64 PairId;
  typedef Arc::Weight Weight;

  // instantiate this class once for each thing you have to decode.
  Lattice2BiglmFasterDecoder(
      const fst::Fst<fst::StdArc> &fst,      
      const Lattice2BiglmFasterDecoderConfig &config,
      fst::DeterministicOnDemandFst<fst::StdArc> *lm_diff_fst);

  void SetOptions(const Lattice2BiglmFasterDecoderConfig &config) { config_ = config; } 
  
  Lattice2BiglmFasterDecoderConfig GetOptions() { return config_; } 
  
  // Releases the HashList and Backfill Maps which are created by 
  // BuildBackfillMap()
  ~Lattice2BiglmFasterDecoder() {
    DeleteElems(toks_.Clear());
    for (int i = 0; i < 2; i++) DeleteElemsShadow(toks_shadowing_[i]);
    ClearActiveTokens();
    // Clean up backfill map
    for (int32 frame = NumFramesDecoded(); frame >= 0; frame--) {
      delete toks_backfill_hclg_[frame];
    }
    KALDI_VLOG(1) << "time: " << expand_time_ << " " << propage_time_<< " " << ta_ << " " << tb_;
  }

  inline int32 NumFramesDecoded() const { return active_toks_.size() - 1; }
  
  // Returns true if any kind of traceback is available (not necessarily from
  // a final state).
  bool Decode(DecodableInterface *decodable);
  bool Decode(DecodableInterface *decodable, const Vector<BaseFloat> &cutoff);

  // The core code of this method. We split this method into two stage --
  // exploration stage and backfill stage. In the exploration stage, we only
  // process the "best cost" token for each specific HCLG state, the rest tokens
  // which have the same HCLG state will be shadowed by the best one.
  // For example, we have two tokens (s, l1) and (s, l2). The previous one has
  // better cost. So we process (s, l1) in exploration stage, and (s, l2) will
  // be shadowed by (s, l1).
  // In backfill stage, we expand the shadowed tokens, namely we process (s,l2).
  // We expand it along the paths of exploration token, namely (s, l1).
  // As they stay in the same HCLG state, so the destinate HCLG state will be
  // the same. At the same time, the acoutic cost, and graph cost on HCLG.fst
  // can be borrowed.
  // We only need to propage it on Diff_LM.fst according to (lm_state, olabel)
  // In the processing of "ExpandShadowTokens", we will encounter two special
  // cases. One is reaching an existing token with better tot_cost. Another is
  // creating an new token with same HCLG state and better tot_cost. They will
  // be processed by ProcessBetterExistingToken() and ProcessBetterHCLGToken()
  // separately.
  // Otherwise, as the ilabel could be 0, so we new shadowed token maybe created
  // during the processing. We process them with a queue. This way is similiar
  // with ProcessNonemitting()
  // Furthermore, the expanding will be related to current frame and next frame.
  // For judging the token has better cost or reaching existing token, we build
  // the backfill maps.
  void ExpandShadowTokens(int32 frame, int32 frame_stop_expand, DecodableInterface *decodable, bool first=false);

  /// says whether a final-state was active on the last frame.  If it was not, the
  /// lattice (or traceback) will end with states that are not final-states.
  bool ReachedFinal() const { return final_active_; }


  // Outputs an FST corresponding to the single best path
  // through the lattice.
  bool GetBestPath(fst::MutableFst<LatticeArc> *ofst, 
                   bool use_final_probs = true) const {
    fst::VectorFst<LatticeArc> fst;
    if (!GetRawLattice(&fst, use_final_probs)) return false;
    // std::cout << "Raw lattice is:\n";
    // fst::FstPrinter<LatticeArc> fstprinter(fst, NULL, NULL, NULL, false, true);
    // fstprinter.Print(&std::cout, "standard output");
    ShortestPath(fst, ofst);
    return true;
  }

  // Outputs an FST corresponding to the raw, state-level
  // tracebacks.
  bool GetRawLattice(fst::MutableFst<LatticeArc> *ofst,
                     bool use_final_probs = true) const;

  // This function is now deprecated, since now we do determinization from
  // outside the LatticeBiglmFasterDecoder class.
  // Outputs an FST corresponding to the lattice-determinized
  // lattice (one path per word sequence).
  bool GetLattice(fst::MutableFst<CompactLatticeArc> *ofst,
                  bool use_final_probs = true) const;
 
 private:
  inline PairId ConstructPair(StateId fst_state, StateId lm_state) {
    return static_cast<PairId>(fst_state) + (static_cast<PairId>(lm_state) << 32);
  }
  
  static inline StateId PairToState(PairId state_pair) {
    return static_cast<StateId>(static_cast<uint32>(state_pair));
  }
  static inline StateId PairToLmState(PairId state_pair) {
    return static_cast<StateId>(static_cast<uint32>(state_pair >> 32));
  }
  
  struct Token;
  // ForwardLinks are the links from a token to a token on the next frame.
  // or sometimes on the current frame (for input-epsilon links).
  struct ForwardLink {
    Token *next_tok; // the next token [or NULL if represents final-state]
    Label ilabel; // ilabel on link.
    Label olabel; // olabel on link.
    BaseFloat graph_cost; // graph cost of traversing link (contains LM, etc.)
    BaseFloat acoustic_cost; // acoustic cost (pre-scaled) of traversing link
    ForwardLink *next; // next in singly-linked list of forward links from a
                       // token.
    BaseFloat graph_cost_ori;  // Record the graph cost from HCLG.fst so that
                               // we needn't revisit HCLG.fst when expanding.
    inline ForwardLink(Token *next_tok, Label ilabel, Label olabel,
                       BaseFloat graph_cost, BaseFloat acoustic_cost, 
                       ForwardLink *next, BaseFloat graph_cost_ori):
        next_tok(next_tok), ilabel(ilabel), olabel(olabel),
        graph_cost(graph_cost), acoustic_cost(acoustic_cost), 
        next(next), graph_cost_ori(graph_cost_ori) {}
  };  
  
  // Token is what's resident in a particular state at a particular time.
  // In this decoder a Token actually contains *forward* links.
  // When first created, a Token just has the (total) cost.    We add forward
  // links to it when we process the next frame.
  struct Token {
    BaseFloat tot_cost; // would equal weight.Value()... cost up to this point.
    BaseFloat extra_cost; // >= 0.  After calling PruneForwardLinks, this equals
    // the minimum difference between the cost of the best path, and the cost of
    // this is on, and the cost of the absolute best path, under the assumption
    // that any of the currently active states at the decoding front may
    // eventually succeed (e.g. if you were to take the currently active states
    // one by one and compute this difference, and then take the minimum).
    
    ForwardLink *links; // Head of singly linked list of ForwardLinks
    
    Token *next; // Next in list of tokens for this frame.
    Token *shadowing_tok;  // If it is NULL, it means the token is expanded or
                           // it is processed in exploration stage. If it isn't
                           // NULL, it points the token who shadows the token.

    // The following two states are used to record the hclg_state id and
    // lm_state id in current token. They will be used in expanding shadowed
    // tokens as the hashlist has been released at that time.
    StateId lm_state; // for expanding shadowed states
    StateId hclg_state; // for expanding shadowed states

    BaseFloat backward_cost; // backward-cost. It will be updated periodly.
                         // It will be used in Backfill stage. We will not
                         // expand all shadowing token. The shadowed token
                         // whose backward_cost < best_backward_cost + config_.beam
                         // will be expanded. In another word, if we prune the
                         // lattice on each frame rather than prune it periodly,
                         // we only expand the survived tokens after pruning.
    bool in_queue;
   
    inline Token(BaseFloat tot_cost, BaseFloat extra_cost, ForwardLink *links,
                 Token *next, StateId lm_state, StateId hclg_state):
                 tot_cost(tot_cost), extra_cost(extra_cost), links(links),
                 next(next), shadowing_tok(NULL), lm_state(lm_state),
                 hclg_state(hclg_state),
                 backward_cost(std::numeric_limits<BaseFloat>::infinity()), in_queue(0) {}

    inline Token(BaseFloat tot_cost, BaseFloat extra_cost, ForwardLink *links,
                 Token *next, StateId lm_state, StateId hclg_state,
                 BaseFloat backward_cost):
                 tot_cost(tot_cost), extra_cost(extra_cost), links(links),
                 next(next), shadowing_tok(NULL), lm_state(lm_state),
                 hclg_state(hclg_state), backward_cost(backward_cost), in_queue(0) {}


    inline void DeleteForwardLinks() {
      ForwardLink *l = links, *m; 
      while (l != NULL) {
        m = l->next;
        delete l;
        l = m;
      }
      links = NULL;
    }
    inline bool operator < (const Token &other) const {
      if (tot_cost == other.tot_cost) // this is important to garrenttee a single shadowing token
        return lm_state < other.lm_state;
      else return tot_cost < other.tot_cost;
    }
    inline bool operator > (const Token &other) const { return other < (*this); }
  };
  
  // head and tail of per-frame list of Tokens (list is in topological order),
  // and something saying whether we ever pruned it using PruneForwardLinks.
  struct TokenList {
    Token *toks;
    bool must_prune_forward_links;
    bool must_prune_tokens;
    TokenList(): toks(NULL), must_prune_forward_links(true),
                 must_prune_tokens(true) { }
  };

  typedef HashList<PairId, Token*>::Elem Elem;
  typedef HashList<StateId, Token*>::Elem ElemShadow;

  void PossiblyResizeHash(size_t num_toks) {
    size_t new_sz = static_cast<size_t>(static_cast<BaseFloat>(num_toks)
                                        * config_.hash_ratio);
    if (new_sz > toks_.Size()) {
      toks_.SetSize(new_sz);
    }
    HashList<StateId, Token*> &h = toks_shadowing_[NumFramesDecoded()%2];
    if (new_sz > h.Size()) h.SetSize(new_sz);
  }

  // FindOrAddToken either locates a token in hash of toks_,
  // or if necessary inserts a new, empty token (i.e. with no forward links)
  // for the current frame.  [note: it's inserted if necessary into hash toks_
  // and also into the singly linked list of tokens active on this frame
  // (whose head is at active_toks_[frame]).
  inline Token *FindOrAddToken(PairId state_pair, int32 frame, BaseFloat tot_cost,
                               bool emitting, bool *changed) {
    // Returns the Token pointer.  Sets "changed" (if non-NULL) to true
    // if the token was newly created or the cost changed.
    KALDI_ASSERT(frame < active_toks_.size());
    Token *&toks = active_toks_[frame].toks;
    Elem *e_found = toks_.Find(state_pair);
    if (e_found == NULL) { // no such token presently.
      const BaseFloat extra_cost = 0.0;
      // tokens on the currently final frame have zero extra_cost
      // as any of them could end up
      // on the winning path.
      Token *new_tok = new Token(tot_cost, extra_cost, NULL, toks,
                                 PairToLmState(state_pair),
                                 PairToState(state_pair));
      // NULL: no forward links yet
      toks = new_tok;
      num_toks_++;
      toks_.Insert(state_pair, new_tok);
      if (changed) *changed = true;
      return new_tok;
    } else {
      Token *tok = e_found->val; // There is an existing Token for this state.
      if (tok->tot_cost > tot_cost) { // replace old token
        tok->tot_cost = tot_cost;
        // we don't allocate a new token, the old stays linked in active_toks_
        // we only replace the tot_cost
        // in the current frame, there are no forward links (and no extra_cost)
        // only in ProcessNonemitting we have to delete forward links
        // in case we visit a state for the second time
        // those forward links, that lead to this replaced token before:
        // they remain and will hopefully be pruned later (PruneForwardLinks...)
        if (changed) *changed = true;
      } else {
        if (changed) *changed = false;
      }
      return tok;
    }
  }
  
  // prunes outgoing links for all tokens in active_toks_[frame]
  // it's called by PruneActiveTokens
  // all links, that have link_extra_cost > lattice_beam are pruned
  void PruneForwardLinks(int32 frame, bool *extra_costs_changed,
                         bool *links_pruned, BaseFloat delta, bool is_expand);

  // PruneForwardLinksFinal is a version of PruneForwardLinks that we call
  // on the final frame.  If there are final tokens active, it uses
  // the final-probs for pruning, otherwise it treats all tokens as final.
  void PruneForwardLinksFinal(int32 frame);

  // Prune away any tokens on this frame that have no forward links.
  // [we don't do this in PruneForwardLinks because it would give us
  // a problem with dangling pointers].
  // It's called by PruneActiveTokens if any forward links have been pruned
  void PruneTokensForFrame(int32 frame, bool is_expand);

  // Go backwards through still-alive tokens, pruning them.  note: cur_frame is
  // where hash toks_ are (so we do not want to mess with it because these tokens
  // don't yet have forward pointers), but we do all previous frames, unless we
  // know that we can safely ignore them because the frame after them was unchanged.
  // delta controls when it considers a cost to have changed enough to continue
  // going backward and propagating the change.
  // for a larger delta, we will recurse less far back
  void PruneActiveTokens(int32 cur_frame, BaseFloat delta);

  // Version of PruneActiveTokens that we call on the final frame.
  // Takes into account the final-prob of tokens.
  void PruneActiveTokensFinal(int32 cur_frame, bool is_expand=false);

  /// Gets the weight cutoff.  Also counts the active tokens.
  BaseFloat GetCutoff(Elem *list_head, size_t *tok_count,
                      BaseFloat *adaptive_beam, Elem **best_elem);

  // Update the graph cost according to lm_state and olabel
  inline StateId PropagateLm(StateId lm_state,
                             Arc *arc) { // returns new LM state.
    if (arc->olabel == 0) {
      return lm_state; // no change in LM state if no word crossed.
    } else { // Propagate in the LM-diff FST.
      Timer timer;
      propage_lm_num_++;
      if (expanding_) propage_lm_expand_num_++;
      Arc lm_arc;
      bool ans = lm_diff_fst_->GetArc(lm_state, arc->olabel, &lm_arc);
      propage_time_+=timer.Elapsed();
      if (!ans) { // this case is unexpected for statistical LMs.
        if (!warned_noarc_) {
          warned_noarc_ = true;
          KALDI_WARN << "No arc available in LM (unlikely to be correct "
              "if a statistical language model); will not warn again";
        }
        arc->weight = Weight::Zero();
        return lm_state; // doesn't really matter what we return here; will
        // be pruned.
      } else {
        arc->weight = Times(arc->weight, lm_arc.weight);
        arc->olabel = lm_arc.olabel; // probably will be the same.
        return lm_arc.nextstate; // return the new LM state.
      }      
    }
  }
 
  // Processes emitting arcs for one frame in exploration stage.
  void ProcessEmitting(DecodableInterface *decodable, int32 frame);

  // Processes nonemitting (epsilon) arcs for one frame in exploration stage.
  // Called after Processemitting() on each frame.
  void ProcessNonemitting(int32 frame);

  // HashList defined in ../util/hash-list.h.  It actually allows us to maintain
  // more than one list (e.g. for current and previous frames), but only one of
  // them at a time can be indexed by StateId.
  HashList<PairId, Token*> toks_;
  HashList<StateId, Token*> toks_shadowing_[2];

  // When do expanding, we have two special cases need to be processed.
  // 1. An arc that we expand in backfill reaches an existing state，but it
  // gives that state a better forward cost than before. It means (s_new, l_new)
  // is existing, we need to propagate the change to current frame.
  // 2. A previously unseen state was created that has a higher probability than
  // an existing copy of the same HCLG.fst state. It means (s_new, l_new) is
  // better than the shadowing token (s_new, l*) which is the best one in this
  // HCLG state at this time.
  // The following variables are used to check the existing tokens and best
  // token in certain frame. It will build in function ExpandShadowTokens()
  // Each element in the vector corresponds to a frame(t).
  // TODO: add comments: we only update toks_shadowing_ but not toks_backfill_hclg_
  typedef std::unordered_map<StateId, Token*,
          std::hash<StateId>, std::equal_to<StateId>,
          fst::PoolAllocator<std::pair<const StateId, Token*> > > StateHash;
  typedef std::unordered_map<PairId, Token*,
          std::hash<PairId>, std::equal_to<PairId>,
          fst::PoolAllocator<std::pair<const PairId, Token*> > > PairHash;
  PairHash toks_backfill_pair_[2];
  std::vector<StateHash* > toks_backfill_hclg_;
  typedef std::pair<Token*, int32> QElem;
  std::queue<QElem> expand_current_frame_queue_[2];
  std::queue<QElem>& GetExpandQueue(int32 frame) { return expand_current_frame_queue_[frame%2]; }
  PairHash& GetBackfillMap(int32 frame) { return toks_backfill_pair_[frame%2]; }
  void InitDecoding();

  // temp variable used to process special case. The pair is (t, state_id).
  // As we want to process the token which has smaller t index at first,
  // so we use a priority_queue
  struct PriorityCompare {
    bool operator() (const std::pair<int32, PairId>& a,
                     const std::pair<int32, PairId>& b) {
      return (a.first > b.first);
    }
  };
  std::priority_queue<std::pair<int32, PairId>,
                      std::vector<std::pair<int32, PairId> >,
                      PriorityCompare> tmp_expand_queue_;


  std::vector<TokenList> active_toks_; // Lists of tokens, indexed by
  int32 ToksNum(int32 f) {
    int32 c=0;
    for (Token *t=active_toks_[f].toks; t; t=t->next) c++;
    return c;
  }
  // frame (members of TokenList are toks, must_prune_forward_links,
  // must_prune_tokens).
  std::vector<PairId> queue_;  // temp variable used in ProcessNonemitting,
  std::vector<BaseFloat> tmp_array_;  // used in GetCutoff.
  // make it class member to avoid internal new/delete.
  const fst::Fst<fst::StdArc> &fst_;
  fst::DeterministicOnDemandFst<fst::StdArc> *lm_diff_fst_;  
  Lattice2BiglmFasterDecoderConfig config_;
  bool warned_noarc_;  
  int32 num_toks_; // current total #toks allocated...
  bool warned_;
  bool final_active_; // use this to say whether we found active final tokens
  // on the last frame.
  std::map<Token*, BaseFloat> final_costs_; // A cache of final-costs
  // of tokens on the last frame-- it's just convenient to store it this way.
  
  // It might seem unclear why we call DeleteElems(toks_.Clear()).
  // There are two separate cleanup tasks we need to do at when we start a new file.
  // one is to delete the Token objects in the list; the other is to delete
  // the Elem objects.  toks_.Clear() just clears them from the hash and gives ownership
  // to the caller, who then has to call toks_.Delete(e) for each one.  It was designed
  // this way for convenience in propagating tokens from one frame to the next.
  void DeleteElems(Elem *list) {
    for (Elem *e = list, *e_tail; e != NULL; e = e_tail) {
      e_tail = e->tail;
      toks_.Delete(e);
    }
  }
  void DeleteElemsShadow(HashList<StateId, Token*> &toks) {
    ElemShadow *list = toks.Clear();
    for (ElemShadow *e = list, *e_tail; e != NULL; e = e_tail) {
      e_tail = e->tail;
      toks.Delete(e);
    }
  }

  
  inline void ClearActiveTokens() { // a cleanup routine, at utt end/begin
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
    KALDI_ASSERT(num_toks_ == 0);
  }

  // For this frame, we create two unordered_map on heap and store them into
  // toks_backfill_pair_/toks_backfill_hclg_ separately.
  // Actually, we only build the two maps for each frame once. Otherwise, in
  // ExpandShadowTokens(), it will be increased. In PruneTokenForFrame(), it
  // will be decreased.
  void BuildBackfillMap(int32 frame, int32 frame_stop_expand, bool clear=false);
  void BuildHCLGMapFromHash(int32 frame, bool append=true);
  Token *ExpandShadowTokensSub(StateId ilabel, 
    StateId new_hclg_state, StateId new_lm_state, int32 frame, 
    int32 new_frame_index, BaseFloat tot_cost, BaseFloat extra_cost, BaseFloat backward_cost,
    bool is_last);

  Vector<BaseFloat> cutoff_;
  uint64 propage_lm_num_;
  uint64 propage_lm_expand_num_;
  bool expanding_;
  double expand_time_;
  double propage_time_;
  double ta_, tb_;
};

} // end namespace kaldi.

#endif
