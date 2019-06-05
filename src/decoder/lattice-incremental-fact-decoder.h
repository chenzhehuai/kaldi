// decoder/lattice-incremental-fact-decoder.h

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

#ifndef KALDI_DECODER_LATTICE_INCREMENTAL_FACT_DECODER_H_
#define KALDI_DECODER_LATTICE_INCREMENTAL_FACT_DECODER_H_

#include "fst/fstlib.h"
#include "fst/memory.h"
#include "fstext/fstext-lib.h"
#include "grammar-fst.h"
#include "itf/decodable-itf.h"
#include "lat/kaldi-lattice.h"
#include "lattice-incremental-decoder.h"
#include "util/stl-utils.h"
#include "lat/determinize-lattice-pruned.h"

namespace kaldi {

/* This is an extention to the "normal" lattice-generating decoder.
   See \ref lattices_generation \ref decoders_faster and \ref decoders_simple
    for more information.

   The main difference is the incremental determinization which will be
   discussed in the function GetLattice().

   The decoder is templated on the FST type and the token type.  The token type
   will normally be BackToken, but also may be BackpointerToken which is to support
   quick lookup of the current best path (see lattice-faster-online-decoder.h)

   The FST you invoke this decoder with is expected to equal
   Fst::Fst<fst::StdArc>, a.k.a. StdFst, or GrammarFst.  If you invoke it with
   FST == StdFst and it notices that the actual FST type is
   fst::VectorFst<fst::StdArc> or fst::ConstFst<fst::StdArc>, the decoder object
   will internally cast itself to one that is templated on those more specific
   types; this is an optimization for speed.
 */
template <typename FST, std::size_t kStatePerPhone,
          typename Token = decoder::BackToken<typename FST::Arc>>
class LatticeIncrementalFactDecoderTpl
    : public LatticeIncrementalDecoderTpl<FST, Token> {
 public:
  typedef uint64 PairId;
  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  // Notably, We include ac_cost and trans_cost in the acoustic_cost of BackwardLinkT;
  // we include graph_cost in graph_cost of BackwardLinkT;
  using BackwardLinkT = typename Token::BackwardLinkT;
  using base = LatticeIncrementalDecoderTpl<FST, Token>;
  using QueueElem = typename base::QueueElem;
  using Elem = typename base::Elem;
  using base::active_toks_;
  using base::best_pair;
  using base::ClearActiveTokens;
  using base::conf_;
  using base::config_;
  using base::cost_offsets_;
  using base::decoding_finalized_;
  using base::delete_fst_;
  using base::DeleteElems;
  using base::determinizer_;
  using base::emit_tok_num_;
  using base::final_costs_;
  using base::FinalizeDecoding;
  using base::FindOrAddToken;
  using base::fst_;
  using base::fst_sorted_;
  using base::GetCutoff;
  using base::GetLattice;
  using base::GetNumToksForFrame;
  using base::last_frame_nonfinal_states_;
  using base::last_get_lattice_frame_;
  using base::link_allocator_;
  using base::nemit_tok_num_;
  using base::num_toks_;
  using base::NumFramesDecoded;
  using base::PossiblyResizeHash;
  using base::pr_time_;
  using base::ProcessNonemitting;
  using base::PruneActiveTokens;
  using base::queue_;
  using base::tmp_array_;
  using base::token_allocator_;
  using base::token_label_available_idx_;
  using base::token_label_final_cost_;
  using base::token_label_map_;
  using base::toks_;
  using base::warned_;

  // Instantiate this class once for each thing you have to decode.
  // This version of the constructor does not take ownership of
  // 'fst'.
  LatticeIncrementalFactDecoderTpl(const LatticeIncrementalDecoderConfig &config,
                                   const TransitionModel *trans_model = nullptr,
                                   const std::string fst_in_str = "");

  // This version of the constructor takes ownership of the fst, and will delete
  // it when this object is destroyed.
  LatticeIncrementalFactDecoderTpl(const LatticeIncrementalDecoderConfig &config,
                                   FST *fst, const TransitionModel *trans_model,
                                   const std::string fst_in_str = "");

  ~LatticeIncrementalFactDecoderTpl();

  void LoadHTransducers(std::string fst_in_str);
  /// InitDecoding initializes the decoding, and should only be used if you
  /// intend to call AdvanceDecoding().  If you call Decode(), you don't need to
  /// call this.  You can also call InitDecoding if you have already decoded an
  /// utterance and want to start with a new utterance.
  void InitDecoding(FST *fst = nullptr, bool keep_context = false);

  /// An example of how to do decoding together with incremental
  /// determinization. It decodes until there are no more frames left in the
  /// "decodable" object. Note, this may block waiting for input
  /// if the "decodable" object blocks.
  /// In this example, config_.determinize_delay, config_.determinize_chunk_size
  /// and config_.determinize_max_active are used to determine the time to
  /// call GetLattice().
  /// Users may do it in their own ways by calling
  /// AdvanceDecoding() and GetLattice(). So the logic for deciding
  /// when we get the lattice would be driven by the user.
  /// The function returns true if any kind
  /// of traceback is available (not necessarily from a final state).
  bool Decode(DecodableInterface *decodable);

  /// This will decode until there are no more frames ready in the decodable
  /// object.  You can keep calling it each time more frames become available.
  /// If max_num_frames is specified, it specifies the maximum number of frames
  /// the function will decode before returning.
  void AdvanceDecoding(DecodableInterface *decodable, int32 max_num_frames = -1);

 protected:
  struct ArcToken {
    Token *tokens[kStatePerPhone];
    // ilabel will be used in ProcessEmitting
    // olabel will be used to construct the last arc of this ArcToken to the nextstate
    Label ilabel, olabel;
    StateId nextstate;

    ArcToken(Token *first_tok, Label ilabel, Label olabel, StateId nextstate)
        : ilabel(ilabel), olabel(olabel), nextstate(nextstate) {
      memset(tokens, 0, sizeof(tokens));
      tokens[0] = first_tok;
    }
  };
  using ElemArc = typename HashList<PairId, ArcToken *>::Elem;
  // pair for arc_toks_ mapping
  inline PairId ConstructPair(StateId fst_state, size_t arc_pos) {
    return static_cast<PairId>(fst_state) + (static_cast<PairId>(arc_pos) << 32);
  }
  /// Gets the weight cutoff.  Also counts the active tokens.
  BaseFloat GetCutoff(ElemArc *list_head, size_t *tok_count, BaseFloat *adaptive_beam,
                      ElemArc **best_elem, BaseFloat &best_weight);

  /// Processes emitting arcs for one frame.  Propagates from prev_toks_ to
  /// cur_toks_.  Returns the cost cutoff for subsequent ProcessNonemitting() to
  /// use.
  BaseFloat ProcessEmitting(DecodableInterface *decodable);

  // similar to FindOrAddToken but for ArcToken
  inline ArcToken *FindOrAddArcToken(PairId state, const Arc &arc,
                                     int32 frame_plus_one, BaseFloat tot_cost,
                                     bool *changed);
  void SetEntryTokenForArcTokens(DecodableInterface *decodable, BaseFloat cur_cutoff,
                                 BaseFloat adaptive_beam);
  void ExpandArcTokensToNextState(BaseFloat next_cutoff);
  void DeleteElemArcs(ElemArc *list);

  fst::PoolAllocator<ArcToken> *arc_token_allocator_;
  HashList<PairId, ArcToken *> arc_toks_;
  // count how many states are introduced by ExpandArcTokensToNextState()
  int expand_num_;
  // keep the followings to help SetEntryTokenForArcTokens()
  BaseFloat last_cutoff_;
  ArcToken *last_best_arc_tok_;
  std::vector<fst::ConstFst<fst::StdArc>> h_transducers_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(LatticeIncrementalFactDecoderTpl);
};

typedef LatticeIncrementalFactDecoderTpl<
    fst::StdFst, 2, decoder::BackToken<typename fst::StdFst::Arc>>
    LatticeIncrementalFactDecoder;

} // end namespace kaldi.

#endif
