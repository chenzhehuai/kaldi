// decoder/simple-decoder.h

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

#ifndef KALDI_CUDA_LATTICE_FASTER_DECODER_H_
#define KALDI_CUDA_LATTICE_FASTER_DECODER_H_

#ifdef __CUDACC__
  #define HOST __host__
  #define DEVICE __device__

#else
  #define HOST
  #define DEVICE
#endif

#include "util/stl-utils.h"
#include "fst/fstlib.h"
#include "lat/kaldi-lattice.h"
#include "itf/decodable-itf.h"
#include "omp.h"

#define __DEBUG__
#ifdef __DEBUG__
#define VERBOSE 1
#define CUDA_PRINTF(VB, format,...) if (VERBOSE > VB) printf( format, ##__VA_ARGS__)
#else
#define VERBOSE 0
#define CUDA_PRINTF(VB, format,...)
#endif

#define USE_NVTX
#ifdef USE_NVTX
#include "nvToolsExt.h"
const uint32 colors[] = {0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff,
                           0x0000ffff, 0x00ff0000, 0x00ffffff};
const int32 num_colors = sizeof(colors) / sizeof(uint32);

#define PUSH_RANGE(name,cid) do { \
      int32 color_id = cid; \
      color_id = color_id%num_colors;\
      nvtxEventAttributes_t eventAttrib = {0}; \
      eventAttrib.version = NVTX_VERSION; \
      eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
      eventAttrib.colorType = NVTX_COLOR_ARGB; \
      eventAttrib.color = colors[color_id]; \
      eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
      eventAttrib.message.ascii = name; \
      nvtxRangePushEx(&eventAttrib); \
} while (0);
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif
namespace kaldi {

#define LAT_BUF_SIZE 2
#define ESTIMATED_PRUNE_RATIO 0.25
  
/** 
 * Simple Cuda Decoder
 */
class CudaLatticeFasterDecoder;

class CudaFst {
  public:
    typedef fst::StdArc StdArc;
    typedef StdArc::Weight StdWeight;
    typedef StdArc::Label Label;
    typedef StdArc::StateId StateId;
    
    CudaFst() {};
    void initialize(const fst::Fst<StdArc> &fst);
    void finalize();

    inline uint32_t NumStates() const {  return numStates; }
    inline StateId Start() const { return start; }    
    HOST DEVICE inline float Final(StateId state) const;
    size_t getCudaMallocBytes() const { return bytes_cudaMalloc; }
  private:
    friend class CudaLatticeFasterDecoder;
  
    unsigned int numStates;               //total number of states
    StateId  start;

    unsigned int max_ilabel;              //the largest ilabel
    unsigned int e_count, ne_count, arc_count;       //number of emitting and non-emitting states
  
    //This data structure is similar to a CSR matrix format 
    //where I have 2 matrices (one emitting one non-emitting).
 
    //Offset arrays are numStates+1 in size. 
    //Arc values for state i are stored in the range of [i,i+1)
    //size numStates+1
    unsigned int *e_offsets_h,*e_offsets_d;               //Emitting offset arrays 
    unsigned int *ne_offsets_h, *ne_offsets_d;            //Non-emitting offset arrays
 
    //These are the values for each arc. Arcs belonging to state i are found in the range of [offsets[i], offsets[i+1]) 
    //non-zeros (Size arc_count+1)
    BaseFloat *arc_weights_h, *arc_weights_d;
    StateId *arc_nextstates_h, *arc_nextstates_d;
    int32 *arc_ilabels_h, *arc_ilabels_d;
    int32 *arc_olabels_h;

    //final costs
    float *final_h, *final_d;
    //allocation size
    size_t bytes_cudaMalloc;
};

struct CudaLatticeFasterDecoderConfig {
  BaseFloat gpu_fraction;
  BaseFloat lat_fraction;
  uint32 max_tokens_per_frame;
  uint32 max_lat_arc_per_frame;
  uint32 max_tokens;
  uint32 max_arcs;
  BaseFloat lattice_beam;
  BaseFloat beam;
  uint32 prune_interval;
  int32 max_active;

  fst::DeterminizeLatticePhonePrunedOptions det_opts;
  bool determinize_lattice;
  int32 mem_print_freq;
  int32 verbose;
  
  CudaLatticeDecoderConfig(): 
                       gpu_fraction(1.0/8.0),
                       lat_fraction(1.0/2.0),
                       max_tokens_per_frame(800000),
                       max_lat_arc_per_frame(1000000),
                       max_tokens(15000000),
                       max_arcs(20000000), 
                       lattice_beam(10.0),
                       beam(16.0),
                       prune_interval(3000),
                       max_active(100000),
                       determinize_lattice(true),
                       mem_print_freq(10),
                       verbose(0) { }
 
  void Register(OptionsItf *opts) {
    det_opts.Register(opts);
    opts->Register("cuda-verbose", &verbose, "debug log verbose.");
    opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
    opts->Register("lat-fraction", &lat_fraction, 
      "Percent of GPU to use for lattice processing, i.e. gpu_fraction*lat_fraction");
    opts->Register("gpu-fraction", &gpu_fraction, 
      "Percent of GPU to use for this LatticeDecoder.  "
      "A single decoding cannot saturate the device.  "
      "Use multiple LatticeDecoders in parallel for the best performance.");
    opts->Register("max-tokens-per-frame", &max_tokens_per_frame, 
      "Maximum tokens used per frame.  If decoding exceeds this resutls are undefined.");
    opts->Register("max-arcs-per-frame", &max_lat_arc_per_frame, 
      "Maximum arcs used per frame.  If decoding exceeds this resutls are undefined.");
    opts->Register("max-tokens-allocated", &max_tokens, 
      "Total number of tokens allocated.  This controls how many tokens"
      " are allocated to the entire decoding process."
      "  If actual usaged exceeds this the results are undefined.");
    opts->Register("max-arcs-allocated", &max_arcs, 
      "Total number of arcs allocated.  This controls how many tokens " 
      " are allocated to the entire decoding process. "
      "  If actual usaged exceeds this the results are undefined.");
    opts->Register("lattice-beam", &lattice_beam, "Lattice generation beam.  Larger->slower, "
                   "and deeper lattices");
    opts->Register("prune-interval", &prune_interval, "Interval (in frames) at "
                   "which to prune tokens");
    opts->Register("max-active", &max_active, "Decoder max active states.  Larger->slower; "
                   "more accurate. It's a faster but approximate version for GPU.");    
    opts->Register("determinize-lattice", &determinize_lattice, "If true, "
                   "determinize the lattice (lattice-determinization, keeping only "
                   "best pdf-sequence for each word-sequence).");    
  }
  void Check() const {
    KALDI_ASSERT(beam > 0.0 && gpu_fraction>0 && gpu_fraction <= 1 &&
     lat_fraction>0 && lat_fraction <= 1 
      && max_tokens_per_frame > 0 && max_tokens>0 && lattice_beam > 0.0
                 && prune_interval > 0);
  }
};


#define Token InfoToken
// align to 16 bits so as to fast memcpy, see store16()
class __align__(16) InfoToken {
 public:
  BaseFloat cost_; // used in total_cost = acoustic_cost + arc_weight + old_tok_cost; in expand_arcs_kernel()
  BaseFloat extra_cost_acoustic_cost_; // acoustic_cost used in lat in compute_degrees_kernel(); extra_cost used in lattice pruning
  
  int prev_token_; // used in lat in compute_degrees_kernel()
  int arc_idx_; // used in lat in compute_degrees_kernel() to get graph info
  //StateId state_id; // used in params.d_lookup[arc_next_state]
  //state_id = fst_.arc_nextstates_d[arc_idx]

  HOST DEVICE Token(BaseFloat cost, BaseFloat acoustic_cost, int prev_token, int arc_idx) : 
                          cost_(cost), extra_cost_acoustic_cost_(acoustic_cost), 
                          prev_token_(prev_token), arc_idx_(arc_idx) {
    assert(sizeof(Token)==16); 
    if(prev) {
      cost_ += prev->cost_;
    }
  }
  HOST DEVICE Token() { } 

  HOST DEVICE bool operator < (const Token &other) {
    return cost_ > other.cost_;
  }
  HOST DEVICE bool operator < (const Token &other) volatile{
    return cost_ > other.cost_;
  }
  DEVICE Copy(const InfoToken &tok) {
    fast_store16(this, &tok);
  }
  DEVICE BaseFloat GetAcousticAndInitExtraCost() {
    BaseFloat acoustic_cost = extra_cost_acoustic_cost_;
    extra_cost_acoustic_cost_ = 0;
    return acoustic_cost;
  }
  HOST DEVICE int GetStateId(int *arc_nextstates) {
    return arc_nextstates[arc_idx_];
  }

};



// we save all info in this structure, so as to collect all info together
// in GPU memory and use memcpy to move to CPU memory
// this structure is only used before D2H memcpy, during decoding, 
// we use LatLinkCompact
class __align__(32) LatLink {  
 public:
   // below variables are with the same size as ForwardLink, so as to enable memcpy
  void *p1; // pack of (int32 next_tok_id, int32 next_tok_fr;)
  int32 ilabel; // ilabel on link.
  int32 olabel; // olabel on link.
  BaseFloat graph_cost; // graph cost of traversing link (contains LM, etc.)
  BaseFloat acoustic_cost; // acoustic cost (pre-scaled) of traversing link
  void *p2; // pack of (int32 prev_tok_id, int32 prev_tok_fr;)

  HOST DEVICE LatLink(int32 prev_tok_id, int32 prev_tok_fr,     
                             int32 next_tok_id, int32 next_tok_fr, 
      int32 ilabel, int32 olabel, BaseFloat graph_cost, BaseFloat acoustic_cost): 
      ilabel(ilabel), olabel(olabel), graph_cost(graph_cost), 
      acoustic_cost(acoustic_cost) {
    assert(sizeof(LatLink)==32); 
    p1=(void*)ENCODE_TOK_IDX_PAIR(next_tok_fr,next_tok_id);
    p2=(void*)ENCODE_TOK_IDX_PAIR(prev_tok_fr,prev_tok_id);
  }
};

// during decoding, as arcs are pre-allocated, we need a more compact 
// structure and it is aligned in 16 bits. 
// Notably, to work in 16 bits, we have to pack both id & whether it is 
// emit arc in is_emit_pack_prev_tok_id
class __align__(16) LatLinkCompact {  
 public:
  uint32 next_tok_id; // token index in the frame level token vector
  BaseFloat acoustic_cost; // acoustic cost (pre-scaled) of traversing link
  int32 arc_id;

  HOST DEVICE LatLinkCompact(uint32 prev_tok_id, int32 prev_tok_fr,     
                                    uint32 next_tok_id, int32 next_tok_fr, 
                                    BaseFloat acoustic_cost, int32 arc_id): 
      next_tok_id(next_tok_id), 
      acoustic_cost(acoustic_cost), arc_id(arc_id),
      is_emit_pack_prev_tok_id(prev_tok_id) {
    assert(sizeof(LatLinkCompact)==16); 
    // we can't cope with that large number
    assert(is_emit_pack_prev_tok_id < ((uint32)1<<31));  
    uint32 is_emit_arc = prev_tok_fr != next_tok_fr;
    // a hack to save is_emit_arc in is_emit_pack_prev_tok_id
    this->is_emit_pack_prev_tok_id |= (is_emit_arc<<31); 
  }
  HOST DEVICE bool IsEmitArc() {
    return is_emit_pack_prev_tok_id >= ((uint32)1<<31);
  }
  HOST DEVICE uint32 GetPrevTokId() {
    return is_emit_pack_prev_tok_id & (((uint32)1<<31) - 1);
  }
  DEVICE Copy(const LatLinkCompact &arc) {
    fast_store16(this, &arc);
  }

 private:
  // a hack to contain both id & whether it is emit arc
  uint32 is_emit_pack_prev_tok_id;  
};

 // for lattice processing
class LatticeProcessor {
 public:  
  void Initialize();
  int32 Allocate(int32 max_tokens_per_frame, int32 max_lat_arc_per_frame, 
    int32 prune_interval, int32 max_toks, int32 max_arcs, const CudaFst& fst);
  void Free();
  // The GPU memory of lattice arcs is shared with LatLinkVector
  LatLinkCompact* GetDeviceArcsBpr() { return arcs_bpr_d; } 

  DEVICE Token* GetTokenByExactIdx(uint32 offset);
  DEVICE int32 GetTokenIdxFromAddr(Token* tok);
  DEVICE int32 GetTokenAllocIdx(uint32 offset);

  // Entry of lattice pruning until this frame
  DEVICE void PruneActiveTokens(int32 frame, BaseFloat lattice_beam, int32 verbose);    
  // Collect after each token passing
  DEVICE void CollectToksPerFrame(int *cur_size, int32 frame);
  DEVICE void CollectArcsPerFrame(int *cur_size, int32 frame);

  // Data transfer from device to host
  void CopyArcsToHost(int32 frame, cudaStream_t st);
  void CopyToksToHost(int32 frame, cudaStream_t st);
  void GetHostData(Token** toks_buf, int** toks_fr_sidx, 
                            LatLink** arcs_buf, int** arcs_fr_size);

 private:    
  // #define ENCODE_TOK_IDX_PAIR(frame,idx) (((uint64)(frame)<<32)+(idx))
  DEVICE int32 AddArc(LatLink* arc);
  DEVICE int32 AddArc(LatLinkCompact* arc, int32 frame);
  // Set start index in the buffer of the next frame
  DEVICE void SetNextSidx(int* sidx_buf, int32 size, int32 frame);
  DEVICE Token* GetActiveToken(void* p, bool check=false, int32 frame=-1) const;
  DEVICE Token* GetActiveToken(int32 frame, int32 id, bool check=false) const;
  DEVICE Token* GetActiveTokenByExactId(int32 frame, 
int32 id_exact, bool check) const;

  DEVICE LatLinkCompact* GetActiveArc(int32 frame, int32 id) const;
  DEVICE int32 GetSize(int* acc_len, int32 frame) const;
  // used in PruneLatticeForFrame()
  DEVICE void UpdateModifiedFlags( 
                volatile int32 **modified0, volatile int32** modified1,
                volatile int32 **modified2, int cnt, int32* modified_d);
  
  // The parallel lattice pruning is based on the algorithm in
  // LatticeFasterDecoder::PruneActiveTokens 
  // with necessary modifications for GPU parallelization:
  // i) parallelize the iterative updating of nodes and arcs over GPU
  // threads; ii) use a global arc vector to replace the linked lists in
  // the old implementation, for its lack of random access features to
  // enable parallel access; iii) implement the extra cost updating as
  // an atomic operation to eliminate write conflicts among threads.
  // When a lattice arc is pruned, we do not physically remove
  // the arc, as memory allocation is expensive. Instead, we do a
  // final merging step to aggregate all remaining arcs using thread
  // parallelism 
  // We do not prune lattice nodes because: i) we need a static mapping
  // for each arc to trace the previous and the next nodes before
  // and after D2H memory copy. We use frame index t and vector
  // index i to trace a node, thus node positions in the vector cannot
  // be changed. ii) the lattice is constructed in CPU by iterating
  // remaining arcs, thus nodes are implicitly pruned. iii) node D2H copy is done
  // in each frame asynchronously, which does not introduce overheads.
  DEVICE void PruneLatticeForFrame(int32 frame, 
                bool merge, BaseFloat lattice_beam, int32 verbose);

 private:
  // before pruning (bpr)

  // Preallocates tokens, allows threads to concurrently
  // allocate/deallocate objects quickly in GPU
  Token* toks_bpr_d; 
  Token* toks_bpr_h;
  // we keep start idx per-frame to fast index a token by (frame, idx) pair
  // see GetActiveToken() & GetActiveArc()
  int* toks_bpr_fr_sidx_d; 
  int* toks_bpr_fr_sidx_h;
  int* toks_num_used;
  // the GPU memory of lattice arcs is shared with LatLinkVector
  // see CudaLatticeDecoder::CudaLatticeDecoder()
  LatLinkCompact* arcs_bpr_d;
  // we keep start idx per-frame to fast index a arc by (frame, idx) pair
  int* arcs_bpr_fr_sidx_d; 
  int* arcs_bpr_used_d; // used in CollectArcsPerFrame() to get the size per-frame

  // after pruning (apr)
  // save size but not start idx because: i) it's in reverse order; 
  // of [frame-2*prune_interval+1, frame-1*prune_interval]
  // ii) we organize arcs by frame in CPU, which needs arc size per frame
  int* arcs_apr_fr_size_d; 
  int* arcs_apr_fr_size_h; 
  LatLink* arcs_apr_d;
  LatLink* arcs_apr_h;
  int* arcs_apr_used_d; // for atomic operations in mergeArc
  int* arcs_apr_used_h; // for final copying arcs to CPU

  // GPU global memory temp variables
  int32 *barrier_;
  int* count_vec_acc_d;
  int* modified_d;

  // configurations
  int32 prune_interval;
  int32 toks_buf_before_pr_size;
  int32 arcs_buf_before_pr_size;

  // for AddArc() from LatLinkCompact to LatLink
  // we record all information in LatLink to eliminate CPU memory lookup
  const int32 *arc_ilabels;
  const int32 *arc_olabels; 
  const BaseFloat *arc_weights;
};

class CudaLatticeFasterDecoder {

 public:
  typedef fst::StdArc StdArc;
  typedef StdArc::Weight StdWeight;
  typedef StdArc::Label Label;
  typedef StdArc::StateId StateId;
  typedef float CostType;

  CudaLatticeFasterDecoder(const CudaFst &fst, const CudaLatticeFasterDecoderConfig &config);  
  ~CudaLatticeFasterDecoder();

  inline size_t getCudaMallocBytes() const { return bytes_cudaMalloc; } 
  inline size_t getCudaMallocManagedBytes() const { return bytes_cudaMallocManaged;  }

  /// Decode this utterance.
  /// Returns true if any tokens reached the end of the file (regardless of
  /// whether they are in a final state); query ReachedFinal() after Decode()
  /// to see whether we reached a final state.
  bool Decode(DecodableInterface *decodable);

  bool ReachedFinal() const;

  // GetBestPath gets the decoding traceback. If "use_final_probs" is true
  // AND we reached a final state, it limits itself to final states;
  // otherwise it gets the most likely token not taking into account final-probs.
  // fst_out will be empty (Start() == kNoStateId) if nothing was available due to
  // search error.
  // If Decode() returned true, it is safe to assume GetBestPath will return true.
  // It returns true if the output lattice was nonempty (i.e. had states in it);
  // using the return value is deprecated.
  bool GetBestPath(Lattice *fst_out, bool use_final_probs = true) const;
  
  /// *** The next functions are from the "new interface". ***
  
  /// FinalRelativeCost() serves the same function as ReachedFinal(), but gives
  /// more information.  It returns the difference between the best (final-cost plus
  /// cost) of any token on the final frame, and the best cost of any token
  /// on the final frame.  If it is infinity it means no final-states were present
  /// on the final frame.  It will usually be nonnegative.
  BaseFloat FinalRelativeCost() const;

  /// InitDecoding initializes the decoding, and should only be used if you
  /// intend to call AdvanceDecoding().  If you call Decode(), you don't need
  /// to call this.  You can call InitDecoding if you have already decoded an
  /// utterance and want to start with a new utterance. 
  void InitDecoding();  


  struct ExpandArcParams {
      StateId *d_q; 
      InfoToken *d_q_info; 

      int *d_q_token_from; 
      int *d_q_token_to;
      int *d_q_token_end;

      int *d_q_token_from_narcs; 
      int *h_q_token_from_narcs;

      int *d_degrees_scan; 

      int *d_q_arc_offset; 
      int *arc_ilabels; 

      BaseFloat *arc_weights; 
      StateId *arc_nextstates; 
      BaseFloat *d_cutoff;
      BaseFloat *d_loglikelihoods;
      BaseFloat beam; 

      uint64 *d_lookup;
        
      bool is_emitting;

      int *d_n_CTA_done;

      int *h_q_token_from_size; // to be set at the end

      int *d_curr_token;
      int *d_dbg_tok_num;
      int *barrier;
      int frame;
      uint *d_arc_offsets;
      int* d_block_sums_scan;

      // lattice
      int *d_q_lat_end;
      LatticeProcessor lattice_processor;
};

    int debug_max_narcs;

  void ExpandArcs(int nthreads, const ExpandArcParams &params);

  void DeviceScan(int *d_degrees, int h_prevTok_size, int *d_degrees_scan);

  void ComputeDegrees(const ExpandArcParams &params);
  void FinalizeDegreesScan();

  /// This will decode until there are no more frames ready in the decodable
  /// object, but if max_num_frames is >= 0 it will decode no more than
  /// that many frames.  If it returns false, then no tokens are alive,
  /// which is a kind of error state.
  void AdvanceDecoding(DecodableInterface *decodable,
                         int32 max_num_frames = -1);
  
  /// Returns the number of frames already decoded.  
  int32 NumFramesDecoded() const { return num_frames_decoded_; }


  StateId *d_allToken; 
  InfoToken *d_allTokenInfo;

  // Used to detect last CTA alive in some kernels
  int *d_n_CTA_done;

  // At each ProcessToken, we will propagate the queue [from, to[ to [to, end[
  int *d_q_token_from;
  int *d_q_token_to;
  int *d_q_token_end; 

  // Save the offset of currToken of the current frame
  // Used for ProcessEmitting of following frame
  int *d_curr_token;

  // Total number of arcs contained in the [from,to[ queue
  // ie total # of arcs from tok.next_state, where tok is in [from,to[
  // (actually one "valid arcs" are counted, cf ComputeDegrees)
  int *d_q_token_from_narcs, *h_q_token_from_narcs; // TODO
 
  // Host Pinned memory
  // size = to - from, total # of tokens in [from,to[
  int *h_q_token_from_size;

  // Host Pinned memory
  // Total number of arcs contained in the [from,to[ queue
  // ie total # of arcs from tok.next_state, where tok is in [from,to[
  // (actually one "valid arcs" are counted, cf ComputeDegrees)

 
  // Scan of the outgoing arc degrees of tokens in [from,to[
  int *d_degrees_scan;

  // # arcs in the corresponding CTA block
  // Cf Compute degrees
  int *d_block_sums_scan;

  // Cf Compute degrees
  int *d_q_arc_offset;

  int *h_reached_final;

  // TODO remove the d_reversed_path, use only host
  StateId *d_reversed_path, *h_reversed_path;

  int *d_path_size;

  // Lookup table of all the costs
  // d_state_cost[state] -> best cost for that state
  // Resetted between frames
  // Costs is stored as an ordered int representing a float
  uint64 *d_state_cost;

  // Current cutoff for current frame
  BaseFloat *d_cutoff;

  int max_tokens;

  BaseFloat *loglikelihoods_h, *loglikelihoods_d, *next_loglikelihoods_d;  

  // Streams, overlap likelihoods copies with compute
  cudaStream_t compute_st, copy_st;
  cudaEvent_t loglikelihood_evt, loglikelihood_processed_evt;

  //pre-computes log likelihoods for the current frame
  void ComputeLogLikelihoods(DecodableInterface *decodable);
 
  // ProcessEmitting decodes the frame num_frames_decoded_ of the
  // decodable object, then increments num_frames_decoded_.
  //void ProcessEmitting(DecodableInterface *decodable);

  // Descriptions in .cu file

  void InitLookup();
  void ResetLookup(bool reset = true);
  void NonEmittingLongTail(unsigned int *d_arc_offsets, const ExpandArcParams &params);

  void GetBestCost(BaseFloat *min, int *arg, bool isfinal) const;
  void ProcessEmitting();
  void ProcessNonemitting();

 
  bool ProcessToken(unsigned int *d_arc_offsets, bool is_emitting);

  
  const CudaFst fst_;

  BaseFloat beam_;

  // Keep track of the number of frames decoded in the current file.
  int32 num_frames_decoded_;

  BaseFloat *cutoff;

  cudaEvent_t event;
  cudaStream_t st1;
  int *d_dbg_tok_num;
  int *d_barrier;

  size_t bytes_cudaMalloc, bytes_cudaMallocManaged;

  // for lattice
  
 public:
  void InitParams(ExpandArcParams &params);
  void FinalProcessLattice(Token** toks_buf, int** toks_fr_sidx,
    LatLink** arcs_buf, int** arcs_fr_size);

 private:
  void PruneActiveTokens(cudaStream_t wait_st,
    cudaStream_t run_st, BaseFloat gpu_ratio);

  LatLinkCompact* lat_arcs_buf_; // obtain from lattice_processor_ as 1343
  int *d_q_lat_end; 
  LatticeProcessor lattice_processor_; 
  cudaStream_t stream_lat[LAT_BUF_SIZE]; // lattice processing and copying 

  KALDI_DISALLOW_COPY_AND_ASSIGN(CudaLatticeFasterDecoder);
};


} // end namespace kaldi.


#endif
