// decoder/cuda-lattice-decoder.h

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

#ifndef KALDI_CUDA_LATTICE_DECODER_H_
#define KALDI_CUDA_LATTICE_DECODER_H_

#ifdef __CUDACC__
  #define HOST __host__
  #define DEVICE __device__

#else
  #define HOST
  #define DEVICE
#endif

//#define __DEBUG__
#ifdef __DEBUG__
#define VERBOSE 5
#define DEBUG(format,...) printf(format, ##__VA_ARGS__)
#else
#define VERBOSE 0
#define DEBUG(format,...)
#endif

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "util/stl-utils.h"
#include "fst/fstlib.h"
#include "lat/determinize-lattice-pruned.h"
#include "lat/kaldi-lattice.h"
#include "itf/decodable-itf.h"


namespace kaldi {

// TODO:
#define LAT_BUF_SIZE 2

class CudaLatticeDecoder;

class CudaFst {
  public:
    typedef fst::StdArc StdArc;
    typedef StdArc::StateId StateId;
    typedef float CostType;
    typedef StdArc::Weight StdWeight;
    typedef StdArc::Label Label;
    
    CudaFst() {};
    void initialize(const fst::Fst<StdArc> &fst);
    void finalize();

    inline uint32_t NumStates() const {  return numStates; }
    inline StateId Start() const { return start; }    
    HOST DEVICE float Final(StateId state) const;
    size_t getCudaMallocBytes() const { return bytes_cudaMalloc; }
  private:
    friend class CudaLatticeDecoder;
    friend class LatticeFasterDecoderCuda;
  
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
    int32 *arc_ilabels_h, *arc_ilabels_d, *arc_olabels_d;
    int32 *arc_olabels_h;

    //final costs
    float *final_h, *final_d;
    //allocation size
    size_t bytes_cudaMalloc;
};

template<typename T>
class CudaVector {
    public:
      HOST DEVICE T& operator[](uint32_t idx); 
      HOST DEVICE const T& operator[](uint32_t idx) const; 
      inline void allocate(uint32_t max_size);
      inline void allocate(uint32_t max_size, 
        uint32_t* icount_h=NULL, uint32_t* icount_d=NULL, T* mem_d=NULL, T* mem_h=NULL) ;
      inline void free(bool create_outside=false);
      HOST DEVICE uint32_t size() const; 
      HOST DEVICE inline uint32_t push_back(const T &val); 
      HOST DEVICE inline void clear(cudaStream_t stream=0); 
      HOST DEVICE inline int get_idx_from_addr(T* addr); 
      inline bool empty() const;
      inline void swap(CudaVector<T> &v); 
      inline void copy_all_to_host(cudaStream_t stream=0);
      inline void copy_all_to_device(cudaStream_t stream=0);
      inline void copy_size_to_host(cudaStream_t stream=0);
      inline void copy_size_to_device(cudaStream_t stream=0);
      inline void copy_data_to_host(cudaStream_t stream=0, T* to_buf=NULL, bool copy_size=true);
      inline void copy_data_to_device(cudaStream_t stream=0);
      inline void copy_data_to_device(int size, T* mem_in_d, cudaStream_t stream=0);

      inline size_t getCudaMallocBytes(); 
      
    public:
      uint32_t *count_d, *count_h;
      uint32_t max_size;
      T* mem_d, *mem_h;
      int alloc_size;
};

struct CudaLatticeDecoderConfig {
  BaseFloat beam;
  double gpu_fraction;
  uint32_t max_tokens_per_frame;
  uint32_t max_lat_tok_per_frame;
  uint32_t max_lat_arc_per_frame;
  uint32_t max_tokens;
  uint32_t max_arcs;
  int32_t prune_interval;
  BaseFloat lattice_beam;
  bool determinize_lattice;
  BaseFloat prune_scale;
  fst::DeterminizeLatticePhonePrunedOptions det_opts;

  int verbose;
  
  CudaLatticeDecoderConfig(): beam(16.0),
                       gpu_fraction(1.0/8.0),
                       max_tokens_per_frame(200000),
                       max_lat_tok_per_frame(200000),
                       max_lat_arc_per_frame(600000),
                       max_tokens(6000000),
                       max_arcs(9000000),
                       prune_interval(2000),
                       lattice_beam(10.0),
                       determinize_lattice(true),
                       prune_scale(0.1), 
                       verbose(0) { }
 
  void Register(OptionsItf *opts) {
    det_opts.Register(opts);
    opts->Register("cuda-verbose", &verbose, "debug log verbose.");
    opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
    opts->Register("gpu-fraction", &gpu_fraction, "Percent of GPU to use for this LatticeDecoder.  "
                                                  "A single decoding cannot saturate the device.  "
                                                  "Use multiple LatticeDecoders in parallel for the best performance.");
    opts->Register("max-tokens-per-frame", &max_tokens_per_frame, "Maximum tokens used per frame.  If decoding exceeds this resutls are undefined.");
    opts->Register("max-arcs-per-frame", &max_lat_arc_per_frame, "Maximum arcs used per frame.  If decoding exceeds this resutls are undefined.");
    opts->Register("max-tokens-allocated", &max_tokens, "Total number of tokens allocated.  This controls how many tokens are allocated to the entire decoding process."
                                                        "  If actual usaged exceeds this the results are undefined.");
    opts->Register("max-arcs-allocated", &max_arcs, "Total number of arcs allocated.  This controls how many tokens are allocated to the entire decoding process."
                                                        "  If actual usaged exceeds this the results are undefined.");

    opts->Register("lattice-beam", &lattice_beam, "Lattice generation beam.  Larger->slower, "
                   "and deeper lattices");
    opts->Register("prune-interval", &prune_interval, "Interval (in frames) at "
                   "which to prune tokens");
    opts->Register("determinize-lattice", &determinize_lattice, "If true, "
                   "determinize the lattice (lattice-determinization, keeping only "
                   "best pdf-sequence for each word-sequence).");    
  }
  void Check() const {
    KALDI_ASSERT(beam > 0.0 && gpu_fraction>0 && gpu_fraction <= 1 
      && max_tokens_per_frame > 0 && max_tokens>0 && lattice_beam > 0.0
                 && prune_interval > 0);
  }
};



class CudaLatticeDecoder {

 public:
  typedef fst::StdArc StdArc;
  typedef StdArc::Weight StdWeight;
  typedef StdArc::Label Label;
  typedef StdArc::StateId StateId;
  typedef float CostType;

class __align__(16) Token {
 public:
  //Token *prev_; in lattice decoder, it's of no use
  CostType cost_; // accumulated total cost up to this point.
  int32_t frame; //used in generation
  float extra_cost;//used in pruning
  StateId state_id;
  //BaseFloat acoustic_cost;   //currently not recording acoustic_cost.  It is trivial to add back in but didn't seem necessary for this use case

  HOST DEVICE inline Token(BaseFloat cost, int frame, Token* prev, StateId state_id) : cost_(cost), frame(frame), extra_cost(0), state_id(state_id) {
    assert(sizeof(Token)==16); 
    if(prev) {
      cost_ += prev->cost_;
    }
  }
  HOST DEVICE inline Token() { } 

  HOST DEVICE inline bool operator < (const Token &other) {
    return cost_ > other.cost_;
  }
  HOST DEVICE inline bool operator < (const Token &other) volatile{
    return cost_ > other.cost_;
  }
};

  //struct to hold pre-allocated tokens (one per state)
  struct __align__(16) TokenLookupElem{
    Token *token;     //pointer for that token
    uint32_t active;  //tells if token has activiated or not
    volatile int32_t tokenstate_idx;     //aligning to 16 bytes
  };


struct TokenState {
  public:
  
  Token* token; //arc and labels
  StateId state;  //to state
  CostType cost_; //for CPU to copy lattice without prefetch allocator
  //int32_t lat_tok_idx;   //-1: havent init 
  HOST DEVICE inline TokenState (Token *token, StateId state, CostType cost)
    : token(token), state(state), cost_(cost) { }
};

typedef CudaVector<TokenState> TokenVector;

#define ENCODE_TOK_ADDR(frame,idx) (((uint64)(frame)<<32)+(idx))
#define DECODE_TOK_ADDR(frame,idx,v) { \
    frame=(((uint64)v)>>32); \
    idx=(((uint64)v)&(((uint64)1<<32)-1)); \
}
  class __align__(32) LatLink {  //300000*50*24=240MB __align__(16)
   public:
     //below value are totally the same to ForwardLink, which enable memcpy
    //*_fr can be combined into 1 single para
    void *p1; //    int next_tok_id;  int next_tok_fr; 
    int32 ilabel;
    int32 olabel;
    float graph_cost;
    float acoustic_cost; // acoustic cost (pre-scaled) of traversing link
    void *p2; //    int prev_tok_id;    int prev_tok_fr;

     HOST DEVICE inline LatLink(int iprev_tok_id, int iprev_tok_fr,     
      int inext_tok_id, int inext_tok_fr, 
      int32 iilabel, int32 iolabel, float igraph_cost, 
      float iacoustic_cost): ilabel(iilabel), olabel(iolabel),
      graph_cost(igraph_cost), acoustic_cost(iacoustic_cost) {
        p1=(void*)ENCODE_TOK_ADDR(inext_tok_fr,inext_tok_id);
        p2=(void*)ENCODE_TOK_ADDR(iprev_tok_fr,iprev_tok_id);
      }

  };

  typedef CudaVector<LatLink> LatLinkVector;
  typedef CudaMergeVector<LatLink> LatLinkVectorMerge;

  union __align__(16) TokenOrSize {
    Token tok;
    int size[4];
  };

  //Preallocates tokens and allocates them in a circular buffer.
  //This allows threads to concurrently allocate/deallocate objects quickly in CUDA
  class TokenAllocator {
    public:
      void initialize(uint32_t size);
      void finalize();

      inline void prefetch_next_to_device(cudaStream_t stream, int count);
      inline void prefetch_next_to_device(cudaStream_t stream);
      inline void prefetch_allocated_to_host(cudaStream_t stream);
      inline void prefetch_allocated_to_host_force(cudaStream_t stream);
      inline void prefetch_allocated_to_host_since_last(cudaStream_t stream);

      inline size_t getCudaMallocManagedBytes();

      //circular buffer,  need to ensure front never gets close to back....  If this happens there can be race conditions 

      DEVICE inline Token* getToken(uint32_t index);   //gets a free token offset by index
      DEVICE inline void advanceFront(uint32_t num);         //advances the allocated token list by num

      void reset();   //returns all memory to the allocator (essentially a garbage collection of oustanding memory.  
    private:

      uint32_t size;
      int32_t device;
      uint32_t *front_d, *front_h, *last_front_h, *last2_front_h;    //next free token index

      Token *tokens_allocation;  //TODO we could have a list of these and dynamically add more.  Just going static for now.
      size_t bytes_cudaMallocManaged;
      uint32_t prefetch_size;         //amount of elements to prefetch beyond front
  };

  //for pruning
  class LatticePruner {
  public:  
    #define DECODE_TOK_ADDR(frame,idx,v) { \
    frame=(((uint64)v)>>32); \
    idx=(((uint64)v)&(((uint64)1<<32)-1)); \
    }
    inline DEVICE void init_buf_before_cp();
    inline DEVICE int merge_arc(LatLink* arc);
    int allocate(int max_tokens_per_frame, int max_lat_arc_per_frame, 
      int prune_interval, int max_toks, int max_arcs);
    void free();
    inline DEVICE void set_next_sidx(int* sidx_buf, int size, int frame);
    inline DEVICE void collect_tok_per_frame(TokenState* cur_toks, int size, int frame);
    inline DEVICE void collect_arc_per_frame(LatLinkVector* cur_arc_array, 
      uint* count_vec_d, int frame);
    template <int verbose>
    inline DEVICE void PruneActiveTokens(int frame, float lattice_beam);
    inline DEVICE Token* ActiveToksMap(void* p, bool check=false, int frame=-1) const;
    inline DEVICE Token* ActiveToksMap(int frame, int id, bool check=false) const;
    inline DEVICE LatLink* ActiveArcsMap(int frame, int id) const;
    inline DEVICE int GetSize(int* acc_len, int frame) const;
    template <int verbose>
    inline DEVICE void PruneForwardLinks_PruneTokensForFrame(int frame, 
                  bool merge, float lattice_beam);
    LatLink* GetArcBpr() { return arcs_bpr_d; }
        
    void init();
    void copy_arcs_to_host(int frame, cudaStream_t st);
    void copy_toks_to_host(int frame, cudaStream_t st);
    void get_data_copied_to_host(Token** toks_buf, int** toks_sidx, LatLink** arcs_buf, int** arcs_size);

  private:
    //after pruning
    //Arc
    //it's in reverse seq, it includes a header to contain sizeof arc&tok per frame, of [frame-2*prune_interval, frame-1*prune_interval-1]
    int* arcs_apr_size_d; //single buffer to contain all size, so that we can copy in a command
    //int* arcs_buf_after_pr_size_arr_used_d;
    int* arcs_apr_size_h; //only used to save size, currently
    //it's in reverse seq; of [frame-2*prune_interval+1, frame-1*prune_interval]
    LatLink* arcs_apr_d;
    LatLink* arcs_apr_h;
    //for mergeArc
    int* arcs_apr_used_d;
    int* arcs_apr_used_h;
    //TokenState
    //TokenState* ts_exact_after_pr_h;
    //int ts_exact_after_pr_h_used;
    //int* ts_exact_after_pr_size_arr_h;

    //before
    Token* toks_bpr_d;  
    Token* toks_bpr_h;
    int* toks_bpr_sidx_d;
    int* toks_bpr_sidx_h;
    LatLink* arcs_bpr_d;
    int* arcs_bpr_sidx_d;
    int* arcs_bpr_used_d;

    int *barrier_;
    int* count_vec_acc_d;
    int prune_interval;
    int toks_buf_before_pr_size;
    int arcs_buf_before_pr_size;
    int* modified_d;
    int* merge_d;
  };
 
 
  struct processTokens_params {
    TokenVector prev_toks;
    TokenVector cur_toks;
    TokenAllocator allocator;
    CostType *cutoff;

    //never change
    const __restrict__ uint32_t *e_offsets;
    const __restrict__ uint32_t *ne_offsets;
    const __restrict__ int32 *arc_ilabels;
    const __restrict__ int32 *arc_olabels; 
    const __restrict__ BaseFloat *arc_weights;
    const __restrict__ StateId *arc_nextstates;
    const __restrict__ BaseFloat *loglikelihoods;
    TokenLookupElem *current_tokens_lookup;
    volatile int *token_locks;
    BaseFloat beam;
    volatile int *modified;
    int *pe_idx;
    int *ne_idx;
    int *fb_idx;
    int *barrier;

    //debug
    int verbose;

    uint32_t frame;
    int prune_interval;
    LatLinkVector lat_arcs_sub_vec;

    LatticePruner lattice_pruner;
    float lattice_beam;
  };


  CudaLatticeDecoder(const CudaFst &fst, const CudaLatticeDecoderConfig &config);  
  ~CudaLatticeDecoder();

  inline size_t getCudaMallocBytes() const { return bytes_cudaMalloc; } 
  inline size_t getCudaMallocManagedBytes() const { return bytes_cudaMallocManaged;  }

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
  void ClearArcVector(LatLinkVector* lat_arcs_sub_vec_);
  void initParams(processTokens_params& params);
  void PreFinalizeDecoding(
TokenVector**last_tokv,  Token** toks_buf, int** toks_sidx, LatLink** arcs_buf, int** arcs_size);
  void CallLaunchPruneActiveTokens(cudaStream_t wait_st, 
    cudaStream_t st, float ratio);

  #if 0
  void PostProcessLattices(bool islast, uint dec_frame);
  void PreProcessLattices(TokenVector** pprev_toks, 
    void** buf_arcs, int *num_arcs, bool islast, int* lat_frame, uint dec_frame);
  #endif
  void PreProcessTokens();

  /// Returns the number of frames already decoded.  
  int32 NumFramesDecoded() const { return num_frames_decoded_; }


 
  TokenAllocator allocator;

  //pre-computes log likelihoods for the current frame
  void ComputeLogLikelihoods(DecodableInterface *decodable);
 
  // ProcessEmitting decodes the frame num_frames_decoded_ of the
  // decodable object, then increments num_frames_decoded_.
  //void ProcessEmitting(DecodableInterface *decodable);

  void ProcessNonemitting();
  void ProcessTokens();
  void PostProcessTokens(); 
  void SetTokArcPointerByFrame(uint frame);

 
  //token lookup table.  Provides constant time lookup for active tokens.
  //One entry per state.  If entry is NULL token is not active.
  TokenLookupElem *current_tokens_lookup_d;

  //Lists of active tokens to be iterated through
  TokenVector* cur_toks_;
  TokenVector* prev_toks_;  

  const CudaFst fst_;

  BaseFloat beam_;
  // Keep track of the number of frames decoded in the current file.
  int32 num_frames_decoded_;

  //data store for log likelihoods needed in the current frame.  Double buffering to avoid synchronization.
  BaseFloat *loglikelihoods_h, *loglikelihoods_old_h, *loglikelihoods_d, *loglikelihoods_old_d;  

  CostType *cutoff_d;
  int *modified_d;

  //TODO
  volatile int *token_locks_d;
  void ClearToks(TokenVector &toks);

  cudaEvent_t event_pt, event_pt_old, event_ll;
  cudaStream_t stream_comp, stream_copy[LAT_BUF_SIZE+1], stream_ll;

  uint32_t total_threads;
  size_t bytes_cudaMalloc, bytes_cudaMallocManaged;

  //warp assignment indexes
  int *pe_idx_d, *ne_idx_d, *fb_idx_d;
  int *barrier_d;  //barrier to allow grid syncs
  
  int verbose;
  int prune_interval_;
  int max_arcs_;
  float lattice_beam_;

  //for recording lattice
  LatLinkVector* lat_arcs_sub_vec_;
  LatLinkVector lat_arcs_sub_vec_buf_;
  
  TokenVector toks_buf_[LAT_BUF_SIZE];
  /*
  LatLinkVectorMerge* arc_copy_buf_;  //used to cur_vec.load() data from sub_vecs
  LatLink* arcs_buf_; //as GPU is so fast, we have to need this; assume cpuLatLink has same size as LatLink
  LatLink* arcs_ready2cpu_[LAT_BUF_SIZE]; //from arcs_buf_
  int arcs_buf_used_;
  */
 
  LatticePruner lattice_pruner_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(CudaLatticeDecoder);
};

} // end namespace kaldi.


#endif
