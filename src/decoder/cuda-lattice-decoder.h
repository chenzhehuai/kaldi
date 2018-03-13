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

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "util/stl-utils.h"
#include "fst/fstlib.h"
#include "lat/determinize-lattice-pruned.h"
#include "lat/kaldi-lattice.h"
#include "itf/decodable-itf.h"


namespace kaldi {

/** 
 * Simple Cuda Decoder
 */

#define LAT_BUF_SIZE 3

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
        uint32_t* icount_h, uint32_t* icount_d) ;
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
      inline void copy_data_to_host(cudaStream_t stream=0, void* to_buf=NULL, bool copy_size=true);
      inline void copy_data_to_device(cudaStream_t stream=0);
      inline void copy_data_to_device(int size, T* mem_in_d, cudaStream_t stream=0);

      inline size_t getCudaMallocBytes(); 
      
    public:
      uint32_t *count_d, *count_h;
      uint32_t max_size;
      T* mem_d, *mem_h;
};

template<typename T>
class CudaMergeVector : public CudaVector<T> {
  #define MAX_SUB_VEC_SIZE (2048+1)
public:
  //HOST DEVICE T& operator[](uint32_t idx); 
  //using CudaVector<T>::operator[];
  //HOST DEVICE uint32_t size() const; 
  //using CudaVector<T>::size;
  using CudaVector<T>::count_d;
  using CudaVector<T>::mem_d;
  using CudaVector<T>::max_size;
  
  inline void load(CudaVector<T>*in, int sub_vec_num, cudaStream_t st, int total_threads, uint32_t* count_vec_d=NULL);
  inline void reg(CudaVector<T>*in, int sub_vec_num, cudaStream_t st);
  inline void allocate(uint32_t max_size);
  inline void free();

  //for arr merge to single; assume create using cudaMallocManaged
  T** arr_; //add cache here
  int *vec_len_acc_;
  int* barrier_;
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
  int32_t sub_vec_num;
  
  CudaLatticeDecoderConfig(): beam(16.0),
                       gpu_fraction(1.0/8.0),
                       max_tokens_per_frame(200000),
                       max_lat_tok_per_frame(200000),
                       max_lat_arc_per_frame(600000),
                       max_tokens(60000000),
                       max_arcs(180000000),
                       prune_interval(25),
                       lattice_beam(10.0),
                       determinize_lattice(true),
                       prune_scale(0.1), 
                       verbose(0),
                       sub_vec_num(32) { }
  
  void Register(OptionsItf *opts) {
    det_opts.Register(opts);
    opts->Register("cuda-verbose", &verbose, "debug log verbose.");
    opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
    opts->Register("sub-vec-num", &sub_vec_num, "sub_vecs.");
    opts->Register("gpu-fraction", &gpu_fraction, "Percent of GPU to use for this LatticeDecoder.  "
                                                  "A single decoding cannot saturate the device.  "
                                                  "Use multiple LatticeDecoders in parallel for the best performance.");
    opts->Register("max-tokens-per-frame", &max_tokens_per_frame, "Maximum tokens used per frame.  If decoding exceeds this resutls are undefined.");
    opts->Register("max-tokens-allocated", &max_tokens, "Total number of tokens allocated.  This controls how many tokens are allocated to the entire decoding process."
                                                        "  If actual usaged exceeds this the results are undefined.");
    opts->Register("max-tokens-allocated", &max_arcs, "Total number of arcs allocated.  This controls how many tokens are allocated to the entire decoding process."
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
  //BaseFloat acoustic_cost;   //currently not recording acoustic_cost.  It is trivial to add back in but didn't seem necessary for this use case

  HOST DEVICE inline Token(BaseFloat cost, int frame) : extra_cost(0), cost_(cost), frame(frame) {
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

  #define GET_ARCIDX(rawid, thd) ((rawid<<5)+thd)   //assume 32 threads
  #define GET_RAWARCIDX(id)  (id>>5) //assume 32 threads
  #define GET_THDIDX(id) (id&((1<<5)-1)) //assume 32 threads

  typedef CudaVector<LatLink> LatLinkVector;
  typedef CudaMergeVector<LatLink> LatLinkVectorMerge;

  union __align__(16) TokenOrSize {
    Token tok;
    int size[4];
  };

  LatLink* 

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
    LatLinkVector* lat_arcs_vec;
    LatLinkVector* lat_arcs_sub_vec;
    int sub_vec_num;
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
  void PreFinalizeDecoding();
  void PostProcessLattices(bool islast, uint dec_frame);
  void PreProcessLattices(TokenVector** pprev_toks, 
    void** buf_arcs, int *num_arcs, bool islast, int* lat_frame, uint dec_frame);
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

  volatile int *token_locks_d;
  void ClearToks(TokenVector &toks);

  cudaEvent_t event_pt, event_pt_old, event_ll;
  cudaStream_t stream_comp, stream_copy[LAT_BUF_SIZE], stream_ll;

  uint32_t total_threads;
  size_t bytes_cudaMalloc, bytes_cudaMallocManaged;

  //warp assignment indexes
  int *pe_idx_d, *ne_idx_d, *fb_idx_d;
  int *barrier_d;  //barrier to allow grid syncs
  
  int verbose;
  int prune_interval_;
  int sub_vec_num_;
  int max_arcs_;

  //for recording lattice
  LatLinkVector* lat_arcs_sub_vec_;
  LatLinkVector* lat_arcs_sub_vec_prev_;
  LatLinkVector* lat_arcs_sub_vec_buf_[LAT_BUF_SIZE];
  uint32_t* lat_arcs_sub_vec_buf_count_[LAT_BUF_SIZE][2]; //0 for CPU 1 for GPU
  
  /*
  TokenVector toks_buf_[LAT_BUF_SIZE];
  LatLinkVectorMerge* arc_copy_buf_;  //used to cur_vec.load() data from sub_vecs
  LatLink* arcs_buf_; //as GPU is so fast, we have to need this; assume cpuLatLink has same size as LatLink
  LatLink* arcs_ready2cpu_[LAT_BUF_SIZE]; //from arcs_buf_
  int arcs_buf_used_;
  */

  //for pruning
  class LatticePruner {
  public:  
    #define GET_IDX_BY_FRAME1(frame) ((frame)%(prune_interval))
    #define GET_IDX_BY_FRAME2(frame) ((frame)%(prune_interval*2))
    #define GET_IDX_BY_FRAME3(frame) ((frame)%(prune_interval*2+1))

    inline DEVICE void init_buf_before_cp() {
      if (threadIdx.x!=0||blockIdx.x!=0) return;
      *toks_buf_after_pr_used_d=prune_interval_/sizeof(int)*2;
      *arcs_buf_after_pr_used_d=0;
    }
    inline DEVICE int* get_tok_size_of_frame(int f) {
      return (toks_buf_after_pr_d->size+(f));
    }
    inline DEVICE int* get_arc_size_of_frame(int f) {
      return toks_buf_after_pr_d->size+(f)+prune_interval;
    }
    inline DEVICE int merge_arc(LatLink* arc) {
      int i=atomicAdd(arcs_buf_after_pr_used_d, 1);
      store32(arcs_buf_after_pr_d+i, arc)
    }
    inline DEVICE int merge_tok(Token* tok) {
      int i=atomicAdd(toks_buf_after_pr_used_d, 1);
      store16(toks_buf_after_pr_d+i, tok)
    }
    int allocate(int max_tokens_per_frame, int max_lat_arc_per_frame, 
      int prune_interval) {
      int sz;
      int bytes_cudaMalloc=0;
      float prune_ratio_assume=0.5;
      //after
      sz=prune_ratio_assume*sizeof(TokenOrSize)*max_tokens_per_frame*prune_interval*(1/sizeof(int)*2+1);
      cudaMalloc((void**)&toks_buf_after_pr_d,sz); bytes_cudaMalloc+=sz;
      sz=prune_ratio_assume*sizeof(LatLink)*max_lat_arc_per_frame*prune_interval*(1);
      cudaMalloc((void**)&arcs_buf_after_pr_d,sz); bytes_cudaMalloc+=sz;
      sz=sizeof(int);
      cudaMalloc((void**)&toks_buf_after_pr_used_d,sz); bytes_cudaMalloc+=sz;
      sz=sizeof(int);
      cudaMalloc((void**)&arcs_buf_after_pr_used_d,sz); bytes_cudaMalloc+=sz;
      //before
      sz=sizeof(Token)*max_tokens_per_frame*prune_interval*(2);
      cudaMalloc((void**)&toks_buf_before_pr_d,sz); bytes_cudaMalloc+=sz;
      toks_buf_before_pr_size=sz/sizeof(Token);
      sz=sizeof(LatLink)*max_lat_arc_per_frame*prune_interval*(2);
      cudaMalloc((void**)&arcs_buf_before_pr_d,sz); bytes_cudaMalloc+=sz;
      arcs_buf_before_pr_size=sz/sizeof(LatLink);
      sz=sizeof(int)*(prune_interval*2+1);
      cudaMalloc((void**)&per_fr_sidx_toks_buf,sz); bytes_cudaMalloc+=sz;
      sz=sizeof(int)*(prune_interval*2+1);
      cudaMalloc((void**)&per_fr_sidx_arcs_buf,sz); bytes_cudaMalloc+=sz;
      sz=sizeof(int);
      cudaMalloc((void**)&barrier_,sz); bytes_cudaMalloc+=sz;
      this->prune_interval=prune_interval;
      return bytes_cudaMalloc;
    }
    void free() {
      cudaFree(toks_buf_after_pr_d);
      cudaFree(toks_buf_after_pr_used_d);
      cudaFree(arcs_buf_after_pr_d);
      cudaFree(arcs_buf_after_pr_used_d);
      cudaFree(toks_buf_before_pr_d);
      cudaFree(arcs_buf_before_pr_d);
      cudaFree(per_fr_sidx_toks_buf);
      cudaFree(per_fr_sidx_arcs_buf);
    }
    inline DEVICE void collect_tok_per_frame(TokenState* cur_toks, int size, int frame) {
      int tid=threadIdx.x+blockIdx.x*blockDim.x;
      int cur_sidx=per_fr_sidx_toks_buf[GET_IDX_BY_FRAME3(frame)]
      if (tid==0) per_fr_sidx_toks_buf[GET_IDX_BY_FRAME3(frame+1)]=cur_sidx+size;
      for (tid<size;tid+=gridDim.x*blockDim.x) {
        Token* to_tok=ActiveToksMap(frame,tid);
        store16(to_tok, cur_toks[tid].token);
      }
    }
    inline DEVICE void collect_arc_per_frame(LatLink** cur_arc_array, 
      int sub_vec_num, int* count_vec_d, int* count_vec_acc_d) {
      int tid=threadIdx.x+blockIdx.x*blockDim.x;
      int idx=tid/(sub_vec_num);
      int subid=tid%(sub_vec_num);
      int rank0=blockIdx.x==0&&threadIdx.x==0?1:0;
      int batch=blockDim.x*gridDim.x/(sub_vec_num);
      
      if (blockIdx.x==0) {
        assert(blockDim.x*gridDim.x%(sub_vec_num)==0);
        assert(blockDim.x>=sub_vec_num);
        typedef cub::BlockScan<int, sub_vec_num+1> BlockScan;
        __shared__ typename BlockScan::TempStorage temp_storage;
        BlockScan(temp_storage).ExclusiveSum(count_vec_d, count_vec_acc_d);
      }//not including self
      __grid_sync_nv_internal(barrier_);

      int cur_sidx=per_fr_sidx_arcs_buf[GET_IDX_BY_FRAME3(frame)]
      if (rank0) {
        int size=count_vec_acc_d[sub_vec_num];
        assert(size==count_vec_acc_d[sub_vec_num-1]+count_vec_d[sub_vec_num-1]);
        per_fr_sidx_arcs_buf[GET_IDX_BY_FRAME3(frame+1)]=cur_sidx+size;
      }
      int sz = count_vec_acc_d[subid+1]-count_vec_acc_d[subid];
      for(; idx < sz; idx += batch) {
        LatLink* to_arc=ActiveArcsMap(frame,(idx+count_vec_acc_d[subid]));
        store32(to_arc, cur_arc_array[subid]+idx);
      }
    }
    inline DEVICE void PruneActiveTokens(int frame, float lattice_beam) {
      if (GET_IDX_BY_FRAME(frame)!=0) return;
      int start=frame-2*prune_interval;
      int end=frame-1*prune_interval;
      for (int f = frame; f > frame-2*prune_interval; f--) {
          merge=f<=frame-1*prune_interval?1:0;
          PruneForwardLinks_PruneTokensForFrame(f,merge);
      }
    }
    #define DECODE_TOK_ADDR(frame,idx,v) { \
    frame=(((uint64)v)>>32); \
    idx=(((uint64)v)&(((uint64)1<<32)-1)); \
    }
    inline DEVICE Token* ActiveToksMap(void* p) const {
      int frame, id;
      DECODE_TOK_ADDR(frame, id, p);
      return ActiveToksMap(frame,id);
    }
    inline DEVICE Token* ActiveToksMap(int frame, int id) const {
      
      int cur_sidx=per_fr_sidx_toks_buf[GET_IDX_BY_FRAME3(frame)]
      assert(cur_sidx+id<toks_buf_before_pr_size);
      Token* tok=toks_buf_before_pr_d+cur_sidx+id;
      assert(tok->frame==frame);
      return tok;
    }
    inline DEVICE LatLink* ActiveArcsMap(int frame, int id) const {
      
      int cur_sidx=per_fr_sidx_arcs_buf[GET_IDX_BY_FRAME3(frame)]
      assert(cur_sidx+id<arcs_buf_before_pr_size);
      LatLink* arc=arcs_buf_before_pr_d+cur_sidx+id;
      //assert(arc);
      return arc;
    }
    inline DEVICE int GetSize(int* acc_len, int frame) const {
      int size=acc_len[GET_IDX_BY_FRAME3(frame+1)]-acc_len[GET_IDX_BY_FRAME3(frame)];
      assert(size>=0);
      return size;
    }
    inline DEVICE void PruneForwardLinks_PruneTokensForFrame(int frame, 
                                  bool merge, float lattice_beam) {
      //init
      {
        int tid=threadIdx.x+blockIdx.x*blockDim.x;
        int size=GetSize(per_fr_sidx_toks_buf,frame);
        for (tid<size;tid+=gridDim.x*blockDim.x) {
          Token* tok=ActiveToksMap(frame-1,tid);
          tok->extra_cost=FLT_MAX;
        }
        __grid_sync_nv_internal(barrier_);
      }

      //update arc
      {
        int tid=threadIdx.x+blockIdx.x*blockDim.x;
        int size=GetSize(per_fr_sidx_arcs_buf,frame);
        for (tid<size;tid+=gridDim.x*blockDim.x) {
          LatLink* link=ActiveArcsMap(frame,tid);
          // TODO: HOW to map toks after pruning?
          Token* next_tok=ActiveToksMap(link->p1);
          Token* tok=ActiveToksMap(link->p2);
          float link_extra_cost = next_tok->extra_cost + 
                  ((tok->tot_cost + link->acoustic_cost + link->graph_cost) 
                    - next_tok->tot_cost);
          if (link_extra_cost > lattice_beam)
            ; //should be pruned
          else
            assert(link_extra_cost>-1);
            if (link_extra_cost < tok->extra_cost)
                atomicMin(tok->extra_cost,link_extra_cost)
            if (merge) mergeArc(link);
        }
        __grid_sync_nv_internal(barrier_);
        //if we do this always in 25 frames, we might dont need this
        //some flag to show whether it is changed   
      }
      
      /*    
      //update tok
      for tok in tok[f-1]:
          if (tok->extra_cost!=FLT_MAX)
            if (merge) mergeTok(tok)
      */
      //get size 
      // TODO:
      if (merge) {
          size_arc_of_frame[f]=cidx-prev_cidx
          prev_cidx=cidx
          //size_tok_of_frame[f-1]=cidx-prev_cidx
          //prev_cidx=cidx
      }

    }
  private:
    //it's in reverse seq, it includes a header to contain sizeof arc&tok per frame, of [frame-2*prune_interval, frame-1*prune_interval-1]
    TokenOrSize* toks_buf_after_pr_d;
    int* toks_buf_after_pr_used_d;
    //it's in reverse seq; of [frame-2*prune_interval+1, frame-1*prune_interval]
    LatLink* arcs_buf_after_pr_d;
    int* arcs_buf_after_pr_used_d;
    //before
    Token* toks_buf_before_pr_d;  
    LatLink* arcs_buf_before_pr_d;
    int* per_fr_sidx_toks_buf;
    int* per_fr_sidx_arcs_buf;
    int *barrier_;

    int prune_interval;
    int toks_buf_before_pr_size;
    int arcs_buf_before_pr_size;

  };
  
  LatticePruner lattice_pruner;

  KALDI_DISALLOW_COPY_AND_ASSIGN(CudaLatticeDecoder);
};

} // end namespace kaldi.


#endif
