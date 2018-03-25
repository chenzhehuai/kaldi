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



namespace kaldi {

// TODO:
#define LAT_BUF_SIZE 2

class CudaLatticeDecoder;

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

  template<typename T>
class CudaVector {
    public:
     inline HOST DEVICE T& operator[](uint32_t idx); 
     inline HOST DEVICE const T& operator[](uint32_t idx) const; 
     inline void allocate(uint32_t max_size, 
        uint32_t* icount_h=NULL, uint32_t* icount_d=NULL, T* mem_d=NULL, T* mem_h=NULL) ;
     inline void free(bool create_outside=false);
     inline HOST DEVICE uint32_t size() const; 
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



template<typename T>
class CudaMergeVector : public CudaVector<T> {
public:
  using CudaVector<T>::operator[];
  using CudaVector<T>::push_back;
  using CudaVector<T>::size;
  using CudaVector<T>::count_d;
  using CudaVector<T>::mem_d;
  using CudaVector<T>::max_size;
  using CudaVector<T>::clear;
  
  DEVICE inline void merge(void* undefined, int* token_per_arc_update, int num_arcs,  bool clear=true);
  DEVICE inline int update(int i);
  DEVICE inline void clear_sub();
  inline void allocate(uint32_t max_size);
  DEVICE inline uint32_t push_back(const T &val, uint64 *val_pack); 
  inline void free();
  inline size_t getCudaMallocBytes(); 
  inline void swap(CudaMergeVector<T> &v);

  //for arr merge to single; assume create using cudaMallocManaged
  int *mem_update_d;
  uint64** mem_pack_buf_d;
  T* mem_buf_d;
  int *mem_buf_count_d;
  int *mem_buf_acc_count_d;
  int* barrier_;
};
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

  HOST DEVICE inline Token(BaseFloat cost, int frame, Token* prev) : cost_(cost), frame(frame), extra_cost(0) {
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
  struct  TokenLookupElem{
    Token *token;     //pointer for that token
    uint32_t active;  //tells if token has activiated or not
    uint64_t token_pack;     //
    volatile int32_t tokenstate_idx;     //
  };

struct __align__(16) TokenState {
  public:
  
  Token* token; //arc and labels
  StateId state;  //to state
  CostType cost_; //for CPU to copy lattice without prefetch allocator
  //int32_t lat_tok_idx;   //-1: havent init 
  HOST DEVICE inline TokenState (Token *token, StateId state, CostType cost)
    : token(token), state(state), cost_(cost) { }
};

//typedef CudaVector<TokenState> TokenVector;
typedef CudaMergeVector<TokenState> TokenMergeVector;

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
    inline DEVICE void collect_arc_per_frame(LatLinkVector& cur_arc_array,
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
    TokenMergeVector prev_toks;
    TokenMergeVector cur_toks;
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
    BaseFloat beam;
    volatile int *modified;
    int *pe_idx;
    int *ne_idx;
    int *ne_queue;
    int *l_ne_idx;
    int *fb_idx;
    int *cidx2;
    int *cidx;
    int *barrier;

    //debug
    int verbose;

    uint32_t frame;
    int prune_interval;
    LatLinkVector lat_arcs_sub_vec;

    LatticePruner lattice_pruner;
    float lattice_beam;

    Token* token_per_arc;
    int* token_per_arc_update;
    int numArcs;
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

  /// InitDecoding initializes the decoding, and should only be used if you
  /// intend to call AdvanceDecoding().  If you call Decode(), you don't need
  /// to call this.  You can call InitDecoding if you have already decoded an
  /// utterance and want to start with a new utterance. 
  void InitDecoding(); 
  void ClearArcVector(LatLinkVector& lat_arcs_sub_vec_);
  void initParams(processTokens_params& params);
  void PreFinalizeDecoding(
TokenMergeVector**last_tokv,  Token** toks_buf, int** toks_sidx, LatLink** arcs_buf, int** arcs_size);
  void CallLaunchPruneActiveTokens(cudaStream_t wait_st, 
    cudaStream_t st, float ratio);

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
  TokenMergeVector* cur_toks_;
  TokenMergeVector* prev_toks_;  
  Token* token_per_arc_d;
  int *token_per_arc_update_d;

  const CudaFst fst_;

  BaseFloat beam_;
  // Keep track of the number of frames decoded in the current file.
  int32 num_frames_decoded_;

  //data store for log likelihoods needed in the current frame.  Double buffering to avoid synchronization.
  BaseFloat *loglikelihoods_h, *loglikelihoods_old_h, *loglikelihoods_d, *loglikelihoods_old_d;  

  CostType *cutoff_d;
  int *modified_d;

  //TODO
  void ClearToks(TokenMergeVector &toks);

  cudaEvent_t event_pt, event_pt_old, event_ll;
  cudaStream_t stream_comp, stream_copy[LAT_BUF_SIZE+1], stream_ll;

  uint32_t total_threads;
  size_t bytes_cudaMalloc, bytes_cudaMallocManaged;

  //warp assignment indexes
  int *pe_idx_d, *ne_idx_d, *fb_idx_d, *l_ne_idx_d, *ne_queue_d;
  int *barrier_d;  //barrier to allow grid syncs
 
  int *cidx_d,*cidx2_d; //for less NE proc
  int verbose;
  int prune_interval_;
  int max_arcs_;
  float lattice_beam_;

  //for recording lattice
  LatLinkVector* lat_arcs_sub_vec_;
  LatLinkVector lat_arcs_sub_vec_buf_;
  
  TokenMergeVector toks_buf_[LAT_BUF_SIZE];
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
