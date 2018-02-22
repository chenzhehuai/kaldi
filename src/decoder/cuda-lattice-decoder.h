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
    int32 *arc_ilabels_h, *arc_ilabels_d;
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
      inline void free();
      HOST DEVICE uint32_t size() const; 
      HOST DEVICE inline uint32_t push_back(const T &val); 
      HOST DEVICE inline void clear(cudaStream_t stream=0); 
      inline bool empty() const;
      inline void swap(CudaVector<T> &v); 
      inline void copy_all_to_host(cudaStream_t stream=0);
      inline void copy_all_to_device(cudaStream_t stream=0);
      inline void copy_size_to_host(cudaStream_t stream=0);
      inline void copy_size_to_device(cudaStream_t stream=0);
      inline void copy_data_to_host(cudaStream_t stream=0);
      inline void copy_data_to_device(cudaStream_t stream=0);

      inline size_t getCudaMallocBytes(); 
    private:
      
      uint32_t *count_d, *count_h;
      uint32_t max_size;
      T* mem_d, *mem_h;
};


struct CudaLatticeDecoderConfig {
  BaseFloat beam;
  double gpu_fraction;
  uint32_t max_tokens_per_frame;
  uint32_t max_lat_tok_per_frame;
  uint32_t max_lat_arc_per_frame;
  uint32_t max_tokens;
  int32_t prune_interval;
  BaseFloat lattice_beam;
  bool determinize_lattice;
  BaseFloat prune_scale;
  fst::DeterminizeLatticePhonePrunedOptions det_opts;

  int verbose;
  int32_t sub_vec_num;
  
  CudaLatticeDecoderConfig(): beam(16.0),
                       gpu_fraction(1.0/8.0),
                       max_tokens_per_frame(900000),
                       max_lat_tok_per_frame(100000),
                       max_lat_arc_per_frame(300000),
                       max_tokens(60000000),
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
    opts->Register("gpu-fraction", &gpu_fraction, "Percent of GPU to use for this LatticeDecoder.  "
                                                  "A single decoding cannot saturate the device.  "
                                                  "Use multiple LatticeDecoders in parallel for the best performance.");
    opts->Register("max-tokens-per-frame", &max_tokens_per_frame, "Maximum tokens used per frame.  If decoding exceeds this resutls are undefined.");
    opts->Register("max-tokens-allocated", &max_tokens, "Total number of tokens allocated.  This controls how many tokens are allocated to the entire decoding process."
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
  Token *prev_;
  CostType cost_; // accumulated total cost up to this point.
  int32_t arc_index_; //TODO
  //BaseFloat acoustic_cost;   //currently not recording acoustic_cost.  It is trivial to add back in but didn't seem necessary for this use case

  HOST DEVICE inline Token(BaseFloat cost, Token *prev) : prev_(prev), cost_(cost), arc_index_(-1) {
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
  volatile int arc_lock;
  //int32_t lat_tok_idx;   //-1: havent init 
  HOST DEVICE inline TokenState (Token *token, StateId state)
    : token(token), state(state), arc_lock(0) { }
  HOST DEVICE inline TokenState () : token(NULL), arc_lock(0) {};
};

typedef CudaVector<TokenState> TokenVector;

  class  LatLink {  //300000*50*24=240MB __align__(16)
   public:
    Token* prev_tok; 
    Token* next_tok;  
    int32_t arc_id;    //if <0, it's pruned, FST arcid to get Ilabel and Olabel
    // graph cost of traversing link (contains LM, etc.)
    //CostType graph_cost; //can be obtained from arc_id
    CostType acoustic_cost; // acoustic cost (pre-scaled) of traversing link

     HOST DEVICE inline LatLink(Token* iprev_tok, Token* inext_tok, int32_t iarc_id, 
      CostType iacoustic_cost): prev_tok(iprev_tok), next_tok(inext_tok),
      arc_id(iarc_id),  acoustic_cost(iacoustic_cost) { }

  };

  #define GET_ARCIDX(rawid, thd) ((rawid<<5)+thd)   //assume 32 threads
  #define GET_RAWARCIDX(id)  (id>>5) //assume 32 threads
  #define GET_THDIDX(id) (id&((1<<5)-1)) //assume 32 threads

  typedef CudaVector<LatLink> LatLinkVector;

  //Preallocates tokens and allocates them in a circular buffer.
  //This allows threads to concurrently allocate/deallocate objects quickly in CUDA
  class TokenAllocator {
    public:
      void initialize(uint32_t size);
      void finalize();

      inline void prefetch_next_to_device(cudaStream_t stream, int count);
      inline void prefetch_next_to_device(cudaStream_t stream);
      inline void prefetch_allocated_to_host(cudaStream_t stream);
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
  void ClearArcVector();
  void initParams(processTokens_params& params);
  void PreFinalizeDecoding();
  void PreProcessLattices(TokenVector** cur_toks_,
      TokenVector** prev_toks_, LatLinkVector** cur_arcs_);
  void PreProcessTokens();

  /// This will decode until there are no more frames ready in the decodable
  /// object, but if max_num_frames is >= 0 it will decode no more than
  /// that many frames.  If it returns false, then no tokens are alive,
  /// which is a kind of error state.
  void AdvanceDecoding(DecodableInterface *decodable,
                         int32 max_num_frames = -1);
  
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

 
  //token lookup table.  Provides constant time lookup for active tokens.
  //One entry per state.  If entry is NULL token is not active.
  TokenLookupElem *current_tokens_lookup_d;

  //Lists of active tokens to be iterated through
  TokenVector cur_toks_;
  TokenVector prev_toks_;



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
  cudaStream_t stream_comp, stream_copy, stream_ll;

  uint32_t total_threads;
  size_t bytes_cudaMalloc, bytes_cudaMallocManaged;

  //warp assignment indexes
  int *pe_idx_d, *ne_idx_d, *fb_idx_d;
  int *barrier_d;  //barrier to allow grid syncs
  
  int verbose;
  int prune_interval_;
  int sub_vec_num_;


  LatLinkVector* lat_arcs_vec_;
  LatLinkVector* lat_arcs_sub_vec_;




  KALDI_DISALLOW_COPY_AND_ASSIGN(CudaLatticeDecoder);
};

} // end namespace kaldi.


#endif
