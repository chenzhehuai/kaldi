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

#include "fstext/remove-eps-local.h"
#include <algorithm>
#include <float.h>
#include <math.h>
#include <cooperative_groups.h>
#include "lattice-faster-decoder-cuda.h"
#include "decoder/cuda-lattice-decoder.h"
#include <cub/block/block_scan.cuh>
#include "math_constants.h"
#include "omp.h"


namespace kaldi {

  typedef CudaLatticeDecoder::Token Token;
  typedef CudaLatticeDecoder::StateId StateId;
  typedef CudaLatticeDecoder::TokenState TokenState;
  typedef CudaLatticeDecoder::CostType CostType;
  typedef CudaLatticeDecoder::TokenLookupElem TokenLookupElem;
  typedef CudaLatticeDecoder::LatLink LatLink;
  typedef CudaLatticeDecoder::LatLinkVector LatLinkVector;
  typedef CudaLatticeDecoder::TokenVector TokenVector;
  typedef CudaLatticeDecoder::processTokens_params processTokens_params;
  typedef CudaLatticeDecoder::LatticePruner LatticePruner;
  //template class CudaVector<LatToken>; 
  //template class CudaVector<LatLink>; 
  //http://en.cppreference.com/w/cpp/language/class_template
  template HOST DEVICE LatLink& CudaVector<LatLink>::operator[](uint32_t idx); 
  template HOST DEVICE TokenState& CudaVector<TokenState>::operator[](uint32_t idx); 
  template HOST DEVICE uint32_t  CudaVector<TokenState>::size() const; 
  template HOST DEVICE uint32_t  CudaVector<LatLink>::size() const; 

DEVICE void release_semaphore(volatile int *lock){
  *lock = 0;
  __threadfence();
  }


DEVICE void acquire_semaphore(volatile int *lock){
  short cnt=0;
  while (atomicCAS((int *)lock, 0, 1) != 0) {
    //if (++cnt==0) release_semaphore(lock); //deadlock hack
  }
  }
    DEVICE void LatticePruner::init_buf_before_cp() {
      if (threadIdx.x!=0||blockIdx.x!=0) return;
      *arcs_apr_used_d=0;
    }
    DEVICE int LatticePruner::merge_arc(LatLink* arc) {
      int i=atomicAdd(arcs_apr_used_d, 1);
      store32(arcs_apr_d+i, arc);
    }
    #define PRUNE_RATIO_ASSUME 0.25
    int LatticePruner::allocate(int max_tokens_per_frame, int max_lat_arc_per_frame, 
      int prune_interval, int max_toks, int max_arcs) {
      int sz;
      int bytes_cudaMalloc=0;
      
      //after
      sz=sizeof(int)*(prune_interval+1);
      cudaMalloc((void**)&arcs_apr_size_d,sz); bytes_cudaMalloc+=sz;
      cudaMallocHost((void**)&arcs_apr_size_h,sz);
      sz=PRUNE_RATIO_ASSUME*sizeof(LatLink)*max_arcs;
      cudaMalloc((void**)&arcs_apr_d,sz); bytes_cudaMalloc+=sz;
      cudaMallocHost((void**)&arcs_apr_h,sz); 
      /*sz=sizeof(int);
      cudaMalloc((void**)&arcs_buf_after_pr_size_arr_used_d,sz); bytes_cudaMalloc+=sz;*/
      sz=sizeof(int);
      cudaMalloc((void**)&arcs_apr_used_d,sz); bytes_cudaMalloc+=sz;
      cudaMalloc((void**)&arcs_bpr_used_d,sz); bytes_cudaMalloc+=sz;
      cudaMallocHost((void**)&arcs_apr_used_h,sz); 

      //before
      sz=sizeof(Token)*max_toks;
      cudaMalloc((void**)&toks_bpr_d,sz); bytes_cudaMalloc+=sz;
      cudaMallocHost((void**)&toks_bpr_h,sz); 
      toks_buf_before_pr_size=sz/sizeof(Token);
      sz=sizeof(LatLink)*max_arcs;
      cudaMalloc((void**)&arcs_bpr_d,sz); bytes_cudaMalloc+=sz;
      arcs_buf_before_pr_size=sz/sizeof(LatLink);
      sz=sizeof(int)*(prune_interval+1);
      cudaMalloc((void**)&toks_bpr_sidx_d,sz); bytes_cudaMalloc+=sz;
      cudaMallocHost((void**)&toks_bpr_sidx_h,sz); 
      sz=sizeof(int)*(prune_interval+1);
      cudaMalloc((void**)&arcs_bpr_sidx_d,sz); bytes_cudaMalloc+=sz;

      sz=sizeof(int);
      cudaMalloc((void**)&barrier_,sz); bytes_cudaMalloc+=sz;
      cudaMalloc((void**)&modified_d,sz); bytes_cudaMalloc+=sz;
      cudaMalloc((void**)&merge_d,sz); bytes_cudaMalloc+=sz;
      sz=sizeof(int)*(2);
      cudaMalloc((void**)&count_vec_acc_d,sz); bytes_cudaMalloc+=sz;
      this->prune_interval=prune_interval;
      
      return bytes_cudaMalloc;
    }
    void LatticePruner::free() {
      cudaFree(arcs_apr_size_d);
      cudaFreeHost(arcs_apr_size_h);
      /*cudaFree(arcs_buf_after_pr_size_arr_used_d);*/
      cudaFree(arcs_apr_d);
      cudaFree(arcs_apr_used_d);
      cudaFree(arcs_bpr_used_d);
      cudaFreeHost(arcs_apr_used_h);
      cudaFree(toks_bpr_d);
      cudaFreeHost(toks_bpr_h);
      cudaFree(arcs_bpr_d);
      cudaFree(toks_bpr_sidx_d);
      cudaFreeHost(toks_bpr_sidx_h);
      cudaFree(arcs_bpr_sidx_d);  
      
      cudaFree(count_vec_acc_d);
      cudaFree(barrier_);
      cudaFree(modified_d);
      cudaFree(merge_d);
      cudaFreeHost(arcs_apr_h);
    }
    inline DEVICE void LatticePruner::set_next_sidx(int* sidx_buf, int size, int frame) {
      assert(frame>=0);
      int cur_sidx=sidx_buf[(frame)];
      sidx_buf[(frame+1)]=cur_sidx+size;
    }
    inline DEVICE void LatticePruner::collect_tok_per_frame(TokenState* cur_toks, int size, int frame) {
      int tid=threadIdx.x+blockIdx.x*blockDim.x;
      if (tid==0) {
        set_next_sidx(toks_bpr_sidx_d, size, frame);
      }
      for (;tid<size;tid+=gridDim.x*blockDim.x) {
        Token* to_tok=ActiveToksMap(frame,tid,false);
        store16(to_tok, cur_toks[tid].token);
        //debug
        //assert(cur_toks[tid].token->frame==frame);
        //ActiveToksMap(frame,tid,true);
      }
    }
    inline DEVICE void LatticePruner::collect_arc_per_frame(LatLinkVector& cur_arc_array, 
      uint* count_vec_d, int frame) {
      int tid=threadIdx.x+blockIdx.x*blockDim.x;
      int idx=tid;
      int rank0=blockIdx.x==0&&threadIdx.x==0?1:0;
      int batch=blockDim.x*gridDim.x;
      
      int sz = *count_vec_d-*arcs_bpr_used_d;
      __grid_sync_nv_internal(barrier_);
      if (rank0) {
        int size=sz;
        set_next_sidx(arcs_bpr_sidx_d, size, frame);
        *arcs_bpr_used_d=*count_vec_d;
      }
      //in this version, we share the mem between vector&pruner, so dont need to copy
/*      
      for(; idx < sz; idx += batch) {
        LatLink* to_arc=ActiveArcsMap(frame,(idx));
        store32(to_arc, cur_arc_array.mem_d+idx);
        //debug
        ActiveToksMap((cur_arc_array.mem_d+idx)->p1,true,frame);
        ActiveToksMap(to_arc->p1,true,frame);
      }
      */
    }
template <int verbose>
    inline DEVICE void LatticePruner::PruneActiveTokens(int frame, float lattice_beam) {
      int rank0=threadIdx.x==0&&blockIdx.x==0?1:0;
      if (frame==0) return;
      init_buf_before_cp();
      __grid_sync_nv_internal(barrier_);
      for (int f = frame; f > 0; f--) {
          PruneForwardLinks_PruneTokensForFrame<verbose>(f,1,lattice_beam);
      }
      //see copy_data_to_host
      assert(*arcs_apr_used_d<arcs_buf_before_pr_size*PRUNE_RATIO_ASSUME);
      if (verbose>2&&rank0) GPU_PRINTF("PRt: %i %i\n", arcs_bpr_sidx_d[frame+1], *arcs_apr_used_d);
    }
    inline DEVICE Token* LatticePruner::ActiveToksMap(void* p, bool check, int iframe) const {
      int frame, id;
      DECODE_TOK_ADDR(frame, id, p);
      if (check) assert(frame==iframe||frame==iframe-1);
      return ActiveToksMap(frame,id,check);
    }
    inline DEVICE Token* LatticePruner::ActiveToksMap(int frame, int id, bool check) const {
      
      int cur_sidx=toks_bpr_sidx_d[frame];
      assert(cur_sidx+id<toks_buf_before_pr_size);
      Token* tok=toks_bpr_d+cur_sidx+id;
      if (check) {
        assert(tok->frame==frame);
      }
      return tok;
    }
    inline DEVICE LatLink* LatticePruner::ActiveArcsMap(int frame, int id) const {
      int cur_sidx=arcs_bpr_sidx_d[(frame)];
      assert(cur_sidx+id<arcs_buf_before_pr_size);
      LatLink* arc=arcs_bpr_d+cur_sidx+id;
      return arc;
    }
    inline DEVICE int LatticePruner::GetSize(int* acc_len, int frame) const {
      int size=acc_len[(frame)+1]-acc_len[(frame)];
      assert(size>=0&&size<=arcs_buf_before_pr_size);
      return size;
    }
template <int verbose>
    inline DEVICE void LatticePruner::PruneForwardLinks_PruneTokensForFrame(int frame, 
                                  bool merge, float lattice_beam) {
      //init
      int prev_cidx;
      int c=0;
      int rank0=threadIdx.x==0&&blockIdx.x==0?1:0;
      
        __grid_sync_nv_internal(barrier_);
      if (rank0&&verbose>3) GPU_PRINTF("%i %i\n",c++, GetSize(toks_bpr_sidx_d,frame-1));
      {
        int tid=threadIdx.x+blockIdx.x*blockDim.x;
        int size=GetSize(toks_bpr_sidx_d,frame-1);
        for (;tid<size;tid+=gridDim.x*blockDim.x) {
          Token* tok=ActiveToksMap(frame-1,tid,true);
          tok->extra_cost=FLT_MAX;
        }
        if (rank0) {//wait for last iteration(frame+1) finish
          *modified_d=1;
          prev_cidx=*arcs_apr_used_d;
        }
        if (rank0&&verbose>3) GPU_PRINTF("%i %p\n",c++,barrier_);
        __grid_sync_nv_internal(barrier_);
      }
      if (rank0&&verbose>3) GPU_PRINTF("%i\n",c++);

      //update arc //need some loop here
      int cnt=0;
      while (cnt++<10&&*modified_d!=0){
        int tid=threadIdx.x+blockIdx.x*blockDim.x;
        int size=GetSize(arcs_bpr_sidx_d,frame);
        if (rank0) *modified_d=0;
        __grid_sync_nv_internal(barrier_);
        for (;tid<size;tid+=gridDim.x*blockDim.x) {
          LatLink* link=ActiveArcsMap(frame,tid);
          Token* next_tok=ActiveToksMap(link->p1,true,frame);
          Token* tok=ActiveToksMap(link->p2, true,frame);
          float link_extra_cost = next_tok->extra_cost + 
                  ((tok->cost_ + link->acoustic_cost + link->graph_cost) 
                    - next_tok->cost_);
          if (isnan(link_extra_cost) || link_extra_cost > lattice_beam)
            ; //should be pruned
          else {
            //debug
            if (link_extra_cost<-1) {
              GPU_PRINTF("%i %f %f %f %f %f\n",frame,next_tok->extra_cost, tok->cost_, link->acoustic_cost, link->graph_cost, next_tok->cost_);
            }
            if (link_extra_cost < tok->extra_cost) {
              atomicMin(&tok->extra_cost,link_extra_cost);
              if (*modified_d==0) atomicAdd(modified_d,1);
            }
          }
        }
        __grid_sync_nv_internal(barrier_);
        //if we do this always in 25 frames, we might dont need this
        //some flag to show whether it is changed   
      }
      if (rank0&&verbose>3) GPU_PRINTF("cnt: %i\n",cnt);
      {
        int tid=threadIdx.x+blockIdx.x*blockDim.x;
        int size=GetSize(arcs_bpr_sidx_d,frame);
        for (;tid<size;tid+=gridDim.x*blockDim.x) {
          LatLink* link=ActiveArcsMap(frame,tid);
          Token* next_tok=ActiveToksMap(link->p1,true,frame);
          Token* tok=ActiveToksMap(link->p2, true,frame);
          float link_extra_cost = next_tok->extra_cost + 
                  ((tok->cost_ + link->acoustic_cost + link->graph_cost) 
                    - next_tok->cost_);
          if (isnan(link_extra_cost) || link_extra_cost > lattice_beam)
            ; //should be pruned
          else {
            if (merge) {
              merge_arc(link);
              //if have seen, we can delete this
              //link->acoustic_cost=CUDART_NAN_F; 
            }
          }
        }
        __grid_sync_nv_internal(barrier_);
      }

      
      /*{
        //update tok
        int tid=threadIdx.x+blockIdx.x*blockDim.x;
        int size=GetSize(toks_bpr_sidx_d,frame);
        for (;tid<size;tid+=gridDim.x*blockDim.x) {
          Token* tok=ActiveToksMap(frame-1,tid);
          if (tok->extra_cost==FLT_MAX)
            tok->tot_cost=CUDART_NAN_F; //prune
        }        
      } */   
      
      //get size 
      if (merge&&rank0) {
          int& size_arc_of_frame=arcs_apr_size_d[frame];
          size_arc_of_frame=*arcs_apr_used_d-prev_cidx;
          if (verbose>3) GPU_PRINTF("PR %i %i %i\n",frame, 
            GetSize(arcs_bpr_sidx_d,frame), size_arc_of_frame);
          //size_tok_of_frame[f-1]=cidx-prev_cidx
          //prev_cidx=cidx
      }
      __grid_sync_nv_internal(barrier_);
        if (rank0&&verbose>3) GPU_PRINTF("%i %p\n",c++,barrier_);
    }
    //#define GET_ARC_BUF_HOST_BY_FRAME(frame) (arcs_apr_h+arcs_apr_h_used)

    void LatticePruner::init() {
      //arcs_apr_h_used=0;
      cudaMemset(arcs_apr_size_d,0,sizeof(int)*(prune_interval+1));
      cudaMemset(arcs_apr_used_d,0,sizeof(int));
      cudaMemset(arcs_bpr_used_d,0,sizeof(int));      
      cudaMemset(toks_bpr_sidx_d,0,sizeof(int)*(prune_interval+1));
      cudaMemset(arcs_bpr_sidx_d,0,sizeof(int)*(prune_interval+1));
    }
    void LatticePruner::copy_arcs_to_host(int frame, cudaStream_t st) {
      int sz;
      //TODO: optimize out this
      cudaMemcpy(arcs_apr_used_h,arcs_apr_used_d,
        sizeof(int), cudaMemcpyDeviceToHost);
      //
      sz=sizeof(LatLink)*(*arcs_apr_used_h);
      //sz=sizeof(LatLink)*(arcs_buf_before_pr_size*0.5); //assume 0.5 parts are pruned
      cudaMemcpyAsync(arcs_apr_h,arcs_apr_d,
        sz, cudaMemcpyDeviceToHost, st);
      //we can call it because currently *arcs_buf_after_pr_size_arr_used_d is static len, 
      //which is only used to save size but not any token
      sz=sizeof(int)*(frame+1)*(1);
      cudaMemcpyAsync(arcs_apr_size_h,arcs_apr_size_d,
        sz, cudaMemcpyDeviceToHost, st);
      // call init_buf_before_cp(); in GPU
    }
    void LatticePruner::copy_toks_to_host(int frame, cudaStream_t st) {
      int sz;
      sz=sizeof(int)*(frame+1+1)*(1); //include frame 0 & finalsum
      cudaMemcpy(toks_bpr_sidx_h,toks_bpr_sidx_d,
        sz, cudaMemcpyDeviceToHost);
      sz=sizeof(Token)*(toks_bpr_sidx_h[frame+1]);
      assert(sz);
      cudaMemcpyAsync(toks_bpr_h,toks_bpr_d,
        sz, cudaMemcpyDeviceToHost,st);
    }

    void LatticePruner::get_data_copied_to_host(Token** toks_buf, int** toks_sidx, LatLink** arcs_buf, int** arcs_size) {
      //copy the real len 
      *toks_sidx=toks_bpr_sidx_h;
      *toks_buf=toks_bpr_h;
      *arcs_size=arcs_apr_size_h;  //next prune_interval len
      *arcs_buf=arcs_apr_h; //start of next prune_interval len arcs
    }

  DEVICE inline void allocateAllTokens_function(TokenLookupElem *current_tokens_lookup, int32 numStates,  CudaLatticeDecoder::TokenAllocator allocator) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<numStates; i+=blockDim.x*gridDim.x) {
      Token *token = allocator.getToken(i);
      token->cost_ = INFINITY;
      token->extra_cost = 0;
      token->frame= -1;
      token->state_id= -1;
      TokenLookupElem elem;
      elem.token=token;
      elem.active=false;
      elem.tokenstate_idx=-1;
      store16(&current_tokens_lookup[i], &elem);
    }
  }
  __global__ void allocateAllTokens(TokenLookupElem *current_tokens_lookup, int32 numStates,  CudaLatticeDecoder::TokenAllocator allocator, int *barrier) {
    allocateAllTokens_function(current_tokens_lookup,numStates,allocator);
     __grid_sync_nv_internal(barrier);
     if(blockIdx.x==0 && threadIdx.x==0) {
      allocator.advanceFront(numStates);
     }
  }

  DEVICE inline void allocateNewTokens_function(TokenLookupElem *current_tokens_lookup, TokenVector cur_toks, CudaLatticeDecoder::TokenAllocator allocator) {
    int32 size = cur_toks.size();
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=blockDim.x*gridDim.x) {
      Token *token = allocator.getToken(i);
      token->cost_ = INFINITY;
      token->extra_cost = 0;
      token->frame= -1;
      token->state_id= -1;
      //a CPU copy of cur_toks can still be used, cur_toks will be clear in PreProcessTokens 
      //lat_arcs_sub_vec_ is clearred in PreProcessTokens
      StateId state=cur_toks[i].state;  
      //cur_toks[i].token->arc_index_=-1; // clear here will result in page fault in prefetch
      TokenLookupElem elem;
      elem.token=token;
      elem.active=false;
      elem.tokenstate_idx=-1;
      store16(&current_tokens_lookup[state], &elem);
    }
  }

  
  void CudaLatticeDecoder::TokenAllocator::prefetch_next_to_device(cudaStream_t stream) {
    prefetch_next_to_device(stream,prefetch_size);
  }

  void CudaLatticeDecoder::TokenAllocator::prefetch_next_to_device(cudaStream_t stream, int count) {
    int front = *front_h;
    //clamp to maximum size
    if(count>size-front)
      count = size-front;

#ifdef MEMADVISE
    cudaMemPrefetchAsync(tokens_allocation+front,sizeof(Token)*count,device,stream);  
#endif
  }

  void CudaLatticeDecoder::TokenAllocator::prefetch_allocated_to_host_force(cudaStream_t stream) {
    if (!*front_h) return;
    cudaMemcpyAsync(tokens_allocation, tokens_allocation,sizeof(Token)* *front_h,cudaMemcpyDeviceToHost, stream);
  }

  void CudaLatticeDecoder::TokenAllocator::prefetch_allocated_to_host(cudaStream_t stream) {
    nvtxRangePushA("prefetch_allocated_to_host"); 
#ifdef MEMADVISE
    if (!*front_h) return;
    cudaMemPrefetchAsync(tokens_allocation,sizeof(Token)* *front_h,cudaCpuDeviceId,stream);  
#endif
    nvtxRangePop();
  }
  void CudaLatticeDecoder::TokenAllocator::prefetch_allocated_to_host_since_last(cudaStream_t stream) {
#ifdef MEMADVISE
    if (*last2_front_h==*front_h) return;
    cudaMemPrefetchAsync(tokens_allocation+*last2_front_h,sizeof(Token)* (*front_h-*last2_front_h),cudaCpuDeviceId,stream); 
    *last2_front_h=*last_front_h;
    *last_front_h=*front_h;
#endif
  }


  size_t CudaLatticeDecoder::TokenAllocator::getCudaMallocManagedBytes() {
    return bytes_cudaMallocManaged;
  }

  void CudaLatticeDecoder::TokenAllocator::reset() {
    *front_h=0;
    *last_front_h=0;
    *last2_front_h=0;
    cudaMemset(front_d,0,sizeof(int));
  }

  void CudaLatticeDecoder::TokenAllocator::initialize(uint32_t size)  {
    cudaGetDevice(&device);
    prefetch_size=250000;

    this->size = size;

    //managed so getBestPath can easily access this data in the end
    cudaMallocManaged((void**)&tokens_allocation,sizeof(Token)*size);  
    bytes_cudaMallocManaged=sizeof(Token)*size;

    cudaMalloc((void**)&front_d,sizeof(uint32_t)); 
    cudaMallocHost((void**)&front_h,sizeof(uint32_t)); 
    cudaMallocHost((void**)&last_front_h,sizeof(uint32_t)); 
    cudaMallocHost((void**)&last2_front_h,sizeof(uint32_t)); 

#ifdef MEMADVISE
    //If we do this we get faster perf as long as we don't over subscribe
    cudaMemAdvise(tokens_allocation,sizeof(Token)*size,cudaMemAdviseSetPreferredLocation,device);
    cudaMemPrefetchAsync(tokens_allocation,sizeof(Token)*size,device);  //force pages to allocate now
#endif

    reset();
  }

  void CudaLatticeDecoder::TokenAllocator::finalize() {
    printf("TokenAllocator::finalize()\n");
    cudaFree(tokens_allocation);
    cudaFree(front_d);
    cudaFreeHost(front_h);
    cudaFreeHost(last_front_h);
    cudaFreeHost(last2_front_h);
  }

  DEVICE inline Token* CudaLatticeDecoder::TokenAllocator::getToken(uint32_t offset) {
    int idx = *front_d + offset;
    return &tokens_allocation[idx];
  }

  DEVICE inline void CudaLatticeDecoder::TokenAllocator::advanceFront(uint32_t num) {
    int front = *front_d + num;
    assert(front<size);
    
    *front_d=front;
    *front_h=front;
  }


  CudaLatticeDecoder::CudaLatticeDecoder(const CudaFst &fst, const CudaLatticeDecoderConfig &config): fst_(fst), beam_(config.beam), bytes_cudaMalloc(0), bytes_cudaMallocManaged(0) {
    printf("CudaLatticeDecoder Constructor\n");
    int device;
    cudaGetDevice(&device);

    if (verbose>4) cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1e7);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,device);

    total_threads = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount * config.gpu_fraction;

    allocator.initialize(config.max_tokens);
    cudaCheckError();
    bytes_cudaMallocManaged+=allocator.getCudaMallocManagedBytes();

    cudaEventCreateWithFlags(&event_pt,cudaEventDisableTiming);
    cudaEventCreateWithFlags(&event_pt_old,cudaEventDisableTiming);
    cudaEventCreateWithFlags(&event_ll,cudaEventDisableTiming);

    cudaStreamCreateWithFlags(&stream_comp, cudaStreamNonBlocking);
    for (int i=0; i<LAT_BUF_SIZE+1; i++) 
      cudaStreamCreateWithFlags(&stream_copy[i], cudaStreamNonBlocking);    
    cudaStreamCreateWithPriority(&stream_ll, cudaStreamNonBlocking, -1);

    cudaMalloc(&pe_idx_d, sizeof(int)); bytes_cudaMalloc+=sizeof(int);
    cudaMalloc(&ne_idx_d, sizeof(int)); bytes_cudaMalloc+=sizeof(int);
    cudaMalloc(&fb_idx_d, sizeof(int)); bytes_cudaMalloc+=sizeof(int);
    cudaMalloc(&barrier_d, sizeof(int)); bytes_cudaMalloc+=sizeof(int);

    cudaMemset(pe_idx_d,0,sizeof(int));
    cudaMemset(ne_idx_d,0,sizeof(int));
    cudaMemset(fb_idx_d,0,sizeof(int));
    cudaMemset(barrier_d,0,sizeof(int));
    cudaCheckError();

    cudaMalloc(&cutoff_d, sizeof(CostType)); bytes_cudaMalloc+=sizeof(CostType);
    cudaMalloc(&modified_d, sizeof(int)*2); bytes_cudaMalloc+=sizeof(CostType)*2;

    cudaMalloc(&token_locks_d,sizeof(int)*fst_.numStates);  bytes_cudaMalloc+=sizeof(int)*fst_.numStates;
    cudaMemset((void*)token_locks_d,0,sizeof(int)*fst_.numStates);

    cudaMalloc((void**)&current_tokens_lookup_d,sizeof(TokenLookupElem)*fst_.numStates); bytes_cudaMalloc+=sizeof(TokenLookupElem)*fst_.numStates;

    cudaMallocHost(&loglikelihoods_h,sizeof(BaseFloat)*(fst_.max_ilabel+1));  
    cudaMallocHost(&loglikelihoods_old_h,sizeof(BaseFloat)*(fst_.max_ilabel+1));

    cudaMalloc((void**)&loglikelihoods_d,sizeof(BaseFloat)*(fst_.max_ilabel+1)); bytes_cudaMalloc+=sizeof(BaseFloat)*(fst_.max_ilabel+1);
    cudaMalloc((void**)&loglikelihoods_old_d,sizeof(BaseFloat)*(fst_.max_ilabel+1)); bytes_cudaMalloc+=sizeof(BaseFloat)*(fst_.max_ilabel+1);

    verbose=config.verbose;
    prune_interval_=config.prune_interval;
    max_arcs_=config.max_arcs;

    //for pruning
    bytes_cudaMalloc+=lattice_pruner_.allocate(config.max_tokens_per_frame, 
                      config.max_lat_arc_per_frame, config.prune_interval,
                      config.max_tokens, config.max_arcs);
    lattice_beam_=config.lattice_beam;

    lat_arcs_sub_vec_buf_.allocate(
      config.max_arcs,
      NULL,
      NULL, 
      lattice_pruner_.GetArcBpr(),
      NULL); 
    bytes_cudaMalloc += lat_arcs_sub_vec_buf_.getCudaMallocBytes();  

    for (int j=0; j<LAT_BUF_SIZE; j++) {
      toks_buf_[j].allocate(config.max_tokens_per_frame);
      bytes_cudaMalloc+=toks_buf_[j].getCudaMallocBytes();
      
    }
    num_frames_decoded_=0;

    SetTokArcPointerByFrame(num_frames_decoded_);
    
    cudaStreamSynchronize(stream_comp);
    cudaStreamSynchronize(stream_copy[0]);
    cudaStreamSynchronize(cudaStreamPerThread);
    //sgemm requires shared memory and we don't want cache config changing.  So set a device wide cache config.
    cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
  }

  CudaLatticeDecoder::~CudaLatticeDecoder() {

    printf("CUDA LatticeDecoder DESTRUCTOR\n");

    for (int j=0; j<LAT_BUF_SIZE; j++) {
      toks_buf_[j].free();
    }
    lat_arcs_sub_vec_buf_.free(true);
    
    lattice_pruner_.free();    
    
    allocator.finalize();

    cudaFreeHost(loglikelihoods_h);
    cudaFreeHost(loglikelihoods_old_h);
    cudaFree(loglikelihoods_d);
    cudaFree(loglikelihoods_old_d);
    cudaFree(current_tokens_lookup_d);

    cudaFree(pe_idx_d);
    cudaFree(ne_idx_d);
    cudaFree(fb_idx_d);
    cudaFree(barrier_d);

    cudaFree((void*)token_locks_d);
    cudaFree(cutoff_d);
    cudaFree(modified_d);

    cudaEventDestroy(event_pt);
    cudaEventDestroy(event_pt_old);
    cudaEventDestroy(event_ll);

    cudaStreamDestroy(stream_comp);
    for (int i=0; i<LAT_BUF_SIZE+1; i++) 
      cudaStreamDestroy(stream_copy[i]);
    cudaStreamDestroy(stream_ll);

  }
  DEVICE inline Token* FindOrAddTokenArc(processTokens_params& params,
    StateId nextstate, CostType total_cost, CostType acoustic_cost,
    TokenState* ts, uint32_t j, bool add_arc, TokenState** next_ts) {
    //TokenLookupElem lookup_elem;
    //load16(&lookup_elem, &params.current_tokens_lookup[nextstate]);
    TokenLookupElem& lookup_elem = params.current_tokens_lookup[nextstate];
    Token *cur_tok = lookup_elem.token;  
    //check if token is active or not.  Double check the lock.
    if(lookup_elem.active==0 && atomicCAS(&lookup_elem.active,0,1)==0) {        //grab sentinal to see who gets to add to cur_toks list
      //if havent seen, add into hash
      lookup_elem.tokenstate_idx=params.cur_toks.push_back(TokenState(cur_tok,nextstate,total_cost));
    }
    //need both 2 steps below, to ensure tokenstate_idx won't run into error
    while (lookup_elem.tokenstate_idx == -1);//hasnt pushed
    __threadfence(); 
    *next_ts=&params.cur_toks[lookup_elem.tokenstate_idx];
    if (add_arc) {
      Token *prev_tok = ts->token;  
      int ts_id=prev_tok->frame==params.frame?
      params.cur_toks.get_idx_from_addr(ts):
      params.prev_toks.get_idx_from_addr(ts);
      LatLink arc=LatLink(ts_id, prev_tok->frame, 
        lookup_elem.tokenstate_idx, params.frame,
        params.arc_ilabels[j], params.arc_olabels[j],
        params.arc_weights[j], acoustic_cost); //duplicate arcs in NE
      int32_t lat_arc_idx=params.lat_arcs_sub_vec.push_back(arc);
    }
    return cur_tok;  
  }
  __global__ void addOneToken(processTokens_params params, StateId state) {
    TokenState *next_ts=NULL;
    Token* cur_tok=FindOrAddTokenArc(params, state, 0, //add first token
      0, NULL, -1, false,  &next_ts);
    Token tok(0, 0, NULL, state);
    *cur_tok = tok;
    cur_tok->frame=params.frame;
  }

  //putting this into a kernel to avoid extra latency of a memory copy
  __global__ void initializeCutoff(CostType *cutoff) {
    *cutoff = INFINITY;
  }
  void CudaLatticeDecoder::ClearArcVector(LatLinkVector& lat_arcs_sub_vec_) {
    int i=0;
    lat_arcs_sub_vec_.clear();
  }
  void CudaLatticeDecoder::InitDecoding() {
    printf("CUDA LatticeDecoder InitDecoding\n");
    num_frames_decoded_ = 0;
  // clean up from last time:
    for (int i=0; i<LAT_BUF_SIZE; i++) {
      ClearToks(toks_buf_[i]);
    }    
    ClearArcVector(lat_arcs_sub_vec_buf_);
    
    SetTokArcPointerByFrame(num_frames_decoded_);
    lattice_pruner_.init();

    allocator.reset();
    int threads=64;
    int blocks=DIV_ROUND_UP(total_threads,threads);
    
    //start moving these / allocating them on the device
    allocator.prefetch_next_to_device(stream_comp, fst_.numStates+5000);

    allocateAllTokens<<<blocks,threads,0,stream_comp>>>(current_tokens_lookup_d, fst_.numStates, allocator, barrier_d);

    // initialize decoding:
    StateId start_state = fst_.Start();
    KALDI_ASSERT(start_state != fst::kNoStateId);

    cudaCheckError();
    //Token tok(StdWeight::One().Value(), NULL, 0);
    //Token tok(StdWeight::One().Value(),0, NULL, 0);
    processTokens_params params;
    initParams(params);
    addOneToken<<<1,1,0,stream_comp>>>(params, start_state);

    cudaCheckError();

    initializeCutoff<<<1,1,0,stream_comp>>>(cutoff_d);
    ProcessNonemitting();
    if (verbose>3) printf("CUDA LatticeDecoder End of InitDecoding\n");
  }

  void CudaLatticeDecoder::PreFinalizeDecoding(TokenVector** last_tokv,
    Token** toks_buf, int** toks_sidx, LatLink** arcs_buf, int** arcs_size) { 
    cudaStreamSynchronize(stream_comp);//after fini comp. we can start copy 
    lattice_pruner_.copy_toks_to_host(num_frames_decoded_, stream_copy[1]);
    CallLaunchPruneActiveTokens(stream_comp, stream_copy[0], 1);
    (toks_buf_[num_frames_decoded_%LAT_BUF_SIZE]).copy_data_to_host(stream_copy[2]);//copy data in post
    *last_tokv=&(toks_buf_[num_frames_decoded_%LAT_BUF_SIZE]);
    if (verbose>1) KALDI_LOG<<"";
    cudaStreamSynchronize(stream_copy[0]);
    if (verbose>1) KALDI_LOG<<"";
    lattice_pruner_.copy_arcs_to_host(num_frames_decoded_, stream_copy[0]);
    cudaStreamSynchronize(stream_copy[0]);
    if (verbose>1) KALDI_LOG<<"";
    cudaStreamSynchronize(stream_copy[1]);
    cudaStreamSynchronize(stream_copy[2]);
    lattice_pruner_.get_data_copied_to_host(toks_buf, toks_sidx, arcs_buf, arcs_size);
  }

  bool CudaLatticeDecoder::ReachedFinal() const {
    for (int i=0;i<cur_toks_->size();i++) {
      TokenState ts = (*cur_toks_)[i];

      if (ts.token->cost_ != std::numeric_limits<BaseFloat>::infinity() &&
          fst_.Final(ts.state) != StdWeight::Zero())
        return true;
    }

    return false;
  }

  // Outputs an FST corresponding to the single best path
  // through the lattice.
  bool CudaLatticeDecoder::GetBestPath(Lattice *fst_out, bool use_final_probs) const {
    nvtxRangePushA("GetBestPath");
    assert(0);
    return true;
  }

  void CudaLatticeDecoder::ComputeLogLikelihoods(DecodableInterface *decodable) {
    PUSH_RANGE("ComputeLogLikelihoods",3)
    int32 frame = num_frames_decoded_;
    //cudaStreamSynchronize(stream_comp); //finish decoding this frame
    std::swap(loglikelihoods_h,loglikelihoods_old_h); //double buffering so we don't overwrite loglikelihoods_h before it is copied down
    std::swap(loglikelihoods_d,loglikelihoods_old_d); //double buffer

    //We really only need about 10% of these but finding out which 10% is more expensive then just computing all of them
    //Computing them inline in the next loop leads to lots of redundant computation
    decodable->ComputeLogLikelihoods(loglikelihoods_h,frame,fst_.max_ilabel+1);

    //copying in another stream to overlap transfer with compute
    cudaMemcpyAsync(loglikelihoods_d,loglikelihoods_h,sizeof(BaseFloat)*(fst_.max_ilabel+1),cudaMemcpyHostToDevice, stream_ll);

    cudaEventRecord(event_ll,stream_ll);  //mark log likelihoods are copied down to the device
    //cudaStreamWaitEvent(stream_comp,event_ll,0); //ensure logliklihoods_d is updated before consuming; wait in ProcessTokens

    POP_RANGE
  }

  //structs to hold kernel parameters.  Large numbers of parameters can slow down launch latency which matters when we are launching very short kernels

  //blockDim.x threads per token
  template<int blockDimx, int blockDimy>
  inline DEVICE void findBestCutoff_function(processTokens_params params) {

    int threadIdxy = threadIdx.x / blockDimx;

    auto group = cooperative_groups::tiled_partition<blockDimx>(cooperative_groups::this_thread_block());

    CostType local_cutoff = INFINITY;
    int32 size = params.prev_toks.size(); 

    //uses dynamically load balanced loop trips.  Tokens are assigned dynamically instead of statically
    while(true) { 
      int i;
      if(group.thread_rank()==0) { //thread 0 nominated to get new token
        i=atomicAdd(params.fb_idx,1);      //get token index
      }
      i=group.shfl(i,0);           //broadcast token index
      //i=__shfl_sync(0xffffffff,i,0);
      if(i>=size) break;  //Work complete
      
      TokenState ts = params.prev_toks[i];
      Token * tok = ts.token;
      StateId state = ts.state;

      uint32_t start=params.e_offsets[state], finish=params.e_offsets[state+1];
      
      int32 ilabel, ilabel_next;

      int j=start+group.thread_rank();

      if(j<finish) {
        ilabel_next = params.arc_ilabels[j];
      }
      int nextj;

      for(j;j<finish;j=nextj) {
        nextj = j+blockDimx;
        ilabel = ilabel_next;
        if(nextj<finish) {
          ilabel_next = params.arc_ilabels[nextj];
        }
        
        BaseFloat acoustic_cost = -params.loglikelihoods[ilabel]; //TODO can I prefetch this?
        CostType weight = params.arc_weights[j];
        
        CostType total_cost = tok->cost_ + weight + acoustic_cost + params.beam;

        if(total_cost<local_cutoff)
          local_cutoff = total_cost;
      }
    }

    //TODO reduce inside block first?
    if(local_cutoff!=INFINITY) {
      atomicMin(params.cutoff, local_cutoff);
    }
  }


  //blockDim.x threads per token
  template<int blockDimx, int blockDimy>
  inline DEVICE void processEmittingTokens_function(processTokens_params& params) {
    int threadIdxy = threadIdx.x / blockDimx;
    
    auto group = cooperative_groups::tiled_partition<blockDimx>(cooperative_groups::this_thread_block());

    CostType cutoff=*params.cutoff;
    int32 size = params.prev_toks.size();
    //uses dynamically load balanced loop trips.  Tokens are assigned dynamically instead of statically
    while(true) {
      int i;
      if(group.thread_rank()==0) { //thread 0 nominated to get new token
        i=atomicAdd(params.pe_idx,1);      //get token index
        if (params.verbose>3 && i%1000==0) {
          GPU_PRINTF("E: %i %i %i\n", i, threadIdx.x, blockIdx.x);
        }
      }
      i=group.shfl(i,0);           //broadcast token index
      //i=__shfl_sync(0xffffffff,i,0);
      if(i>=size) break;

      TokenState& ts = params.prev_toks[i];
      Token * tok = ts.token;
      StateId state = ts.state;

      uint32_t start=params.e_offsets[state], finish=params.e_offsets[state+1];
      int32 ilabel, ilabel_next;  //prefetch ilabel since it leads to a dependent load

      int j=start+group.thread_rank();

      if(j<finish) {
        ilabel_next = params.arc_ilabels[j];
      }
      int nextj;

      for(j;j<finish;j=nextj) {
        nextj = j+blockDimx;

        ilabel = ilabel_next;

        if(nextj<finish) {
          ilabel_next = params.arc_ilabels[nextj];//prefetch ilabel since it leads to a dependent load 
        }
        BaseFloat acoustic_cost = -params.loglikelihoods[ilabel];  //TODO can I prefetch this?  
        BaseFloat weight = params.arc_weights[j];
        StateId nextstate = params.arc_nextstates[j];

        CostType total_cost = tok->cost_ + weight + acoustic_cost;

        if(total_cost<=cutoff) 
        {
          Token next_tok =  Token(acoustic_cost+weight, params.frame, tok, nextstate);
          TokenState *next_ts=NULL;
          Token *cur_tok = FindOrAddTokenArc(params, nextstate, total_cost, 
            acoustic_cost, &ts, j, true, &next_ts);
          
          volatile Token* cur_tokv = reinterpret_cast<volatile Token*>(cur_tok);  //need volatile reads to ensure we don't get cached versions

          while(*cur_tokv < next_tok) {   //check if we need to update
          acquire_semaphore((int*)&params.token_locks[nextstate]);
              if(*cur_tokv < next_tok) {                                                                          //recheck if we are min           

                if(sizeof(Token)==16)
                  store16(cur_tok,&next_tok);                                                                       //update token
                else
                  *cur_tok=next_tok;
                /*cur_tok->cost_=next_tok.cost_;
                cur_tok->frame=params.frame;*/
                next_ts->cost_=cur_tok->cost_;
              }
              release_semaphore((int*)&params.token_locks[nextstate]);
              break;                                                                                              //exit loop as our update is done
          } //end while
        } //end total_cost<=cutoff
      } //end arc loop
    } //end token loop
  }
  
    template<int blockDimx, int blockDimy>
  DEVICE __inline__ void processNonEmittingTokens_function(processTokens_params& params, CostType cutoff, uint32_t size,  volatile int *modified) {
    
    auto group = cooperative_groups::tiled_partition<blockDimx>(cooperative_groups::this_thread_block());

    int threadIdxy = threadIdx.x / blockDimx;

    //uses dynamically load balanced loop trips.  Tokens are assigned dynamically instead of statically
    while(true) {
      int i;
      if(group.thread_rank()==0) { //thread 0 nominated to get new token
        i=atomicAdd(params.ne_idx,1);      //get token index
        if (params.verbose>3 && i%1000==0) {
          GPU_PRINTF("NE: %i %i %i\n", i, threadIdx.x, blockIdx.x);
        }
      }
      i=group.shfl(i,0);           //broadcast token index
      //i=__shfl_sync(0xffffffff,i,0);
      if(i>=size) break;
      
      TokenState& ts = params.cur_toks[i];
      Token * tok = ts.token;
      StateId state = ts.state;
      

      uint32_t start=params.ne_offsets[state], finish=params.ne_offsets[state+1];
      for(int j=start+group.thread_rank();j<finish;j+=blockDimx) {
        BaseFloat weight = params.arc_weights[j];
        StateId nextstate = params.arc_nextstates[j];

        Token next_tok = Token(weight, params.frame, tok, nextstate);

        CostType total_cost = tok->cost_ + weight;

      if (params.verbose>4) GPU_PRINTF("D: %i %i %i %i %i \n",threadIdx.x, threadIdx.y, j, blockIdx.x,i);
        if (next_tok.cost_ <= cutoff) {
          TokenState *next_ts=NULL;
          Token *cur_tok = FindOrAddTokenArc(params, nextstate, total_cost, 
            0, &ts, j, true, &next_ts);

          volatile Token* cur_tokv = reinterpret_cast<volatile Token*>(cur_tok);  //need volatile reads to ensure we don't get cached versions

          while(*cur_tokv < next_tok) {   //check if we need to update
            acquire_semaphore((int*)&params.token_locks[nextstate]);
              if(*cur_tokv < next_tok) {                                                                     //recheck that we are minimum
                if(sizeof(Token)==16)
                  store16(cur_tok,&next_tok);                                                                       //update token
                else
                  *cur_tok=next_tok;
                /*cur_tok->cost_=next_tok.cost_;
                cur_tok->frame=params.frame;*/
                next_ts->cost_=cur_tok->cost_;
                (*modified) = true;                                                                            //mark as updated
              }
            release_semaphore((int*)&params.token_locks[nextstate]);
              break;  //exit loop as our update is done
          } //end try update loop
        }
      }

    }
      if (params.verbose>4) GPU_PRINTF("ED: %i %i %i \n",threadIdx.x, group.thread_rank(), blockIdx.x);
  }

  __launch_bounds__(64,64)
  __global__ void processTokens_cg(processTokens_params params, bool is_init=false) {
    bool rank0 = blockIdx.x==0 && threadIdx.x==0;

    if (!is_init) {
      findBestCutoff_function<32,2>(params);
      __grid_sync_nv_internal(params.barrier);
    }
   
    volatile int *modified0 = params.modified;    //modified flag for current iteration
    volatile int *modified1 = params.modified+1;  //modified flag for next/last iteration
    *modified1 = false;
    CostType cutoff=*params.cutoff;

    if (!is_init) {
      processEmittingTokens_function<32,2>(params);
      __grid_sync_nv_internal(params.barrier);  //ensure cur_toks size is final
    }
  
    int tok_E;
    int itv = params.verbose>2? 1: 10;
    if (rank0&&params.verbose>1&&params.frame%itv==0) 
      tok_E=params.cur_toks.size();

    do {

      uint32_t size = params.cur_toks.size();

      *params.ne_idx=0;

      //grid.sync();  
      __grid_sync_nv_internal(params.barrier); //wait for everyone to read size and modified0

      //swap buffers
      swap(modified0,modified1); //double buffered to avoid extra sync when resetting modified to false

      *modified1 = false;

      processNonEmittingTokens_function<32,2>(params,cutoff,size,modified0);

      //grid.sync();
      __grid_sync_nv_internal(params.barrier);  //wait for everyone to finish process tokens and writes modified0

    } while ((*modified0)==true);

    if (rank0&&params.verbose>1&&params.frame%itv==0) 
          GPU_PRINTF("TK: %i %i %i %f\n", params.frame, tok_E, params.cur_toks.size(), cutoff);


    //proc lattice before allocate new toks to TokenState
    params.lattice_pruner.collect_tok_per_frame(params.cur_toks.mem_d, 
                  *params.cur_toks.count_d, params.frame);
    params.lattice_pruner.collect_arc_per_frame(params.lat_arcs_sub_vec, 
      params.lat_arcs_sub_vec.count_d, params.frame);
    

    __grid_sync_nv_internal(params.barrier); 

    allocateNewTokens_function(params.current_tokens_lookup, params.cur_toks, params.allocator);
  
    if(rank0) {
      //prepare for next iteration
      //params.prev_toks.clear(); //change to be done in PreProcessTokens
      *params.cutoff = INFINITY;
      *params.fb_idx=0;  
      *params.pe_idx=0;
    }
    
    __grid_sync_nv_internal(params.barrier);  //wait for allocation to finish
    
    if(rank0) {
      params.allocator.advanceFront(params.cur_toks.size());
    }
  }

  __launch_bounds__(64,64)
  __global__ void PostProcessTokens_cg(processTokens_params params) {
//    auto grid = cooperative_groups::this_grid();
    assert(0);
  }
    
  void CudaLatticeDecoder::PostProcessTokens() {
    processTokens_params params;
    dim3 threads(64,1);
    dim3 blocks(DIV_ROUND_UP(total_threads,(threads.x*threads.y)));

    initParams(params);

    PostProcessTokens_cg<<<blocks,threads,0,stream_comp>>>(params);  //doesn't work
    cudaCheckError();
  }

  void CudaLatticeDecoder::initParams(processTokens_params& params) {
    params.prev_toks=(*prev_toks_);
    params.allocator=allocator;
    params.cutoff=cutoff_d;
    params.loglikelihoods=loglikelihoods_d;
    params.cur_toks=(*cur_toks_);
    params.e_offsets=fst_.e_offsets_d;
    params.ne_offsets=fst_.ne_offsets_d;
    params.arc_ilabels=fst_.arc_ilabels_d;
    params.arc_olabels=fst_.arc_olabels_d;
    params.arc_weights=fst_.arc_weights_d;
    params.arc_nextstates=fst_.arc_nextstates_d;
    params.current_tokens_lookup=current_tokens_lookup_d;
    params.token_locks=token_locks_d;
    params.modified=modified_d;
    params.beam=beam_;
    params.pe_idx=pe_idx_d;
    params.ne_idx=ne_idx_d;
    params.fb_idx=fb_idx_d;
    params.barrier=barrier_d;
    params.verbose=verbose;
    params.frame=num_frames_decoded_;
    params.prune_interval = prune_interval_;
    params.lat_arcs_sub_vec = *lat_arcs_sub_vec_;
    params.lattice_pruner=lattice_pruner_;
    uint idx=(num_frames_decoded_)%LAT_BUF_SIZE;
    params.lattice_beam=lattice_beam_;
  }
  void CudaLatticeDecoder::ProcessNonemitting() {
    nvtxRangePushA("ProcessNonemitting");

    dim3 threads(64,1);

    dim3 blocks(DIV_ROUND_UP(total_threads,(threads.x*threads.y)));

    processTokens_params params;
    initParams(params);
    processTokens_cg<<<blocks,threads,0,stream_comp>>>(params, true);

    cudaCheckError();
    nvtxRangePop();
  }
  void CudaLatticeDecoder::SetTokArcPointerByFrame(uint frame) {
    cur_toks_=&toks_buf_[frame%LAT_BUF_SIZE];
    prev_toks_=&toks_buf_[(frame-1)%LAT_BUF_SIZE];
    lat_arcs_sub_vec_=&lat_arcs_sub_vec_buf_;
  }

  __launch_bounds__(64,64)
  __global__ void LaunchPruneActiveTokens(processTokens_params params) {
//    auto grid = cooperative_groups::this_grid();
    params.lattice_pruner.PruneActiveTokens<VERBOSE>(params.frame, params.lattice_beam
      );
  }

  void CudaLatticeDecoder::CallLaunchPruneActiveTokens(cudaStream_t wait_st, 
    cudaStream_t st, float ratio) {
    processTokens_params params;
    initParams(params);
    dim3 threads(64,1);
    dim3 blocks(DIV_ROUND_UP(total_threads*ratio,(threads.x*threads.y)));
    cudaStreamSynchronize(wait_st);
    if (params.verbose>1) KALDI_LOG <<"CallLaunchPruneActiveTokens, # of blocks: "<<blocks.x<<std::endl;
    LaunchPruneActiveTokens<<<blocks,threads,0,st>>>(params);
    //lattice_pruner_.copy_arcs_to_host(num_frames_decoded_, st);
  }
  void CudaLatticeDecoder::PreProcessTokens() {
    nvtxRangePushA("PreProcessTokens"); 
    num_frames_decoded_++;
#ifndef MEMADVISE
    //no need to prefetch if we have done a memadvise
    allocator.prefetch_next_to_device(cudaStreamPerThread);
#endif
    //TODO prefetch here
    SetTokArcPointerByFrame(num_frames_decoded_);

    //(*cur_toks_).swap((*prev_toks_));
   
    {
      ClearToks(*cur_toks_);
      //uint32_t frame=num_frames_decoded_%prune_interval_;
      //dont need to clear as we directly take the final buffer into this vector
      //ClearArcVector(lat_arcs_sub_vec_);
    }
    nvtxRangePop();
  }
  
  void CudaLatticeDecoder::ProcessTokens() {
    PUSH_RANGE("ProcessTokens",2)
    if (verbose>4) KALDI_LOG << num_frames_decoded_<<std::endl;

    processTokens_params params;
    dim3 threads(64,1);
    dim3 blocks(DIV_ROUND_UP(total_threads,(threads.x*threads.y)));

    initParams(params);

    if (params.verbose>2&&params.frame==1) KALDI_LOG <<"# of blocks: "<<blocks.x<<std::endl;

     if (params.verbose>4) KALDI_LOG <<std::endl;
    cudaStreamWaitEvent(stream_comp,event_ll,0); //make sure log likelihoods are on the device before starting these kernels

     if (params.verbose>4)  KALDI_LOG <<std::endl;
#if 0
    void *args[] = { (void*) &params };
    cudaLaunchCooperativeKernel((void*)processTokens_cg, blocks, threads, args, 0, stream_comp);
#else
    processTokens_cg<<<blocks,threads,0,stream_comp>>>(params);  //doesn't work
#endif
    cudaCheckError();
      
    cudaEventSynchronize(event_pt); //throttle
    cudaEventRecord(event_pt,stream_comp);



    POP_RANGE
  }

  void CudaLatticeDecoder::ClearToks(TokenVector &toks) {
    //cannot acctually delete tokens as they may still be connected to active tokens
    toks.clear(stream_comp);
  }
} // end namespace kaldi.
