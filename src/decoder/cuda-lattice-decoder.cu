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

#include "lat/determinize-lattice-pruned.h"
#include "lat/kaldi-lattice.h"
#include "itf/decodable-itf.h"

#include "cuda-decoder-utils.h"
//#include "lattice-faster-decoder-cuda.h"
#include "decoder/cuda-lattice-decoder.h"


namespace kaldi {

#define CudaVector  CudaLatticeDecoder::CudaVector
#define CudaMergeVector CudaLatticeDecoder::CudaMergeVector
  typedef CudaLatticeDecoder::Token Token;
  typedef CudaLatticeDecoder::StateId StateId;
  typedef CudaLatticeDecoder::TokenState TokenState;
  typedef CudaLatticeDecoder::CostType CostType;
  typedef CudaLatticeDecoder::TokenLookupElem TokenLookupElem;
  typedef CudaLatticeDecoder::LatLink LatLink;
  typedef CudaLatticeDecoder::LatLinkVector LatLinkVector;
  //typedef CudaLatticeDecoder::TokenVector TokenVector;
  typedef CudaLatticeDecoder::TokenMergeVector TokenMergeVector;
  typedef CudaLatticeDecoder::processTokens_params processTokens_params;
  typedef CudaLatticeDecoder::LatticePruner LatticePruner;
  //template class CudaVector<LatToken>; 
  //template class CudaVector<LatLink>; 
  //http://en.cppreference.com/w/cpp/language/class_template
  template HOST DEVICE LatLink& CudaVector<LatLink>::operator[](uint32_t idx); 
  template HOST DEVICE TokenState& CudaVector<TokenState>::operator[](uint32_t idx); 
  template HOST DEVICE uint32_t  CudaVector<TokenState>::size() const; 
  template HOST DEVICE uint32_t  CudaVector<LatLink>::size() const; 

// for speedup purpose, make them inline (5% 0.165->0.158)
inline DEVICE uint64_t pack (float cost, int ptr) {
  //assert (!isnan(cost));
  //assert (ptr >= 0 && ptr < 1L<<32);
  uint32_t i_cost = *(uint32_t *)&cost;
  if (i_cost & 0x80000000)
    i_cost = i_cost ^ 0xFFFFFFFF;
  else
    i_cost = i_cost ^ 0x80000000;
  return (uint64_t)i_cost << 32 | ptr;
}

// Unpacks a probability.
inline DEVICE float unpack_cost (uint64_t packed) {
  uint32_t i_cost = packed >> 32;
  if (i_cost & 0x80000000)
    i_cost = i_cost ^ 0x80000000;
  else
    i_cost = i_cost ^ 0xFFFFFFFF;
  return *(float *)&i_cost;
}

// Unpacks a back-pointer.
inline DEVICE int unpack_ptr (uint64_t packed) {
  //assert (!(packed & 0x80000000));
  return packed & 0x7FFFFFFF;
}


inline  DEVICE void load16(void *a, const void *b) {
    const ulong2 *src = reinterpret_cast<const ulong2*>(b);
    ulong2 &dst = *reinterpret_cast<ulong2*>(a);
    asm("ld.global.v2.u64 {%0,%1}, [%2];" : "=l"(dst.x), "=l"(dst.y) : "l"(src));
  }
  
inline  DEVICE void store16(void *a, const void *b) {
    const ulong2 src = *reinterpret_cast<const ulong2*>(b);
    asm("st.global.v2.u64 [%0], {%1,%2};" :: "l"(a), "l"(src.x), "l"(src.y));
  }

  
inline  DEVICE void store32(void *a, const void *b) {
    //const ulong4 src = *reinterpret_cast<const ulong4*>(b);
    //asm("st.global.v4.u64 [%0], {%1,%2,%3,%4};" :: "l"(a), "l"(src.x), "l"(src.y),
    //  "l"(src.z), "l"(src.w));
    memcpy(a, b, 32);
  }

inline DEVICE void atomicMin(double *address, double val) {
  unsigned long long *address_ull = (unsigned long long *)address;

  double minval = *address;

  while (val < minval) {  //if my value is less than minimum
    minval = val;         //update the minimum to my value locally
    val = __longlong_as_double(atomicExch(address_ull, __double_as_longlong(val))); //write minimum and read back value
  } //if the new value is < the minimum I wrote I need to try again.
}
inline DEVICE void atomicMin(float *address, float val) {
  unsigned int *address_ui = (unsigned int  *)address;

  float minval = *address;

  while (val < minval) {  //if my value is less than minimum
    minval = val;         //update the minimum to my value locally
    val = __uint_as_float(atomicExch(address_ui, __float_as_uint(val))); //write minimum and read back value
  } //if the new value is < the minimum I wrote I need to try again.
}

// end of "for speedup purpose, make them inline (5% 0.165->0.158)"


//private, as we need to instantiate them  
template<typename T> 
  inline DEVICE void swap(T &a, T &b) {
    T c = a;
    a = b;
    b = c;
  }

/******************************************CudaVector Implementation*******************************/
template<typename T>
  HOST DEVICE inline T& CudaVector<T>::operator[](uint32_t idx) { 
#ifdef __CUDA_ARCH__
    assert(idx<*count_d);
    return mem_d[idx];
#else
    assert(idx<*count_h);
    return mem_h[idx];
#endif
  }

template<typename T>
  HOST DEVICE inline const T& CudaVector<T>::operator[](uint32_t idx) const { 
#ifdef __CUDA_ARCH__
    assert(idx<*count_d);
    return mem_d[idx];
#else
    assert(idx<*count_h);
    return mem_h[idx];
#endif
  } 

template<typename T>
  inline void CudaVector<T>::Allocate(uint32_t max_size, 
     uint32_t* icount_h, uint32_t* icount_d, T* mem_d, T* mem_h) {
    this->max_size=max_size;
    alloc_size=0;

    if (icount_h) this->count_h=icount_h;
    else {
      cudaMallocHost(&this->count_h,sizeof(uint32_t));
    }
      if (icount_d) this->count_d=icount_d;
      else {
        alloc_size+=sizeof(uint32_t);
        cudaMalloc(&this->count_d, sizeof(uint32_t));
      }
      cudaMemset(this->count_d, 0,sizeof(uint32_t));
      *count_h=0;

      if (mem_d) {
        this->mem_d=mem_d;        
      } else {
        alloc_size+=max_size*sizeof(T);
        cudaMalloc(&this->mem_d,max_size*sizeof(T));
      }
      if (mem_h) {
        this->mem_h=mem_h;        
      } else {
        cudaMallocHost(&this->mem_h,max_size*sizeof(T));
      }
    }

  template<typename T>
    inline size_t CudaVector<T>::getCudaMallocBytes() {
      return alloc_size;
    }

  template<typename T>
    inline void CudaVector<T>::Free(bool create_outside) { 
      cudaFreeHost(mem_h);
      if (!create_outside) {
        cudaFree(mem_d); 
      }
      cudaFreeHost(count_h);
      cudaFree(count_d);       
    }


  template<typename T>
    inline void CudaVector<T>::copy_all_to_host(cudaStream_t stream) {
      cudaStreamSynchronize(stream);
      cudaMemcpy(count_h,count_d,sizeof(int32),cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(mem_h,mem_d,*count_h*sizeof(T),cudaMemcpyDeviceToHost, stream);
    }

  template<typename T>
    inline void CudaVector<T>::copy_all_to_device(cudaStream_t stream) {
      cudaStreamSynchronize(stream);
      cudaMemcpyAsync(count_d,count_h,sizeof(int32),cudaMemcpyHostToDevice);
      cudaMemcpyAsync(mem_d,mem_h,*count_h*sizeof(T),cudaMemcpyHostToDevice, stream);
    }

  template<typename T>
    inline void CudaVector<T>::copy_size_to_host(cudaStream_t stream) {
      cudaMemcpyAsync(count_h,count_d,sizeof(int32),cudaMemcpyDeviceToHost, stream);
    }

  template<typename T>
    inline void CudaVector<T>::copy_size_to_device(cudaStream_t stream) {
      cudaMemcpyAsync(count_d,count_h,sizeof(int32),cudaMemcpyHostToDevice, stream);
    }
  
template<typename T>
    inline void CudaVector<T>::copy_data_to_host(cudaStream_t stream, T* to_buf, bool copy_size) {
      if (!to_buf) {
        to_buf=mem_h;
      }
      if (copy_size) cudaMemcpy(count_h,count_d,sizeof(int32),cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(to_buf,mem_d,*count_h*sizeof(T),cudaMemcpyDeviceToHost, stream);
    }

  template<typename T>
    inline void CudaVector<T>::copy_data_to_device(cudaStream_t stream) {
      cudaMemcpyAsync(mem_d,mem_h,*count_h*sizeof(T),cudaMemcpyHostToDevice, stream);
    }

  template<typename T>
    inline void CudaVector<T>::copy_data_to_device(int size, T* mem_in_d, cudaStream_t stream) {
      cudaMemcpyAsync(mem_d+*count_d*sizeof(T),mem_in_d,size*sizeof(T),cudaMemcpyDeviceToDevice, stream);
      *count_d+=size;
    }



  //Note:  This will cause page faults back and forth when we switch from host to device.
  template<typename T>
    HOST DEVICE inline uint32_t CudaVector<T>::size() const 
    {
#ifdef __CUDA_ARCH__
      return *count_d; 
#else
      return *count_h;
#endif
    }

  template<typename T> 
    HOST DEVICE inline uint32_t CudaVector<T>::push_back(const T &val) { 
#ifdef __CUDA_ARCH__
      assert(*count_d<max_size);
      uint32_t idx = atomicAdd(count_d,1);
      mem_d[idx]=val; 
#else
      assert(*count_h<max_size);
      uint32_t idx = (*count_h)++;
      mem_h[idx]=val; 
#endif
      return idx;
    }
  template<typename T> 
    HOST DEVICE inline void CudaVector<T>::clear(cudaStream_t stream) { 
#ifdef __CUDA_ARCH__
      *count_d = 0;
#else
      *count_h = 0; 
      cudaMemsetAsync(count_d,0,sizeof(int32),stream); 
#endif
    }
  template<typename T> 
    HOST DEVICE inline int CudaVector<T>::get_idx_from_addr(T* addr) { 
#ifdef __CUDA_ARCH__
      int ret=addr-mem_d;
      assert(ret<*count_d&&ret>=0);
      return ret;
#else
      int ret=addr-mem_h;
      assert(ret<*count_h&&ret>=0);
      return ret;
#endif
    }
  template<typename T> 
    inline bool CudaVector<T>::empty() const { return size()==0; }
  template<typename T> 
    inline void CudaVector<T>::swap(CudaVector<T> &v) {
      std::swap(mem_h,v.mem_h);
      std::swap(mem_d,v.mem_d);
      std::swap(count_h,v.count_h);
      std::swap(count_d,v.count_d);
      std::swap(max_size,v.max_size);
    }
  /**************************************End CudaVector Implementation**********************************/

//end of "private, as we need to instantiate them  "

//private


  template<typename T> 
    inline void CudaMergeVector<T>::swap(CudaMergeVector<T> &v) {
      CudaVector<T>::swap(v);
      std::swap(mem_update_d,v.mem_update_d);
    }


template<typename T> 
  DEVICE inline int CudaMergeVector<T>::update(int i) {
    if (i>=*count_d) return 0;
    return mem_update_d[i];
  }
template<> 
DEVICE inline void CudaMergeVector<TokenState>::merge(void* token_per_arc, int* token_per_arc_update, int num_arcs, bool clear) {
  int tid=threadIdx.x+blockIdx.x*blockDim.x;
  int idx=tid;
  int rank0=blockIdx.x==0&&threadIdx.x==0?1:0;
  int batch=blockDim.x*gridDim.x; 
  if (rank0) {
    int acc=0;
    acc+=(count_d[0]);
    if (clear) count_d[0]=0;
    assert(acc<=max_size);
    mem_buf_acc_count_d[0]=0;
    mem_buf_acc_count_d[1]=acc;
  }
  __grid_sync_nv_internal(barrier_);
  int sz = mem_buf_acc_count_d[1]-mem_buf_acc_count_d[0];
  for(; idx < sz; idx += batch) {
    uint64_t* pack_v=mem_pack_buf_d[idx];
    int ptr=unpack_ptr(*pack_v);
    //assert(ptr<num_arcs);
    mem_update_d[(idx+mem_buf_acc_count_d[0])]=token_per_arc_update[ptr];
  #if 1
    if (token_per_arc_update[ptr]) token_per_arc_update[ptr]=0;
    else continue;
  #endif
    TokenState* to_ts=mem_d+(idx+mem_buf_acc_count_d[0]);
    Token* cur_tok=((Token *)token_per_arc)+ptr;
    Token* to_tok=to_ts->token;
    store16(to_tok, cur_tok);
    //memcpy(to_tok,cur_tok,sizeof(T));
  }    
}

template<typename T> 
DEVICE inline void CudaMergeVector<T>::clear_sub() {
  int rank0=blockIdx.x==0&&threadIdx.x==0?1:0;
  if (rank0) {
  }
}


template<typename T> 
DEVICE inline void CudaMergeVector<T>::merge(void* undefined, int* token_per_arc_update, int num_arcs, bool clear) {
  assert(0);
}

  template<typename T> 
    DEVICE inline uint32_t CudaMergeVector<T>::push_back(const T &val, 
                                    uint64 *val_pack) { 
      uint32_t idx = atomicAdd(count_d,1);
      assert(*count_d<max_size);
      assert(sizeof(val)==16);
      store16(&mem_d[idx],&val); 
      mem_pack_buf_d[idx]=val_pack;  
      return idx;
    }

  template<typename T>
    inline void CudaMergeVector<T>::Allocate(uint32_t max_size) {
      CudaVector<T>::Allocate(max_size);

      cudaMalloc(&mem_pack_buf_d,sizeof(uint64_t*)*max_size);
      cudaMalloc(&mem_update_d,sizeof(int)*max_size);
      cudaMemset(mem_update_d,0,sizeof(int)*max_size);
      cudaMalloc(&mem_buf_acc_count_d,sizeof(int)*(2));
      cudaMalloc(&barrier_,sizeof(int)*1);
    }

  template<typename T>
    inline size_t CudaMergeVector<T>::getCudaMallocBytes() {
      return CudaVector<T>::getCudaMallocBytes()+
        sizeof(uint32_t)*(1+2*(2))+max_size*(sizeof(T)+sizeof(uint64_t*)+sizeof(int));
    }

  template<typename T>
    inline void CudaMergeVector<T>::Free() { 
      CudaVector<T>::Free();
      cudaFree(mem_pack_buf_d);
      cudaFree(mem_update_d);
      cudaFree(mem_buf_acc_count_d);
      cudaFree(barrier_);
    }
    DEVICE int LatticePruner::AddArc(LatLink* arc) {
      int i=atomicAdd(arcs_apr_used_d, 1);
      store32(arcs_apr_d+i, arc);
    }
    #define PRUNE_RATIO_ASSUME 0.25
    int LatticePruner::Allocate(int max_tokens_per_frame, int max_lat_arc_per_frame, 
      int prune_interval, int max_toks, int max_arcs) {
      int sz;
      int bytes_cudaMalloc=0;
      
      //after
      sz=sizeof(int)*(prune_interval+1);
      cudaMalloc((void**)&arcs_apr_fr_size_d,sz); bytes_cudaMalloc+=sz;
      cudaMallocHost((void**)&arcs_apr_fr_size_h,sz);
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
      cudaMalloc((void**)&toks_bpr_fr_sidx_d,sz); bytes_cudaMalloc+=sz;
      cudaMallocHost((void**)&toks_bpr_fr_sidx_h,sz); 
      sz=sizeof(int)*(prune_interval+1);
      cudaMalloc((void**)&arcs_bpr_fr_sidx_d,sz); bytes_cudaMalloc+=sz;

      sz=sizeof(int);
      cudaMalloc((void**)&barrier_,sz); bytes_cudaMalloc+=sz;
      cudaMalloc((void**)&modified_d,sz); bytes_cudaMalloc+=sz;
      sz=sizeof(int)*(2);
      cudaMalloc((void**)&count_vec_acc_d,sz); bytes_cudaMalloc+=sz;
      this->prune_interval=prune_interval;
      
      return bytes_cudaMalloc;
    }
    void LatticePruner::Free() {
      cudaFree(arcs_apr_fr_size_d);
      cudaFreeHost(arcs_apr_fr_size_h);
      /*cudaFree(arcs_buf_after_pr_size_arr_used_d);*/
      cudaFree(arcs_apr_d);
      cudaFree(arcs_apr_used_d);
      cudaFree(arcs_bpr_used_d);
      cudaFreeHost(arcs_apr_used_h);
      cudaFree(toks_bpr_d);
      cudaFreeHost(toks_bpr_h);
      cudaFree(arcs_bpr_d);
      cudaFree(toks_bpr_fr_sidx_d);
      cudaFreeHost(toks_bpr_fr_sidx_h);
      cudaFree(arcs_bpr_fr_sidx_d);  
      
      cudaFree(count_vec_acc_d);
      cudaFree(barrier_);
      cudaFree(modified_d);
      cudaFreeHost(arcs_apr_h);
    }
    inline DEVICE void LatticePruner::SetNextSidx(int* sidx_buf, int size, int frame) {
      assert(frame>=0);
      int cur_sidx=sidx_buf[(frame)];
      sidx_buf[(frame+1)]=cur_sidx+size;
    }
    inline DEVICE void LatticePruner::CollectToksPerFrame(TokenState* cur_toks, int size, int frame) {
      int tid=threadIdx.x+blockIdx.x*blockDim.x;
      if (tid==0) {
        SetNextSidx(toks_bpr_fr_sidx_d, size, frame);
      }
      for (;tid<size;tid+=gridDim.x*blockDim.x) {
        Token* to_tok=GetActiveToks(frame,tid,false);
        store16(to_tok, cur_toks[tid].token);
        //debug
        //assert(cur_toks[tid].token->frame==frame);
        //GetActiveToks(frame,tid,true);
      }
    }
    inline DEVICE void LatticePruner::CollectArcsPerFrame(LatLinkVector& cur_arc_array, 
      uint* count_vec_d, int frame) {
      int tid=threadIdx.x+blockIdx.x*blockDim.x;
      int idx=tid;
      int rank0=blockIdx.x==0&&threadIdx.x==0?1:0;
      int batch=blockDim.x*gridDim.x;
      
      int sz = *count_vec_d-*arcs_bpr_used_d;
      __grid_sync_nv_internal(barrier_);
      if (rank0) {
        int size=sz;
        SetNextSidx(arcs_bpr_fr_sidx_d, size, frame);
        *arcs_bpr_used_d=*count_vec_d;
      }
      //in this version, we share the mem between vector&pruner, so dont need to copy
/*      
      for(; idx < sz; idx += batch) {
        LatLink* to_arc=GetActiveArcs(frame,(idx));
        store32(to_arc, cur_arc_array.mem_d+idx);
        //debug
        GetActiveToks((cur_arc_array.mem_d+idx)->p1,true,frame);
        GetActiveToks(to_arc->p1,true,frame);
      }
      */
    }
    inline DEVICE void LatticePruner::PruneActiveTokens(int frame, float lattice_beam, int verbose) {
      int rank0=threadIdx.x==0&&blockIdx.x==0?1:0;
      if (frame==0) return;
      if (rank0) *arcs_apr_used_d=0;
      __grid_sync_nv_internal(barrier_);
      for (int f = frame; f > 0; f--) {
          PruneLatticeForFrame(f,1,lattice_beam, verbose);
      }
      //see copy_data_to_host
      assert(*arcs_apr_used_d<arcs_buf_before_pr_size*PRUNE_RATIO_ASSUME);
      if (verbose>2&&rank0) GPU_PRINTF("PRt: %i %i\n", arcs_bpr_fr_sidx_d[frame+1], *arcs_apr_used_d);
    }
    inline DEVICE Token* LatticePruner::GetActiveToks(void* p, bool check, int iframe) const {
      int frame, id;
      DecodeFrIdxFromPair(frame, id, (uint64)p);
      if (check) assert(frame==iframe||frame==iframe-1);
      return GetActiveToks(frame,id,check);
    }
    inline DEVICE Token* LatticePruner::GetActiveToks(int frame, int id, bool check) const {
      
      int cur_sidx=toks_bpr_fr_sidx_d[frame];
      assert(cur_sidx+id<toks_buf_before_pr_size);
      Token* tok=toks_bpr_d+cur_sidx+id;
      if (check) {
        assert(tok->frame==frame);
      }
      return tok;
    }
    inline DEVICE LatLink* LatticePruner::GetActiveArcs(int frame, int id) const {
      int cur_sidx=arcs_bpr_fr_sidx_d[(frame)];
      assert(cur_sidx+id<arcs_buf_before_pr_size);
      LatLink* arc=arcs_bpr_d+cur_sidx+id;
      return arc;
    }
    inline DEVICE int LatticePruner::GetSize(int* acc_len, int frame) const {
      int size=acc_len[(frame)+1]-acc_len[(frame)];
      assert(size>=0&&size<=arcs_buf_before_pr_size);
      return size;
    }
    inline DEVICE void LatticePruner::PruneLatticeForFrame(int frame, 
                                  bool merge, float lattice_beam, int verbose) {
      //init
      int prev_cidx;
      int c=0;
      int rank0=threadIdx.x==0&&blockIdx.x==0?1:0;
      
        __grid_sync_nv_internal(barrier_);
      if (rank0&&verbose>3) GPU_PRINTF("%i %i\n",c++, GetSize(toks_bpr_fr_sidx_d,frame-1));
      {
        int tid=threadIdx.x+blockIdx.x*blockDim.x;
        int size=GetSize(toks_bpr_fr_sidx_d,frame-1);
        for (;tid<size;tid+=gridDim.x*blockDim.x) {
          Token* tok=GetActiveToks(frame-1,tid,true);
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
        int size=GetSize(arcs_bpr_fr_sidx_d,frame);
        if (rank0) *modified_d=0;
        __grid_sync_nv_internal(barrier_);
        for (;tid<size;tid+=gridDim.x*blockDim.x) {
          LatLink* link=GetActiveArcs(frame,tid);
          Token* next_tok=GetActiveToks(link->p1,true,frame);
          Token* tok=GetActiveToks(link->p2, true,frame);
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
        int size=GetSize(arcs_bpr_fr_sidx_d,frame);
        for (;tid<size;tid+=gridDim.x*blockDim.x) {
          LatLink* link=GetActiveArcs(frame,tid);
          Token* next_tok=GetActiveToks(link->p1,true,frame);
          Token* tok=GetActiveToks(link->p2, true,frame);
          float link_extra_cost = next_tok->extra_cost + 
                  ((tok->cost_ + link->acoustic_cost + link->graph_cost) 
                    - next_tok->cost_);
          if (isnan(link_extra_cost) || link_extra_cost > lattice_beam)
            ; //should be pruned
          else {
            if (merge) {
              AddArc(link);
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
        int size=GetSize(toks_bpr_fr_sidx_d,frame);
        for (;tid<size;tid+=gridDim.x*blockDim.x) {
          Token* tok=GetActiveToks(frame-1,tid);
          if (tok->extra_cost==FLT_MAX)
            tok->tot_cost=CUDART_NAN_F; //prune
        }        
      } */   
      
      //get size 
      if (merge&&rank0) {
          int& size_arc_of_frame=arcs_apr_fr_size_d[frame];
          size_arc_of_frame=*arcs_apr_used_d-prev_cidx;
          if (verbose>3) GPU_PRINTF("PR %i %i %i\n",frame, 
            GetSize(arcs_bpr_fr_sidx_d,frame), size_arc_of_frame);
          //size_tok_of_frame[f-1]=cidx-prev_cidx
          //prev_cidx=cidx
      }
      __grid_sync_nv_internal(barrier_);
        if (rank0&&verbose>3) GPU_PRINTF("%i %p\n",c++,barrier_);
    }
    //#define GET_ARC_BUF_HOST_BY_FRAME(frame) (arcs_apr_h+arcs_apr_h_used)

    void LatticePruner::Initialize() {
      //arcs_apr_h_used=0;
      cudaMemset(arcs_apr_fr_size_d,0,sizeof(int)*(prune_interval+1));
      cudaMemset(arcs_apr_used_d,0,sizeof(int));
      cudaMemset(arcs_bpr_used_d,0,sizeof(int));      
      cudaMemset(toks_bpr_fr_sidx_d,0,sizeof(int)*(prune_interval+1));
      cudaMemset(arcs_bpr_fr_sidx_d,0,sizeof(int)*(prune_interval+1));
    }
    inline DEVICE void LatticePruner::DecodeFrIdxFromPair(int& frame, int& idx, uint64 v) const {
      frame=(((uint64)v)>>32);
      idx=(((uint64)v)&(((uint64)1<<32)-1));
    }
    void LatticePruner::CopyArcsToHost(int frame, cudaStream_t st) {
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
      cudaMemcpyAsync(arcs_apr_fr_size_h,arcs_apr_fr_size_d,
        sz, cudaMemcpyDeviceToHost, st);
      // clear arcs_apr_used_d in GPU
    }
    void LatticePruner::CopyToksToHost(int frame, cudaStream_t st) {
      int sz;
      sz=sizeof(int)*(frame+1+1)*(1); //include frame 0 & finalsum
      cudaMemcpy(toks_bpr_fr_sidx_h,toks_bpr_fr_sidx_d,
        sz, cudaMemcpyDeviceToHost);
      sz=sizeof(Token)*(toks_bpr_fr_sidx_h[frame+1]);
      assert(sz);
      cudaMemcpyAsync(toks_bpr_h,toks_bpr_d,
        sz, cudaMemcpyDeviceToHost,st);
    }

    void LatticePruner::GetHostData (Token** toks_buf, 
      int** toks_fr_sidx, LatLink** arcs_buf, int** arcs_fr_size) {
      *toks_fr_sidx=toks_bpr_fr_sidx_h;
      *toks_buf=toks_bpr_h;
      *arcs_fr_size=arcs_apr_fr_size_h;  //next prune_interval len
      *arcs_buf=arcs_apr_h; //start of next prune_interval len arcs
    }

  DEVICE inline void allocateAllTokens_function(TokenLookupElem *current_tokens_lookup, int32 numStates,  CudaLatticeDecoder::TokenAllocator allocator) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<numStates; i+=blockDim.x*gridDim.x) {
      Token *token = allocator.getToken(i);
      token->cost_ = INFINITY;
      token->extra_cost = 0;
      token->frame= -1;
      //token->state_id= -1;
      TokenLookupElem elem;
      elem.token=token;
      elem.active=false;
      elem.tokenstate_idx=-1;
     elem.token_pack=pack(-FLT_MAX, 0);
      memcpy(&current_tokens_lookup[i], &elem, sizeof(TokenLookupElem));
    }
  }
  __global__ void allocateAllTokens(TokenLookupElem *current_tokens_lookup, int32 numStates,  CudaLatticeDecoder::TokenAllocator allocator, int *barrier) {
    allocateAllTokens_function(current_tokens_lookup,numStates,allocator);
     __grid_sync_nv_internal(barrier);
     if(blockIdx.x==0 && threadIdx.x==0) {
      allocator.advanceFront(numStates);
     }
  }

  DEVICE inline void allocateNewTokens_function(TokenLookupElem *current_tokens_lookup, TokenMergeVector cur_toks, CudaLatticeDecoder::TokenAllocator allocator) {
    int32 size = cur_toks.size();
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=blockDim.x*gridDim.x) {
      Token *token = allocator.getToken(i);
      token->cost_ = INFINITY;
      token->extra_cost = 0;
      token->frame= -1;
      //token->state_id= -1;
      //a CPU copy of cur_toks can still be used, cur_toks will be clear in PreProcessTokens 
      StateId state=cur_toks[i].state;  
      //cur_toks[i].token->arc_index_=-1; // clear here will result in page fault in prefetch
      TokenLookupElem elem;
      elem.token=token;
      elem.active=false;
      elem.tokenstate_idx=-1;
      elem.token_pack=pack(-FLT_MAX, 0);
      memcpy(&current_tokens_lookup[state], &elem, sizeof(TokenLookupElem));
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


  CudaLatticeDecoder::CudaLatticeDecoder(const CudaFst &fst, const CudaLatticeDecoderConfig &config): config_(config), fst_(fst), bytes_cudaMalloc(0), bytes_cudaMallocManaged(0) {
    printf("CudaLatticeDecoder Constructor\n");
    int device;
    cudaGetDevice(&device);
    cudaCheckError();

    if (config_.verbose>4) cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1e7);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,device);

    total_threads = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount * config.gpu_fraction;

    token_allocator_.initialize(config.max_tokens);
    cudaCheckError();
    bytes_cudaMallocManaged+=token_allocator_.getCudaMallocManagedBytes();

    cudaEventCreateWithFlags(&event_pt,cudaEventDisableTiming);
    cudaEventCreateWithFlags(&event_ll,cudaEventDisableTiming);

    cudaStreamCreateWithFlags(&stream_comp, cudaStreamNonBlocking);
    for (int i=0; i<LAT_BUF_SIZE; i++) 
      cudaStreamCreateWithFlags(&stream_lat[i], cudaStreamNonBlocking);    
    cudaStreamCreateWithPriority(&stream_ll, cudaStreamNonBlocking, -1);

    cudaMalloc(&pe_idx_d, sizeof(int)); bytes_cudaMalloc+=sizeof(int);
    cudaMalloc(&agg_idx_d, sizeof(int)); bytes_cudaMalloc+=sizeof(int);
    cudaMalloc(&ne_idx_d, sizeof(int)); bytes_cudaMalloc+=sizeof(int);
    cudaMalloc(&ne_queue_d, sizeof(int)*config.max_tokens_per_frame); bytes_cudaMalloc+=sizeof(int);
    cudaMalloc(&fb_idx_d, sizeof(int)); bytes_cudaMalloc+=sizeof(int);
    cudaMalloc(&barrier_d, sizeof(int)); bytes_cudaMalloc+=sizeof(int);

    cudaMemset(pe_idx_d,0,sizeof(int));
    cudaMemset(ne_idx_d,0,sizeof(int));
    cudaMemset(agg_idx_d,0,sizeof(int));
    cudaMemset(fb_idx_d,0,sizeof(int));
    cudaMemset(barrier_d,0,sizeof(int));
    cudaCheckError();

    cudaMalloc(&cutoff_d, sizeof(CostType)); bytes_cudaMalloc+=sizeof(CostType);
    cudaMalloc(&modified_d, sizeof(int)*1); bytes_cudaMalloc+=sizeof(int)*1;


    cudaMalloc((void**)&current_tokens_lookup_d,sizeof(TokenLookupElem)*fst_.numStates); bytes_cudaMalloc+=sizeof(TokenLookupElem)*fst_.numStates;

    cudaMallocHost(&loglikelihoods_h,sizeof(BaseFloat)*(fst_.max_ilabel+1));  
    cudaMallocHost(&loglikelihoods_old_h,sizeof(BaseFloat)*(fst_.max_ilabel+1));

    cudaMalloc((void**)&loglikelihoods_d,sizeof(BaseFloat)*(fst_.max_ilabel+1)); bytes_cudaMalloc+=sizeof(BaseFloat)*(fst_.max_ilabel+1);
    cudaMalloc((void**)&loglikelihoods_old_d,sizeof(BaseFloat)*(fst_.max_ilabel+1)); bytes_cudaMalloc+=sizeof(BaseFloat)*(fst_.max_ilabel+1);

    //for pruning
    bytes_cudaMalloc+=lattice_pruner_.Allocate(config.max_tokens_per_frame, 
                      config.max_lat_arc_per_frame, config.prune_interval,
                      config.max_tokens, config.max_arcs);

    lat_arcs_buf_.Allocate(
      config.max_arcs,
      NULL,
      NULL, 
      lattice_pruner_.GetDeviceArcsBpr(),
      NULL); 
    bytes_cudaMalloc += lat_arcs_buf_.getCudaMallocBytes();  

    for (int j=0; j<LAT_BUF_SIZE; j++) {
      lat_toks_bufs_[j].Allocate(config.max_tokens_per_frame);
      bytes_cudaMalloc+=lat_toks_bufs_[j].getCudaMallocBytes();
    }

    cudaMalloc((void**)&token_per_arc_d,sizeof(Token)*fst.NumArcs()); //temp solution
    cudaMalloc((void**)&token_per_arc_update_d,sizeof(int)*fst.NumArcs()); //temp solution
    cudaMemset(token_per_arc_update_d,0,sizeof(int)*fst.NumArcs()); //temp solution
    bytes_cudaMalloc+=(sizeof(Token)+sizeof(int))*(fst.NumArcs());

    num_frames_decoded_=0;
    UpdateTokPointersByFrame(num_frames_decoded_);
    
    cudaStreamSynchronize(stream_comp);
    cudaStreamSynchronize(stream_lat[0]);
    cudaStreamSynchronize(cudaStreamPerThread);
    //sgemm requires shared memory and we don't want cache config changing.  So set a device wide cache config.
    cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
  }

  CudaLatticeDecoder::~CudaLatticeDecoder() {

    printf("CUDA LatticeDecoder DESTRUCTOR\n");

    for (int j=0; j<LAT_BUF_SIZE; j++) {
      lat_toks_bufs_[j].Free();
    }
    lat_arcs_buf_.Free(true);
    
    lattice_pruner_.Free();    
    
    token_allocator_.finalize();

    cudaFreeHost(loglikelihoods_h);
    cudaFreeHost(loglikelihoods_old_h);
    cudaFree(loglikelihoods_d);
    cudaFree(loglikelihoods_old_d);
    cudaFree(current_tokens_lookup_d);

    cudaFree(pe_idx_d);
    cudaFree(agg_idx_d);
    cudaFree(ne_idx_d);
    cudaFree(ne_queue_d);
    cudaFree(fb_idx_d);
    cudaFree(barrier_d);

    cudaFree(cutoff_d);
    cudaFree(modified_d);

    cudaFree(token_per_arc_d);
    cudaFree(token_per_arc_update_d);


    cudaEventDestroy(event_pt);
    cudaEventDestroy(event_ll);

    cudaStreamDestroy(stream_comp);
    for (int i=0; i<LAT_BUF_SIZE; i++) 
      cudaStreamDestroy(stream_lat[i]);
    cudaStreamDestroy(stream_ll);

  }
  DEVICE inline Token* FindOrAddTokenArc(processTokens_params& params,
    StateId nextstate, CostType total_cost, CostType acoustic_cost,
    TokenState* ts, uint32_t j, bool add_arc, TokenState** next_ts,
   uint64_t **token_pack, int* update) {
    //TokenLookupElem lookup_elem;
    //load16(&lookup_elem, &params.current_tokens_lookup[nextstate]);
    TokenLookupElem& lookup_elem = params.current_tokens_lookup[nextstate];
    Token *cur_tok = lookup_elem.token;  
    //check if token is active or not.  Double check the lock.
    if(lookup_elem.active==0 && atomicCAS(&lookup_elem.active,0,1)==0) {        //grab sentinal to see who gets to add to cur_toks list
      //if havent seen, add into hash
     *update=1;
      lookup_elem.tokenstate_idx=params.cur_toks.push_back(TokenState(cur_tok,nextstate,total_cost),
        &lookup_elem.token_pack);
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
    *token_pack=&lookup_elem.token_pack;
    return cur_tok;  
  }
  __global__ void addOneToken(processTokens_params params, StateId state) {
    TokenState *next_ts=NULL;
    uint64_t* token_pack;
    int j=0;
    if (threadIdx.x!=0 || blockIdx.x!=0) return;
    Token* cur_tok=FindOrAddTokenArc(params, state, 0, //add first token
      0, NULL, j, false,  &next_ts,
      &token_pack, params.token_per_arc_update+j);
        uint64_t new_token_pack=pack(0, j);
    Token* cur_te=params.token_per_arc+j;
    params.token_per_arc_update[j]=1;
    store16(cur_te, &(Token(0, params.frame, NULL)));
    atomicMax((unsigned long long *)token_pack, (unsigned long long)new_token_pack);
    params.cur_toks.merge(params.token_per_arc,params.token_per_arc_update, params.numArcs, false);
  }

namespace CudaLatticeDecoder_kernel {
  //putting this into a kernel to avoid extra latency of a memory copy
  __global__ void initializeCutoff(CostType *cutoff) {
    *cutoff = INFINITY;
  }
}
 
  void CudaLatticeDecoder::InitDecoding() {
    printf("CUDA LatticeDecoder InitDecoding\n");
    num_frames_decoded_ = 0;
    for (int i=0; i<LAT_BUF_SIZE; i++) {
      ClearToks(lat_toks_bufs_[i]);
    }    
    lat_arcs_buf_.clear();
    
    UpdateTokPointersByFrame(num_frames_decoded_);
    lattice_pruner_.Initialize();

    token_allocator_.reset();
    int threads=64;
    int blocks=DIV_ROUND_UP(total_threads,threads);
    
    //start moving these / allocating them on the device
    token_allocator_.prefetch_next_to_device(stream_comp, fst_.numStates+5000);

    allocateAllTokens<<<blocks,threads,0,stream_comp>>>(current_tokens_lookup_d, fst_.numStates, token_allocator_, barrier_d);

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

    CudaLatticeDecoder_kernel::initializeCutoff<<<1,1,0,stream_comp>>>(cutoff_d);
    ProcessNonemitting();
    printf("end of CUDA LatticeDecoder InitDecoding\n");
  }

  void CudaLatticeDecoder::FinalProcessLattice(
    Token** toks_buf, int** toks_fr_sidx, LatLink** arcs_buf, int** arcs_fr_size,
    TokenMergeVector** toks_vec_last_fr) {
    cudaStreamSynchronize(stream_comp); // after fini comp. we can start copy 
    // copy unpruned toks to host
    lattice_pruner_.CopyToksToHost(num_frames_decoded_, stream_lat[0]);
    PruneActiveTokens(stream_comp, stream_comp, 1); // GPU lattice pruning
    // copy the TokenState vector in the last frame, used by ComputeFinalCosts()
    (lat_toks_bufs_[num_frames_decoded_%LAT_BUF_SIZE]).copy_data_to_host(stream_lat[1]);
    *toks_vec_last_fr=&(lat_toks_bufs_[num_frames_decoded_%LAT_BUF_SIZE]);
    cudaStreamSynchronize(stream_comp); // wait for lattice pruning
    // copy pruned lattice arcs to host
    lattice_pruner_.CopyArcsToHost(num_frames_decoded_, stream_lat[1]);
    // wait for all streams finishing
    cudaStreamSynchronize(stream_lat[0]);
    cudaStreamSynchronize(stream_lat[1]);
    // get host data from lattice_pruner_, used by CPU lattice processing
    lattice_pruner_.GetHostData(toks_buf, toks_fr_sidx, 
                                            arcs_buf, arcs_fr_size);
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
    KALDI_ERR << "we don't have this implementation in lattice decoder";
    return false;
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
    auto group = cooperative_groups::tiled_partition<blockDimx>(cooperative_groups::this_thread_block());

    CostType cutoff=*params.cutoff;
    int32 size = params.prev_toks.size();
    //uses dynamically load balanced loop trips.  Tokens are assigned dynamically instead of statically
    while(true) {
      int i;
      if(group.thread_rank()==0) { //thread 0 nominated to get new token
        i=atomicAdd(params.pe_idx,1);      //get token index
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
          uint64_t* token_pack;
          TokenState *next_ts=NULL;
          //get cur_tok&token_pack addr
          Token *cur_tok = FindOrAddTokenArc(params, nextstate, total_cost, 
          acoustic_cost, &ts, j, true, 
          &next_ts, &token_pack, params.token_per_arc_update+j);
          //get cur_te&new_token_pack
          uint64_t new_token_pack=pack(-total_cost, j);
          uint64_t ret=atomicMax((unsigned long long *)token_pack, (unsigned long long)new_token_pack);
          if (ret<new_token_pack) {
            Token* cur_te=params.token_per_arc+j;
            store16(cur_te, &(Token(acoustic_cost+weight, params.frame, tok)));
            params.token_per_arc_update[j]=1;
          }
        } //end total_cost<=cutoff
      } //end arc loop
    } //end token loop
    __grid_sync_nv_internal(params.barrier);
    params.cur_toks.merge(params.token_per_arc,params.token_per_arc_update, params.numArcs, false);
  }
  
    template<int blockDimx, int blockDimy>
  DEVICE __inline__ void processNonEmittingTokens_function(processTokens_params &params, CostType cutoff, uint32_t size,  volatile int *modified, bool aggregate=false) {
    assert(size);
    auto group = cooperative_groups::tiled_partition<blockDimx>(cooperative_groups::this_thread_block());
    int* agg_tok_idx=params.agg_idx;
    int* cur_tok_idx=params.ne_idx;
    int tid=threadIdx.x+blockIdx.x*blockDim.x;
    if (threadIdx.x==0&&blockIdx.x==0) { *agg_tok_idx=*cur_tok_idx=0; }
    __grid_sync_nv_internal(params.barrier);
    if (aggregate) {
      for (tid;tid<size;tid+=blockDim.x*gridDim.x) {
        if(params.cur_toks.update(tid)) {
          int i=atomicAdd(agg_tok_idx,1);      //get changed token index for faster NE proc
          if (i>=size) break;
          params.ne_queue[i]=tid;
        }
      }
      __grid_sync_nv_internal(params.barrier);
    }
    if (params.verbose>3&&threadIdx.x==0 && blockIdx.x==0) GPU_PRINTF("PNE: %i %i %i\n",params.frame, params.cur_toks.size(), *agg_tok_idx);
    //uses dynamically load balanced loop trips.  Tokens are assigned dynamically instead of statically
    while(true) {
      int i,j;
      if(group.thread_rank()==0) { //thread 0 nominated to get new token
      if (aggregate) {
          j=atomicAdd(cur_tok_idx,1);      //get token index
          if (j>=*agg_tok_idx) i=size; // to exit
          else i=params.ne_queue[j];
      } else {
          i=atomicAdd(cur_tok_idx,1);      //get token index
      }
      }
      i=group.shfl(i,0);           //broadcast token index
      if(i>=size) break;//TODO: optimize this to reduce iteration
      
      TokenState& ts = params.cur_toks[i];
      Token * tok = ts.token;
      StateId state = ts.state;
      assert(params.ne_offsets); 
      uint32_t start=params.ne_offsets[state], finish=params.ne_offsets[state+1];
      for(int j=start+group.thread_rank();j<finish;j+=blockDimx) {
        BaseFloat weight = params.arc_weights[j];
        StateId nextstate = params.arc_nextstates[j];

        Token next_tok = Token(weight, params.frame, tok);

        CostType total_cost = tok->cost_ + weight;

      if (params.verbose>4) GPU_PRINTF("D: %i %i %i %i %i \n",threadIdx.x, threadIdx.y, j, blockIdx.x,i);
        if (next_tok.cost_ <= cutoff) {
          TokenState *next_ts=NULL;
          uint64_t* token_pack;
          Token *cur_tok = FindOrAddTokenArc(params, nextstate, total_cost, 
            0, &ts, j, true, &next_ts,
          &token_pack, params.token_per_arc_update+j);

         uint64_t new_token_pack=pack(-total_cost, j);
          uint64_t ret=atomicMax((unsigned long long *)token_pack, (unsigned long long)new_token_pack);
          if (ret<new_token_pack) {
            Token* cur_te=params.token_per_arc+j;
            store16(cur_te, &(Token(weight, params.frame, tok)));
            params.token_per_arc_update[j]=1;
            (*modified) = true;
          }
        }
      }

    }
    __grid_sync_nv_internal(params.barrier);
    params.cur_toks.merge(params.token_per_arc,params.token_per_arc_update, params.numArcs, false);
  }

  __launch_bounds__(64,64)
  __global__ void processTokens_cg(processTokens_params params, bool is_init=false) {
    bool rank0 = blockIdx.x==0 && threadIdx.x==0;

    if (!is_init) {
      findBestCutoff_function<32,2>(params);
      __grid_sync_nv_internal(params.barrier);
    }
   
    volatile int *modified = params.modified;    //modified flag for current iteration
    CostType cutoff=*params.cutoff;

    if (!is_init) {
      processEmittingTokens_function<32,2>(params);
      __grid_sync_nv_internal(params.barrier);  //ensure cur_toks size is final
    }
  
    int tok_E;
    int itv = params.verbose>2? 1: 10;
    if (rank0&&params.verbose>1&&params.frame%itv==0) 
      tok_E=params.cur_toks.size();

      int cnt=0;
      uint32_t size = 0;
      uint32_t psize=size;
    do {
      psize=size;
      size = params.cur_toks.size();
      cnt++;
      bool aggregate=(!is_init)&&cnt>1?1:0;
      __grid_sync_nv_internal(params.barrier); //wait for everyone to read size and modified
      if (rank0) *modified = false;
      __grid_sync_nv_internal(params.barrier); //wait for everyone to read size and modified

      processNonEmittingTokens_function<32,2>(params,cutoff,size,modified, aggregate);

      __grid_sync_nv_internal(params.barrier);  //wait for everyone to finish process tokens and writes modified
    } while ((*modified)==true);

    if (rank0&&params.verbose>1&&params.frame%itv==0) 
          GPU_PRINTF("TK: %i %i %i %f\n", params.frame, tok_E, params.cur_toks.size(), cutoff);


    //proc lattice before allocate new toks to TokenState
    params.lattice_pruner.CollectToksPerFrame(params.cur_toks.mem_d, 
                  *params.cur_toks.count_d, params.frame);
    params.lattice_pruner.CollectArcsPerFrame(params.lat_arcs_sub_vec, 
      params.lat_arcs_sub_vec.count_d, params.frame);
    

    __grid_sync_nv_internal(params.barrier); 

    allocateNewTokens_function(params.current_tokens_lookup, params.cur_toks, params.token_allocator);
  
    if(rank0) {
      //prepare for next iteration
      //params.prev_toks.clear(); //change to be done in PreProcessTokens
      *params.cutoff = INFINITY;
      *params.fb_idx=0;  
      *params.pe_idx=0;
    }
    params.cur_toks.clear_sub();
    __grid_sync_nv_internal(params.barrier);  //wait for allocation to finish
    
    if(rank0) {
      params.token_allocator.advanceFront(params.cur_toks.size());
    }
  }

  void CudaLatticeDecoder::initParams(processTokens_params& params) {
    params.prev_toks=(*prev_toks_);
    params.cur_toks=(*cur_toks_);
    params.current_tokens_lookup=current_tokens_lookup_d;
    params.cutoff=cutoff_d;
    params.lat_arcs_sub_vec = lat_arcs_buf_;
    params.token_per_arc=token_per_arc_d;
    params.token_per_arc_update=token_per_arc_update_d;

    params.token_allocator=token_allocator_;
    params.lattice_pruner=lattice_pruner_;

    params.e_offsets=fst_.e_offsets_d;
    params.ne_offsets=fst_.ne_offsets_d;
    params.arc_ilabels=fst_.arc_ilabels_d;
    params.arc_olabels=fst_.arc_olabels_d;
    params.arc_weights=fst_.arc_weights_d;
    params.arc_nextstates=fst_.arc_nextstates_d;

    params.loglikelihoods=loglikelihoods_d;
    params.modified=modified_d;
    params.pe_idx=pe_idx_d;
    params.ne_idx=ne_idx_d;
    params.ne_queue=ne_queue_d;
    params.fb_idx=fb_idx_d;
    params.agg_idx=agg_idx_d;
    params.barrier=barrier_d;

    params.beam=config_.beam;
    params.verbose=config_.verbose;
    params.lattice_beam=config_.lattice_beam;
    params.prune_interval = config_.prune_interval;
    params.numArcs=fst_.NumArcs();
    params.frame=num_frames_decoded_;
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
  void CudaLatticeDecoder::UpdateTokPointersByFrame(uint frame) {
    cur_toks_=&lat_toks_bufs_[frame%LAT_BUF_SIZE];
    prev_toks_=&lat_toks_bufs_[(frame-1)%LAT_BUF_SIZE];
    //single buffer in lat_arcs_buf_;
  }

  __launch_bounds__(64,64)
  __global__ void cuda_decoder_lat_prune_active_tokens(processTokens_params params) {
//    auto grid = cooperative_groups::this_grid();
    params.lattice_pruner.PruneActiveTokens(params.frame, params.lattice_beam, params.verbose
      );
  }

  void CudaLatticeDecoder::PruneActiveTokens(cudaStream_t wait_st, 
    cudaStream_t run_st, float gpu_ratio) {
    processTokens_params params;
    initParams(params);
    dim3 threads(64,1);
    dim3 blocks(DIV_ROUND_UP(total_threads*gpu_ratio,(threads.x*threads.y)));
    cudaStreamSynchronize(wait_st);
    if (params.verbose>1) KALDI_LOG <<"PruneActiveTokens, # of blocks: "<<blocks.x<<std::endl;
    cuda_decoder_lat_prune_active_tokens<<<blocks,threads,0,run_st>>>(params);
    //lattice_pruner_.CopyArcsToHost(num_frames_decoded_, st);
  }
  void CudaLatticeDecoder::PreProcessTokens() {
    PUSH_RANGE("PreProcessTokens", 1)
    num_frames_decoded_++;
#ifndef MEMADVISE
    //no need to prefetch if we have done a memadvise
    token_allocator_.prefetch_next_to_device(cudaStreamPerThread);
#endif
    //TODO prefetch here
    UpdateTokPointersByFrame(num_frames_decoded_);
    ClearToks(*cur_toks_);
    //dont need to clear arcs as we directly take the final buffer into this vector

    POP_RANGE
  }
  
  void CudaLatticeDecoder::ProcessTokens() {
    PUSH_RANGE("ProcessTokens", 2)
    KALDI_VLOG(4) << num_frames_decoded_<<std::endl;

    processTokens_params params;
    dim3 threads(64,1);
    dim3 blocks(DIV_ROUND_UP(total_threads,(threads.x*threads.y)));

    initParams(params);

    if (params.frame==1) KALDI_VLOG(2) <<"# of blocks: "<<blocks.x<<std::endl;

    cudaStreamWaitEvent(stream_comp,event_ll,0); //make sure log likelihoods are on the device before starting these kernels

    processTokens_cg<<<blocks,threads,0,stream_comp>>>(params);  //doesn't work
    cudaCheckError();
      
    cudaEventSynchronize(event_pt); //throttle
    cudaEventRecord(event_pt,stream_comp);



    POP_RANGE
  }

  void CudaLatticeDecoder::ClearToks(TokenMergeVector &toks) {
    //cannot actually delete tokens as they may still be connected to active tokens
    toks.clear(stream_comp);
  }
} // end namespace kaldi.
