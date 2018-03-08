// decoder/cuda-lattice-decoder.cu

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
#include <cub/cub.cuh>

#define USE_NVTX
#ifdef USE_NVTX
#include "nvToolsExt.h"
const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif


#define MEMADVISE //only in Pascal?: http://mug.mvapich.cse.ohio-state.edu/static/media/mug/presentations/2016/MUG16_GPU_tutorial_V5.pdf 

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

#define DIV_ROUND_UP(a,b) ((a+b-1)/b)
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

  //template class CudaVector<LatToken>; 
  //template class CudaVector<LatLink>; 
  //http://en.cppreference.com/w/cpp/language/class_template
  template HOST DEVICE LatLink& CudaVector<LatLink>::operator[](uint32_t idx); 
  //template HOST DEVICE LatLink& CudaMergeVector<LatLink>::operator[](uint32_t idx); 
  template HOST DEVICE TokenState& CudaVector<TokenState>::operator[](uint32_t idx); 
  template HOST DEVICE uint32_t  CudaVector<TokenState>::size() const; 
  template HOST DEVICE uint32_t  CudaVector<LatLink>::size() const; 
  //template HOST DEVICE uint32_t  CudaMergeVector<LatLink>::size() const; 

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


  template <typename T>
  DEVICE __forceinline__ void load16(T *a, const T *b) {
    const ulong2 *src = reinterpret_cast<const ulong2*>(b);
    ulong2 &dst = *reinterpret_cast<ulong2*>(a);
    asm("ld.global.v2.u64 {%0,%1}, [%2];" : "=l"(dst.x), "=l"(dst.y) : "l"(src));
  }
  
template <typename T>
  DEVICE __forceinline__ void store16(T *a, const T *b) {
    const ulong2 src = *reinterpret_cast<const ulong2*>(b);
    asm("st.global.v2.u64 [%0], {%1,%2};" :: "l"(a), "l"(src.x), "l"(src.y));
  }

// Assumptions: 1-d grid and blocks. No threads "early-exit" the grid.
// No stream priorities
DEVICE inline void __gpu_sync_fast(volatile int *fast_epoch)
{
    __syncthreads();
    if (threadIdx.x == 0) {
        // gridDim.x-1 blocks are adding 1
        // and one block is adding 0x80000000 - (gridDim.x-1)
        // so the whole sum is 0x80000000
        int nb = 1;
        if (blockIdx.x == 0) {
            nb = 0x80000000 - (gridDim.x-1);
        }
 
        int old_epoch = *fast_epoch;
        __threadfence();
        atomicAdd((int*)fast_epoch, nb);
 
        // wait for the sign bit to commute   
        int cnt=0;
        while (((*fast_epoch) ^ old_epoch) >= 0)//&& ++cnt!=(1<<19) //deadlock hack
            ;
        //if (blockIdx.x == 0) *fast_epoch=0;
    }
    __syncthreads();
}

DEVICE __noinline__ void __grid_sync_nv_internal(int *barrier)
{
    __gpu_sync_fast((volatile int*)barrier);
}

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
    inline void CudaVector<T>::allocate(uint32_t max_size, 
        uint32_t* icount_h, uint32_t* icount_d) {
      this->max_size=max_size;

      count_h=icount_h;
      count_d=icount_d;
      cudaMemset(count_d, 0,sizeof(uint32_t));
      *count_h=0;

      cudaMalloc(&mem_d,max_size*sizeof(T));
      cudaMallocHost(&mem_h,max_size*sizeof(T));
    }


  template<typename T>
    inline void CudaVector<T>::allocate(uint32_t max_size) {
      this->max_size=max_size;

      cudaMallocHost(&count_h,sizeof(uint32_t));
      cudaMalloc(&count_d, sizeof(uint32_t));
      cudaMemset(count_d, 0,sizeof(uint32_t));
      *count_h=0;

      cudaMalloc(&mem_d,max_size*sizeof(T));
      cudaMallocHost(&mem_h,max_size*sizeof(T));
    }

  template<typename T>
    inline size_t CudaVector<T>::getCudaMallocBytes() {
      return sizeof(uint32_t)+max_size*sizeof(T);
    }

  template<typename T>
    inline void CudaVector<T>::free(bool create_outside) { 
      cudaFree(mem_d); 
      cudaFreeHost(mem_h);
      if (!create_outside) {
        cudaFreeHost(count_h);
        cudaFree(count_d); 
      }
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
    inline void CudaVector<T>::copy_data_to_host(cudaStream_t stream, void* to_buf, bool copy_size) {
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
      //assert(*count_h<max_size);
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

__global__ void getCnt_function( int* vec_len_acc, uint32_t* vec_len, uint32_t* count_d, int sub_vec_num) {
  int acc=0;
  for (int i=0; i<sub_vec_num; i++) {
    vec_len_acc[i]=acc;
    acc+=(vec_len[i]);
  }
  vec_len_acc[sub_vec_num]=acc;
  *count_d=acc;
}
__global__ void copyArr_function(char **arr, int* vec_len_acc, char *to, int psize, uint32_t* vec_len, uint32_t* count_d, int sub_vec_num, int* barrier) {
  int rank0=blockIdx.x==0&&threadIdx.x==0?1:0;
  int batch=blockDim.x;
  int i=blockIdx.x;
  int tid = threadIdx.x;
  if (rank0) {
    int acc=0;
    for (int i=0; i<sub_vec_num; i++) {
      vec_len_acc[i]=acc;
      acc+=(vec_len[i]);
    }
    vec_len_acc[sub_vec_num]=acc;
    *count_d=acc;
  }
  __grid_sync_nv_internal(barrier);
  int sz = vec_len_acc[i+1]-vec_len_acc[i];

  for(; tid < sz; tid += batch) {
    memcpy(to+psize*(tid+vec_len_acc[i]),arr[i]+tid*psize,psize);
  }
}
template<typename T> 
void CudaMergeVector<T>::load(CudaVector<T>*in, int sub_vec_num, cudaStream_t st, int total_threads, uint32_t* count_vec_d) {
  //loading vec_len_acc, arr
  int acc=0;
  uint32_t* vec_len;
  T** arr;
  if (count_vec_d) vec_len=count_vec_d;
  else {
    KALDI_ERR<<"unsupported since 9cc21bbd868906d06d40d1b230d61690a584927e";
  }
  arr=arr_;
  //we have to do this first as copy_data_to_host need count_d
  //getCnt_function<<<1,1,0,st>>>(vec_len_acc_, vec_len, count_d, sub_vec_num);
  //cudaStreamSynchronize(st);
  copyArr_function<<<sub_vec_num,min(1024,min(total_threads,max_size)/sub_vec_num),0,st>>>
    ((char**)arr,vec_len_acc_,(char*)mem_d,sizeof(T), vec_len, count_d, sub_vec_num, barrier_);
  cudaCheckError();
  //this->copy_data_to_host(st);
}
template<typename T> 
void CudaMergeVector<T>::reg(CudaVector<T>*in, int sub_vec_num, cudaStream_t st) {
    int32_t device;
    cudaGetDevice(&device);
    for (int i=0; i<sub_vec_num; i++) arr_[i]=in[i].mem_d;
    arr_[sub_vec_num]=NULL;
    cudaMemPrefetchAsync(arr_,sizeof(T*)*(sub_vec_num+1), device, st);
    cudaCheckError();
}
template<typename T> 
void CudaMergeVector<T>::allocate(uint32_t max_size) {
    CudaVector<T>::allocate(max_size);
    cudaMallocManaged(&arr_,MAX_SUB_VEC_SIZE*sizeof(T*));
    cudaMalloc(&vec_len_acc_,MAX_SUB_VEC_SIZE*sizeof(int));
    cudaMalloc(&barrier_,sizeof(int));
    cudaMemset(barrier_,0,sizeof(int32)); 
}
template<typename T> 
void CudaMergeVector<T>::free() {
    CudaVector<T>::free();
    cudaFree(arr_);
    cudaFree(vec_len_acc_);
}


  /***************************************CudaFst Implementation*****************************************/
  HOST DEVICE float CudaFst::Final(StateId state) const {
    #ifdef __CUDA_ARCH__
    return final_d[state];
    #else
    return final_h[state];
    #endif

  }
  void CudaFst::initialize(const fst::Fst<StdArc> &fst) {
    nvtxRangePushA("CudaFst constructor");
    bytes_cudaMalloc=0;
    //count states since Fst doesn't provide this functionality
    numStates=0;
    for( fst::StateIterator<fst::Fst<StdArc> > iter(fst); !iter.Done(); iter.Next()) {
      numStates++;
    }
    start=fst.Start();
    cudaMallocHost(&final_h,sizeof(float)*numStates);
    cudaMalloc(&final_d,sizeof(float)*numStates);

    //allocate and initialize offset arrays
    e_offsets_h=(unsigned int *)malloc(sizeof(unsigned int)*(numStates+1));
    ne_offsets_h=(unsigned int *)malloc(sizeof(unsigned int)*(numStates+1));

    cudaMalloc((void**)&e_offsets_d,sizeof(unsigned int)*(numStates+1)); bytes_cudaMalloc+=sizeof(unsigned int)*(numStates+1);
    cudaMalloc((void**)&ne_offsets_d,sizeof(unsigned int)*(numStates+1)); bytes_cudaMalloc+=sizeof(unsigned int)*(numStates+1);

    memset(e_offsets_h,0,sizeof(unsigned int)*(numStates+1));
    memset(ne_offsets_h,0,sizeof(unsigned int)*(numStates+1));

    //iterate through states and arcs and count number of arcs per state
    e_count=0;
    ne_count=0;
    max_ilabel=0;

    for(int i=0;i<numStates;i++) {
      final_h[i]=fst.Final(i).Value();
      //count emmiting and non_emitting arcs
      for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, i); !aiter.Done(); aiter.Next()) {
        StdArc arc = aiter.Value();
        int32 ilabel = arc.ilabel;
        int32 olabel = arc.olabel;

        if(ilabel>max_ilabel) {
          max_ilabel=ilabel;
        }

        if(ilabel!=0) { //emitting
          e_count++;
        } else { //non-emitting
          ne_count++;
        }
      }
      ne_offsets_h[i+1]=ne_count;
      e_offsets_h[i+1]=e_count;
    }

    //offset ne_offsets by the number of emitting arcs
    for(int i=0;i<numStates+1;i++) {
      e_offsets_h[i]+=1;          //add dummy arc at the beginingg.
      ne_offsets_h[i]+=e_count+1;   //add dummy arc and put e_arcs before
    }

    arc_count=e_count+ne_count+1;

    cudaMemcpyAsync(final_d,final_h,sizeof(float)*numStates,cudaMemcpyHostToDevice,cudaStreamPerThread);
    
    cudaMemcpyAsync(e_offsets_d,e_offsets_h,sizeof(unsigned int)*(numStates+1),cudaMemcpyHostToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(ne_offsets_d,ne_offsets_h,sizeof(unsigned int)*(numStates+1),cudaMemcpyHostToDevice, cudaStreamPerThread);


    //Allocate non-zero arrays
    cudaMallocHost(&arc_weights_h,arc_count*sizeof(BaseFloat));
    cudaMallocHost(&arc_nextstates_h,arc_count*sizeof(StateId));
    cudaMallocHost(&arc_ilabels_h,arc_count*sizeof(int32));
    cudaMallocHost(&arc_olabels_h,arc_count*sizeof(int32));

    cudaMalloc((void**)&arc_weights_d,arc_count*sizeof(BaseFloat));  bytes_cudaMalloc+=arc_count*sizeof(BaseFloat);
    cudaMalloc((void**)&arc_nextstates_d,arc_count*sizeof(StateId));  bytes_cudaMalloc+=arc_count*sizeof(StateId);
    cudaMalloc((void**)&arc_ilabels_d,arc_count*sizeof(int32));  bytes_cudaMalloc+=arc_count*sizeof(int32);
    cudaMalloc((void**)&arc_olabels_d,arc_count*sizeof(int32));  bytes_cudaMalloc+=arc_count*sizeof(int32);
    
    //now populate arc data
    int e_idx=1;          //save room for dummy arc (so start at 1)
    int ne_idx=e_count+1; //starts where e_offsets ends

    //create dummy arc
    arc_weights_h[0]=StdWeight::One().Value();
    arc_nextstates_h[0]=fst.Start();
    arc_ilabels_h[0]=0;
    arc_olabels_h[0]=0;

    for(int i=0;i<numStates;i++) {
      //count emmiting and non_emitting arcs

      for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, i); !aiter.Done(); aiter.Next()) {
        StdArc arc = aiter.Value();
        int idx;
        if(arc.ilabel!=0) { //emitting
          idx=e_idx++;
        } else {
          idx=ne_idx++;
        }
        arc_weights_h[idx]=arc.weight.Value();
        arc_nextstates_h[idx]=arc.nextstate;
        arc_ilabels_h[idx]=arc.ilabel;
        arc_olabels_h[idx]=arc.olabel;
      }
    }

    cudaMemcpyAsync(arc_weights_d,arc_weights_h,arc_count*sizeof(BaseFloat),cudaMemcpyHostToDevice,cudaStreamPerThread);
    cudaMemcpyAsync(arc_nextstates_d,arc_nextstates_h,arc_count*sizeof(StateId),cudaMemcpyHostToDevice,cudaStreamPerThread);
    cudaMemcpyAsync(arc_ilabels_d,arc_ilabels_h, arc_count*sizeof(int32),cudaMemcpyHostToDevice,cudaStreamPerThread);
    cudaMemcpyAsync(arc_olabels_d,arc_olabels_h, arc_count*sizeof(int32),cudaMemcpyHostToDevice,cudaStreamPerThread);

    cudaStreamSynchronize(cudaStreamPerThread);
    nvtxRangePop();
  }

  void CudaFst::finalize() {
    nvtxRangePushA("CudaFst destructor");
    printf("CudaFst::finalize()\n");
    cudaFreeHost(final_h);
    cudaFree(final_d);
    free(e_offsets_h);
    free(ne_offsets_h);

    cudaFree(e_offsets_d);
    cudaFree(ne_offsets_d);

    cudaFreeHost(arc_weights_h);
    cudaFreeHost(arc_nextstates_h);
    cudaFreeHost(arc_ilabels_h);
    cudaFreeHost(arc_olabels_h);

    cudaFree(arc_weights_d);
    cudaFree(arc_nextstates_d);
    cudaFree(arc_ilabels_d);
    cudaFree(arc_olabels_d);
    nvtxRangePop();
  }

  /***************************************End CudaFst****************************************************/

  DEVICE inline void allocateAllTokens_function(TokenLookupElem *current_tokens_lookup, int32 numStates,  CudaLatticeDecoder::TokenAllocator allocator) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<numStates; i+=blockDim.x*gridDim.x) {
      Token *token = allocator.getToken(i);
      token->cost_ = INFINITY;
      token->prev_ = NULL;
      token->frame= -1;
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
      token->prev_ = NULL;
      token->frame= -1;
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
    for (int i=0; i<LAT_BUF_SIZE; i++) 
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
    prune_interval_=1; //config.prune_interval;
    sub_vec_num_=config.sub_vec_num;
    max_arcs_=config.max_arcs;

    //lattice: should use manage
    cudaMallocHost((void**)&arcs_buf_,sizeof(LatLink)*(config.max_arcs)); 
    cudaMallocManaged((void**)&arc_copy_buf_,sizeof(LatLinkVectorMerge)*(LAT_BUF_SIZE)); 
    for (int j=0; j<LAT_BUF_SIZE; j++) {
      cudaMallocHost((void**)&lat_arcs_sub_vec_buf_count_[j][0],sizeof(uint32_t)*(config.sub_vec_num)); 
      //coount bytes_cudaMalloc in the loop below
      cudaMalloc((void**)&lat_arcs_sub_vec_buf_count_[j][1],sizeof(uint32_t)*(config.sub_vec_num)); 

      cudaMallocManaged((void**)&lat_arcs_sub_vec_buf_[j] ,sizeof(LatLinkVector)*(config.sub_vec_num)); 
      for (int i=0; i < config.sub_vec_num; i++) {
        lat_arcs_sub_vec_buf_[j][i].allocate(
            config.max_lat_arc_per_frame*(2.0/config.sub_vec_num), //because it isn't always average
            lat_arcs_sub_vec_buf_count_[j][0]+i,
            lat_arcs_sub_vec_buf_count_[j][1]+i); 
        bytes_cudaMalloc += lat_arcs_sub_vec_buf_[j][i].getCudaMallocBytes(); 
      }  
      toks_buf_[j].allocate(config.max_tokens_per_frame);
      bytes_cudaMalloc+=toks_buf_[j].getCudaMallocBytes();
      arc_copy_buf_[j].allocate(config.max_lat_arc_per_frame);
      bytes_cudaMalloc+=arc_copy_buf_[j].getCudaMallocBytes();
      cudaCheckError();
      arc_copy_buf_[j].reg(lat_arcs_sub_vec_buf_[j], config.sub_vec_num, stream_copy[0]);
    }
    cudaMalloc((void**)&tok2scansum_numarc_d,sizeof(int32_t)*(config.max_tokens_per_frame)); 
    max_arcs_per_frame_search_=config.max_lat_arc_per_frame*10;
    cudaMalloc((void**)&tid2arc_d,sizeof(int32_t)*(max_arcs_per_frame_search_)); 
    cudaMalloc((void**)&tid2tok_d,sizeof(int32_t)*(max_arcs_per_frame_search_)); 
    d_temp_storage=NULL;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
        tok2scansum_numarc_d, tok2scansum_numarc_d, config.max_tokens_per_frame);
    cudaMalloc((void**)&d_temp_storage,temp_storage_bytes); 

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
      for (int i=0; i < sub_vec_num_; i++) {
        lat_arcs_sub_vec_buf_[j][i].free(true);
      }
      cudaFree(lat_arcs_sub_vec_buf_[j]);
      if (j<LAT_BUF_SIZE) {
        arc_copy_buf_[j].free();
      }
      cudaFreeHost(lat_arcs_sub_vec_buf_count_[j][0]);
      cudaFree(lat_arcs_sub_vec_buf_count_[j][1]);
    }
    cudaFree(arc_copy_buf_);
    cudaFreeHost(arcs_buf_);
    
    cudaFree(tid2tok_d);
    cudaFree(tid2arc_d);
    cudaFree(tok2scansum_numarc_d);
    cudaFree(d_temp_storage);
    
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
    for (int i=0; i<LAT_BUF_SIZE; i++) 
      cudaStreamDestroy(stream_copy[i]);
    cudaStreamDestroy(stream_ll);

  }
  DEVICE inline Token* FindOrAddTokenArc(processTokens_params& params,
    StateId nextstate, CostType total_cost, CostType acoustic_cost,
    TokenState* ts, uint32_t j, bool add_arc, int32_t subid, TokenState** next_ts) {
    //TokenLookupElem lookup_elem;
    //load16(&lookup_elem, &params.current_tokens_lookup[nextstate]);
    TokenLookupElem& lookup_elem = params.current_tokens_lookup[nextstate];
    Token *cur_tok = lookup_elem.token;  
    //check if token is active or not.  Double check the lock.
    if(lookup_elem.active==0 && atomicCAS(&lookup_elem.active,0,1)==0) {        //grab sentinal to see who gets to add to cur_toks list
      //if havent seen, add into hash
      lookup_elem.tokenstate_idx=params.cur_toks.push_back(TokenState(cur_tok,nextstate,total_cost));
      const uint32_t* arc_offset=params.e_offsets;
      params.tok2scansum_numarc[lookup_elem.tokenstate_idx]=arc_offset[nextstate+1]
        -arc_offset[nextstate];
    }
    //need both 2 steps below, to ensure tokenstate_idx won't run into error
    while (lookup_elem.tokenstate_idx == -1);//hasnt pushed
    __threadfence(); 
    *next_ts=&params.cur_toks[lookup_elem.tokenstate_idx];
    //we shouldnt do tid2arc in frame 0
    if (add_arc) {
      Token *prev_tok = ts->token;  
      int ts_id=prev_tok->frame==params.frame?
      params.cur_toks.get_idx_from_addr(ts):
      params.prev_toks.get_idx_from_addr(ts);
      
      LatLink arc=LatLink(ts_id, prev_tok->frame, 
        lookup_elem.tokenstate_idx, params.frame,
        params.arc_ilabels[j], params.arc_olabels[j],
        params.arc_weights[j], acoustic_cost); //duplicate arcs in NE
      int32_t lat_arc_idx=params.lat_arcs_sub_vec[subid].push_back(arc);
    }
    return cur_tok;  
  }
  __global__ void addOneToken(processTokens_params params, StateId state) {
    TokenState *next_ts=NULL;
    Token* cur_tok=FindOrAddTokenArc(params, state, 0, //add first token
      0, NULL, -1, false, 0, &next_ts);
    Token tok(0, NULL);
    *cur_tok = tok;
    cur_tok->frame=params.frame;
  }

  //putting this into a kernel to avoid extra latency of a memory copy
  __global__ void initializeCutoff(CostType *cutoff) {
    *cutoff = INFINITY;
  }
  void CudaLatticeDecoder::ClearArcVector(LatLinkVector* lat_arcs_sub_vec_) {
    //std::swap(lat_arcs_sub_vec_prev_, lat_arcs_sub_vec_); //change to be done in PreProcessTokens
    //replaced with faster imp
    //for (int i=0; i < sub_vec_num_; i++) {
    //  lat_arcs_sub_vec_[i].clear(stream_comp);  
    //}
    int i=0;
    for (; i<LAT_BUF_SIZE; i++) {
      if (lat_arcs_sub_vec_buf_[i]==lat_arcs_sub_vec_) break;
    }
    assert(i<LAT_BUF_SIZE);
    uint32_t** cur_count_buf=lat_arcs_sub_vec_buf_count_[i];
    cudaMemsetAsync(cur_count_buf[0],0,sub_vec_num_*sizeof(uint32_t),stream_comp);
    cudaMemsetAsync(cur_count_buf[1],0,sub_vec_num_*sizeof(uint32_t),stream_comp);
  }
  void CudaLatticeDecoder::InitDecoding() {
    printf("CUDA LatticeDecoder InitDecoding\n");
    num_frames_decoded_ = 0;
  // clean up from last time:
    for (int i=0; i<LAT_BUF_SIZE; i++) {
      ClearToks(toks_buf_[i]);
      ClearArcVector(lat_arcs_sub_vec_buf_[i]);
    }    
    SetTokArcPointerByFrame(num_frames_decoded_);
    arcs_buf_used_=0;

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
    //cur_toks_->copy_size_to_host(stream_comp); //for PreProcessTokens
  }

  void CudaLatticeDecoder::PreFinalizeDecoding() { 
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

  BaseFloat CudaLatticeDecoder::FinalRelativeCost() const {
    // as a special case, if there are no active tokens at all (e.g. some kind of
    // pruning failure), return infinity.
    CostType infinity = std::numeric_limits<CostType>::infinity();
    if ((*cur_toks_).empty())
      return infinity;
    CostType best_cost = infinity,
             best_cost_with_final = infinity;


    //for each active token
    //compute minimum cost
    for (int i=0;i<(*cur_toks_).size();i++) {
      TokenState ts = (*cur_toks_)[i];
      StateId state = ts.state;
      CostType cost = ts.token->cost_;

      // Note: Plus is taking the minimum cost, since we're in the tropical
      // semiring.
      best_cost = std::min(best_cost, cost);
      best_cost_with_final = std::min(best_cost_with_final,
          cost +
          fst_.Final(state));
          //fst_.Final(state).Value());
    }

    BaseFloat extra_cost = best_cost_with_final - best_cost;
    if (extra_cost != extra_cost) { // NaN.  This shouldn't happen; it indicates some
      // kind of error, most likely.
      KALDI_WARN << "Found NaN (likely search failure in decoding)";
      return infinity;
    }
    // Note: extra_cost will be infinity if no states were final.
    return extra_cost;
  }

  // Outputs an FST corresponding to the single best path
  // through the lattice.
  bool CudaLatticeDecoder::GetBestPath(Lattice *fst_out, bool use_final_probs) const {
    nvtxRangePushA("GetBestPath");
    assert(0);
    return true;
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

  void CudaLatticeDecoder::ComputeLogLikelihoods(DecodableInterface *decodable) {
    PUSH_RANGE("ComputeLogLikelihoods",3)
    int32 frame = num_frames_decoded_;//TODO
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
  inline DEVICE void findBestCutoff_tid2arc_function(processTokens_params params) {

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
      assert(params.tok2scansum_numarc[i+1]-params.tok2scansum_numarc[i]==finish-start);
      
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
        int arc_i=j-start;
        int idx=params.tok2scansum_numarc[i]+arc_i;
        params.tid2arc[idx]=j;
        params.tid2tok[idx]=i;
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
    int tid=blockIdx.x*blockDimx+threadIdx.x;

    CostType cutoff=*params.cutoff;
    assert(params.prev_toks.size());
    int32 size = params.tok2scansum_numarc[params.prev_toks.size()];
    __grid_sync_nv_internal(params.barrier);
    //uses dynamically load balanced loop trips.  Tokens are assigned dynamically instead of statically
    if(tid==0) { //thread 0 nominated to get new token
      if (params.verbose>3) {
        printf("E: %i %i\n", size, params.prev_toks.size());
      }
      assert(size<=params.max_arcs_per_frame_search);
      assert(params.tid2tok[size-1]<=params.prev_toks.size()-1);
    }
    while(true) {
      //i=__shfl_sync(0xffffffff,i,0);
      if(tid>=size) break;
      int j=params.tid2arc[tid];
      int i=params.tid2tok[tid];
      TokenState& ts = params.prev_toks[i];
      Token * tok = ts.token;
      StateId state = ts.state;

      int32 ilabel=params.arc_ilabels[j];  //prefetch ilabel since it leads to a dependent load
      BaseFloat acoustic_cost = -params.loglikelihoods[ilabel];  //TODO can I prefetch this?  
      BaseFloat weight = params.arc_weights[j];
      StateId nextstate = params.arc_nextstates[j];

      CostType total_cost = tok->cost_ + weight + acoustic_cost;

      if(total_cost<=cutoff) 
      {
        Token next_tok =  Token(acoustic_cost+weight, tok);
        TokenState *next_ts=NULL;
        Token *cur_tok = FindOrAddTokenArc(params, nextstate, total_cost, 
          acoustic_cost, &ts, j, true, (i+j)%params.sub_vec_num, &next_ts);
        
        volatile Token* cur_tokv = reinterpret_cast<volatile Token*>(cur_tok);  //need volatile reads to ensure we don't get cached versions

        while(*cur_tokv < next_tok) {   //check if we need to update
        acquire_semaphore((int*)&params.token_locks[nextstate]);
            if(*cur_tokv < next_tok) {                                                                          //recheck if we are min           

              //if(sizeof(Token)==16)
              //  store16(cur_tok,&next_tok);                                                                       //update token
              //else
              //  *cur_tok=next_tok;
              cur_tok->cost_=next_tok.cost_;
              cur_tok->frame=params.frame;
              next_ts->cost_=cur_tok->cost_;
            }
            release_semaphore((int*)&params.token_locks[nextstate]);
            break;                                                                                              //exit loop as our update is done
        } //end while
      } //end total_cost<=cutoff
      tid+=blockDimx*gridDim.x;
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
          printf("NE: %i %i %i\n", i, threadIdx.x, blockIdx.x);
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

        Token next_tok = Token(weight, tok);

        CostType total_cost = tok->cost_ + weight;

      if (params.verbose>4) printf("D: %i %i %i %i %i \n",threadIdx.x, threadIdx.y, j, blockIdx.x,i);
        if (next_tok.cost_ <= cutoff) {
          TokenState *next_ts=NULL;
          Token *cur_tok = FindOrAddTokenArc(params, nextstate, total_cost, 
            0, &ts, j, true, (i+j)%params.sub_vec_num, &next_ts);

          volatile Token* cur_tokv = reinterpret_cast<volatile Token*>(cur_tok);  //need volatile reads to ensure we don't get cached versions

          while(*cur_tokv < next_tok) {   //check if we need to update
            acquire_semaphore((int*)&params.token_locks[nextstate]);
              if(*cur_tokv < next_tok) {                                                                     //recheck that we are minimum
                //if(sizeof(Token)==16)
                //  store16(cur_tok,&next_tok);                                                                       //update token
                //else
                //  *cur_tok=next_tok;
                cur_tok->cost_=next_tok.cost_;
                cur_tok->frame=params.frame;
                next_ts->cost_=cur_tok->cost_;
                (*modified) = true;                                                                            //mark as updated
              }
            release_semaphore((int*)&params.token_locks[nextstate]);
              break;  //exit loop as our update is done
          } //end try update loop
        }
      }

    }
      if (params.verbose>4) printf("ED: %i %i %i \n",threadIdx.x, group.thread_rank(), blockIdx.x);
  }

  //Loop through all tokens repeatdly updating costs until nothing changes
  //__launch_bounds__(64,32)
  __global__ void processNonEmittingTokens_cg(processTokens_params params) {

    //auto grid = cooperative_groups::this_grid();
    //double buffer to reduce synchronization
    volatile int *modified0 = params.modified;    //modified flag for current iteration
    volatile int *modified1 = params.modified+1;  //modified flag for next/last iteration
    *modified1 = false;

    CostType cutoff=*params.cutoff;
    do {

      uint32_t size = params.cur_toks.size();

      *params.ne_idx=0;
      //grid.sync();  
      __grid_sync_nv_internal(params.barrier);

      //swap buffers
      swap(modified0,modified1);

      *modified1 = false;

      processNonEmittingTokens_function<32,2>(params,cutoff,size,modified0);

      //grid.sync();
      __grid_sync_nv_internal(params.barrier);

    } while ((*modified0)==true);
    
    //prepare for next iteration
    *params.cutoff = INFINITY;

    allocateNewTokens_function(params.current_tokens_lookup, params.cur_toks, params.allocator);
    __grid_sync_nv_internal(params.barrier);
    if(threadIdx.x==0 && blockIdx.x==0)
      params.allocator.advanceFront(params.cur_toks.size());
  }

  __launch_bounds__(64,64)
  __global__ void processTokens_cg(processTokens_params params) {
//    auto grid = cooperative_groups::this_grid();

    bool rank0 = blockIdx.x==0 && threadIdx.x==0;
    int p=0;
    if(rank0) {
      cub::DeviceScan::ExclusiveSum(params.d_temp_storage, params.temp_storage_bytes, 
        params.tok2scansum_numarc, params.tok2scansum_numarc, params.prev_toks.size()+1);
    }
    __grid_sync_nv_internal(params.barrier);  //wait for allocation to finish

    if(rank0&&params.verbose>4)  
    {p++;printf("S: %i\n",p);}

    findBestCutoff_tid2arc_function<32,2>(params);
    //grid.sync();
    __grid_sync_nv_internal(params.barrier);
   
   
    if(rank0&&params.verbose>4)  
    {p++;printf("S: %i\n",p);}

    volatile int *modified0 = params.modified;    //modified flag for current iteration
    volatile int *modified1 = params.modified+1;  //modified flag for next/last iteration
    *modified1 = false;
    CostType cutoff=*params.cutoff;

    processEmittingTokens_function<64,1>(params);
    //grid.sync();
    __grid_sync_nv_internal(params.barrier);  //ensure cur_toks size is final
  
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
          printf("TK: %i %i %i %f\n", params.frame, tok_E, params.cur_toks.size(), cutoff);

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
    params.lat_arcs_sub_vec = lat_arcs_sub_vec_;
    params.sub_vec_num=sub_vec_num_;
    params.tid2tok=tid2tok_d;
    params.tok2scansum_numarc=tok2scansum_numarc_d;
    params.tid2arc=tid2arc_d;
    params.max_arcs_per_frame_search=max_arcs_per_frame_search_;
    params.d_temp_storage=d_temp_storage;
    params.temp_storage_bytes=temp_storage_bytes;
  }
  void CudaLatticeDecoder::ProcessNonemitting() {
    nvtxRangePushA("ProcessNonemitting");

    dim3 threads(32,1);

    dim3 blocks(DIV_ROUND_UP(total_threads,(threads.x*threads.y)));

    processTokens_params params;
    initParams(params);


#if 0
    void *args[] = { (void*) &params };

    cudaLaunchCooperativeKernel((void*)processNonEmittingTokens_cg, blocks, threads, args, 0, stream_comp);
#else
    processNonEmittingTokens_cg<<<blocks,threads,0,stream_comp>>>(params);
#endif

    cudaCheckError();
    nvtxRangePop();
  }
  void CudaLatticeDecoder::SetTokArcPointerByFrame(uint frame) {
    cur_toks_=&toks_buf_[frame%LAT_BUF_SIZE];
    prev_toks_=&toks_buf_[(frame-1)%LAT_BUF_SIZE];
    lat_arcs_sub_vec_=lat_arcs_sub_vec_buf_[frame%LAT_BUF_SIZE];
    lat_arcs_sub_vec_prev_=lat_arcs_sub_vec_buf_[(frame-1)%LAT_BUF_SIZE];    
  }
  void CudaLatticeDecoder::PostProcessLattices(bool islast, uint dec_frame) {
    PUSH_RANGE("PostProcessLattices",1); 
    uint prev_idx=(dec_frame-1)%LAT_BUF_SIZE;
    uint prev_idx2=(dec_frame-1)%(LAT_BUF_SIZE-1);
    uint pprev_idx=(dec_frame-2)%LAT_BUF_SIZE;
    uint pprev_idx2=(dec_frame-2)%(LAT_BUF_SIZE-1);
 
    {//do this first because there is a sync inner to get count first
      //cudaMemPrefetchAsync(&cur_vec,sizeof(LatLinkVectorMerge), cudaCpuDeviceId, stream_copy[prev_idx]);
      cudaStreamSynchronize(stream_copy[prev_idx]);
      LatLinkVectorMerge& cur_vec=arc_copy_buf_[prev_idx];
      cur_vec.copy_data_to_host(stream_copy[prev_idx], arcs_buf_+arcs_buf_used_, false);
      assert(arcs_buf_used_+cur_vec.size()<=max_arcs_);
      arcs_ready2cpu_[prev_idx]=arcs_buf_+arcs_buf_used_;
      arcs_buf_used_+=cur_vec.size();
      //cudaCheckError();
      (toks_buf_[prev_idx]).copy_data_to_host(stream_copy[prev_idx], NULL, false);
    }
    cudaCheckError();


    //if (islast) 
      //cudaStreamSynchronize(stream_copy); //else overlap CPU&GPU
    cudaStreamSynchronize(stream_comp);
    POP_RANGE
  }

  void CudaLatticeDecoder::PreProcessLattices(TokenVector** pprev_toks, 
    void** pprev_arcs, int *num_arcs, bool islast, int* lat_frame, uint dec_frame) {
    PUSH_RANGE("PreProcessLattices_and_Wait",0)
    uint prev_idx=(dec_frame-1)%LAT_BUF_SIZE;
    uint prev_idx2=(dec_frame-1)%(LAT_BUF_SIZE-1);
    uint pprev_idx=(dec_frame-2)%LAT_BUF_SIZE;
    uint pprev_idx2=(dec_frame-2)%(LAT_BUF_SIZE-1);
    *lat_frame=dec_frame-2; //for CPU
    {
      LatLinkVectorMerge& cur_vec=arc_copy_buf_[prev_idx];
      cur_vec.load(lat_arcs_sub_vec_buf_[prev_idx], 
          sub_vec_num_, stream_copy[prev_idx], total_threads,
          lat_arcs_sub_vec_buf_count_[prev_idx][1]);
      cudaCheckError();
      cur_vec.copy_size_to_host(stream_copy[prev_idx]);
      (toks_buf_[prev_idx]).copy_size_to_host(stream_copy[prev_idx]);
    }
    //stream_copy[pprev_idx]
    cudaStreamSynchronize(stream_copy[pprev_idx]);
    cudaCheckError();
    *pprev_toks = &toks_buf_[pprev_idx];
    *pprev_arcs=arcs_ready2cpu_[pprev_idx];
    *num_arcs=arc_copy_buf_[pprev_idx].size();
    POP_RANGE
  }
  void CudaLatticeDecoder::PreProcessTokens() {
    nvtxRangePushA("PreProcessTokens"); 
    //before reset, we should update tid2arc_d for the next frame
    num_frames_decoded_++;
    if (num_frames_decoded_) {
      //cudaStreamSynchronize(stream_comp);
      //size_t tmp_len;
      //assert(cur_toks_->size());
      //cub::DeviceScan::ExclusiveSum(NULL, tmp_len, 
      //  tok2scansum_numarc_d, tok2scansum_numarc_d, cur_toks_->size()+1);
      //cudaMalloc((void**)&d_temp_storage,tmp_len); 
      //cub::DeviceScan::ExclusiveSum(d_temp_storage, tmp_len, 

      //cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
      //  tok2scansum_numarc_d, tok2scansum_numarc_d, cur_toks_->size()+1, stream_comp);
      //cudaFree(d_temp_storage);
      cudaCheckError();
    }

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
      ClearArcVector(lat_arcs_sub_vec_);
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
    //cur_toks_->copy_size_to_host(stream_comp); //for PreProcessTokens
      
    cudaEventSynchronize(event_pt); //throttle
    cudaEventRecord(event_pt,stream_comp);



    POP_RANGE
  }

  void CudaLatticeDecoder::ClearToks(TokenVector &toks) {
    //cannot acctually delete tokens as they may still be connected to active tokens
    toks.clear(stream_comp);
  }
} // end namespace kaldi.
