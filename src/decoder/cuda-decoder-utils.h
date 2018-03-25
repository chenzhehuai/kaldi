// decoder/cuda-decoder-utils.cuh

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

#ifndef KALDI_CUDA_DECODER_UTILS_H_
#define KALDI_CUDA_DECODER_UTILS_H_


#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "util/stl-utils.h"
#include "fst/fstlib.h"
#include "lat/determinize-lattice-pruned.h"
#include "lat/kaldi-lattice.h"
#include "itf/decodable-itf.h"
#include "omp.h"
#include "cuda_runtime.h"

namespace kaldi {


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
#define GPU_PRINTF(format,...) printf(format, ##__VA_ARGS__)
#else
#define VERBOSE 0
#define GPU_PRINTF(format,...)
#endif

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


//#define MEMADVISE //only in Pascal?: http://mug.mvapich.cse.ohio-state.edu/static/media/mug/presentations/2016/MUG16_GPU_tutorial_V5.pdf 

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

#define DIV_ROUND_UP(a,b) ((a+b-1)/b)


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

  
template <typename T>
  DEVICE __forceinline__ void store32(T *a, const T *b) {
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
    inline uint32_t NumArcs() const {  return numArcs; }
    inline StateId Start() const { return start; }    
    HOST DEVICE inline float Final(StateId state) const;
    size_t getCudaMallocBytes() const { return bytes_cudaMalloc; }
  
    unsigned int numStates;               //total number of states
    unsigned int numArcs;               //total number of states
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

} // end namespace kaldi.


