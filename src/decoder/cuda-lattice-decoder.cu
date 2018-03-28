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

typedef CudaLatticeDecoder::Token Token;
typedef CudaLatticeDecoder::StateId StateId;
typedef CudaLatticeDecoder::TokenState TokenState;
typedef CudaLatticeDecoder::CostType CostType;
typedef CudaLatticeDecoder::TokenLookupElem TokenLookupElem;
typedef CudaLatticeDecoder::LatLink LatLink;
typedef CudaLatticeDecoder::LatLinkVector LatLinkVector;
typedef CudaLatticeDecoder::TokenMergeVector TokenMergeVector;
typedef CudaLatticeDecoder::processTokens_params processTokens_params;
typedef CudaLatticeDecoder::LatticePruner LatticePruner;
#define CudaVector CudaLatticeDecoder::CudaVector
#define CudaMergeVector CudaLatticeDecoder::CudaMergeVector

// instantiation of templates
template HOST DEVICE LatLink& CudaVector<LatLink>::operator[](uint32 idx);
template HOST DEVICE TokenState& CudaVector<TokenState>::operator[](uint32 idx);
template HOST DEVICE uint32  CudaVector<TokenState>::Size() const;
template HOST DEVICE uint32  CudaVector<LatLink>::Size() const;

// inline functions

// for speedup purpose, make them inline (5% 0.165->0.158)
// we define them here but not in a shared header file, because they
// are device code and can't be defined in a header file

// In atomic based token recombination, we pack the
// cost and the arc index into an uint64 to represent the token
// before recombination, with the former one in the higher bits
// for comparison purpose.
inline DEVICE uint64 pack_cost_idx_into_uint64(BaseFloat cost, int32 idx) {
  //assert (!isnan(cost));
  //assert (idx >= 0 && idx < 1L<<32);
  uint32 i_cost = *(uint32 *) & cost;
  if (i_cost & 0x80000000)
    i_cost = i_cost ^ 0xFFFFFFFF;
  else
    i_cost = i_cost ^ 0x80000000;
  return (uint64)i_cost << 32 | idx;
}

// Unpacks a cost
inline DEVICE BaseFloat unpack_cost_from_uint64(uint64 packed) {
  uint32 i_cost = packed >> 32;
  if (i_cost & 0x80000000)
    i_cost = i_cost ^ 0x80000000;
  else
    i_cost = i_cost ^ 0xFFFFFFFF;
  return *(BaseFloat *) & i_cost;
}

// Unpacks a idx for tracing the data
inline DEVICE int32 unpack_idx_from_uint64(uint64 packed) {
  //assert (!(packed & 0x80000000));
  return packed & 0x7FFFFFFF;
}

// fast load 16 bits using CUDA ASM
inline  DEVICE void cuda_load16(void *a, const void *b) {
  const ulong2 *src = reinterpret_cast<const ulong2*>(b);
  ulong2 &dst = *reinterpret_cast<ulong2*>(a);
  asm("ld.global.v2.u64 {%0,%1}, [%2];" : "=l"(dst.x), "=l"(dst.y) : "l"(src));
}

// fast store 16 bits using CUDA ASM
inline  DEVICE void cuda_store16(void *a, const void *b) {
  const ulong2 src = *reinterpret_cast<const ulong2*>(b);
  asm("st.global.v2.u64 [%0], {%1,%2};" :: "l"(a), "l"(src.x), "l"(src.y));
}

// TODO: we need a fast 32 bits storing function
inline  DEVICE void cuda_store32(void *a, const void *b) {
  memcpy(a, b, 32);
}

// overload CUDA atomicMin to consume double
inline DEVICE void atomic_min(double *address, double val) {
  unsigned long long *address_ull = (unsigned long long *)address;
  double minval = *address;
  while (val < minval) {  //if my value is less than minimum
    minval = val;         //update the minimum to my value locally
    //write minimum and read back value
    val = __longlong_as_double(atomicExch(address_ull, __double_as_longlong(val)));
  } //if the new value is < the minimum I wrote I need to try again.
}

// overload CUDA atomicMin to consume BaseFloat
inline DEVICE void atomic_min(BaseFloat *address, BaseFloat val) {
  uint32 *address_ui = (uint32  *)address;
  BaseFloat minval = *address;
  while (val < minval) {  //if my value is less than minimum
    minval = val;         //update the minimum to my value locally
    //write minimum and read back value
    val = __uint_as_float(atomicExch(address_ui, __float_as_uint(val)));
  } //if the new value is < the minimum I wrote I need to try again.
}

// swap code in device, as we need to instantiate them, so we define it here
template<typename T>
inline DEVICE void cuda_swap(T &a, T &b) {
  T c = a;
  a = b;
  b = c;
}

// end of inline functions

// CudaVector Implementation
template<typename T>
inline void CudaVector<T>::Allocate(uint32 max_size,
                    uint32* count_h, uint32* count_d, T* mem_h, T* mem_d) {
  alloc_size = 0;
  this->max_size = max_size;

  if (count_h) this->count_h = count_h;
  else cudaMallocHost(&this->count_h, sizeof(uint32));
  if (count_d) this->count_d = count_d;
  else {
    alloc_size += sizeof(uint32);
    cudaMalloc(&this->count_d, sizeof(uint32));
  }
  if (mem_h) this->mem_h = mem_h;
  else cudaMallocHost(&this->mem_h, max_size * sizeof(T));
  if (mem_d) this->mem_d = mem_d;
  else {
    alloc_size += max_size * sizeof(T);
    cudaMalloc(&this->mem_d, max_size * sizeof(T));
  }

  cudaMemset(this->count_d, 0, sizeof(uint32));
  *this->count_h = 0;
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
HOST DEVICE inline T& CudaVector<T>::operator[](uint32 idx) {
#ifdef __CUDA_ARCH__
  assert(idx < *count_d);
  return mem_d[idx];
#else
  assert(idx < *count_h);
  return mem_h[idx];
#endif
}

template<typename T>
HOST DEVICE inline const T& CudaVector<T>::operator[](uint32 idx) const {
#ifdef __CUDA_ARCH__
  assert(idx < *count_d);
  return mem_d[idx];
#else
  assert(idx < *count_h);
  return mem_h[idx];
#endif
}

// This will cause page faults back and forth when we switch from host to device.
// need to call e.g. CopySizeToHost() before this function
template<typename T>
HOST DEVICE inline uint32 CudaVector<T>::Size() const {
#ifdef __CUDA_ARCH__
  return *count_d;
#else
  return *count_h;
#endif
}

// push back function implemented
// by an atomic operation, where the memory is pre-allocated
template<typename T>
HOST DEVICE inline uint32 CudaVector<T>::PushBack(const T &val) {
#ifdef __CUDA_ARCH__
  assert(*count_d < max_size);
  uint32 idx = atomicAdd(count_d, 1);
  mem_d[idx] = val;
#else
  assert(*count_h < max_size);
  uint32 idx = (*count_h)++;
  mem_h[idx] = val;
#endif
  return idx;
}

template<typename T>
HOST DEVICE inline void CudaVector<T>::Clear(cudaStream_t stream) {
#ifdef __CUDA_ARCH__
  *count_d = 0;
#else
  *count_h = 0;
  cudaMemsetAsync(count_d, 0, sizeof(int32), stream);
#endif
}

template<typename T>
inline void CudaVector<T>::Swap(CudaVector<T> &v) {
  std::swap(mem_h, v.mem_h);
  std::swap(mem_d, v.mem_d);
  std::swap(count_h, v.count_h);
  std::swap(count_d, v.count_d);
  std::swap(max_size, v.max_size);
}

// given an allocated address in vector memory, calculate its index in the vector
template<typename T>
HOST DEVICE inline int32 CudaVector<T>::GetIdxFromAddr(T* addr) {
#ifdef __CUDA_ARCH__
  int32 ret = addr - mem_d;
  assert(ret < *count_d && ret >= 0);
  return ret;
#else
  int32 ret = addr - mem_h;
  assert(ret < *count_h && ret >= 0);
  return ret;
#endif
}

// a series of data transfer functions between host and device
template<typename T>
inline void CudaVector<T>::CopyAllToHost(cudaStream_t stream) {
  cudaStreamSynchronize(stream);
  cudaMemcpy(count_h, count_d, sizeof(int32), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(mem_h, mem_d, *count_h * sizeof(T), cudaMemcpyDeviceToHost,
                  stream);
}

template<typename T>
inline void CudaVector<T>::CopyAllToDevice(cudaStream_t stream) {
  cudaStreamSynchronize(stream);
  cudaMemcpyAsync(count_d, count_h, sizeof(int32), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(mem_d, mem_h, *count_h * sizeof(T), cudaMemcpyHostToDevice,
                  stream);
}

template<typename T>
inline void CudaVector<T>::CopySizeToHost(cudaStream_t stream) {
  cudaMemcpyAsync(count_h, count_d, sizeof(int32), cudaMemcpyDeviceToHost, stream);
}

template<typename T>
inline void CudaVector<T>::CopySizeToDevice(cudaStream_t stream) {
  cudaMemcpyAsync(count_d, count_h, sizeof(int32), cudaMemcpyHostToDevice, stream);
}

template<typename T>
inline void CudaVector<T>::CopyDataToHost(cudaStream_t stream, T* to_buf,
                                          bool copy_size) {
  if (!to_buf) {
    to_buf = mem_h;
  }
  if (copy_size) cudaMemcpy(count_h, count_d, sizeof(int32),
                            cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(to_buf, mem_d, *count_h * sizeof(T), cudaMemcpyDeviceToHost,
                  stream);
}

template<typename T>
inline void CudaVector<T>::CopyDataToDevice(cudaStream_t stream) {
  cudaMemcpyAsync(mem_d, mem_h, *count_h * sizeof(T), cudaMemcpyHostToDevice,
                  stream);
}

// CudaVector Implementation

template<typename T>
inline void CudaMergeVector<T>::Allocate(uint32 max_size) {
  CudaVector<T>::Allocate(max_size);

  cudaMalloc(&mem_pack_buf_d, sizeof(uint64*) * max_size);
  cudaMalloc(&mem_update_d, sizeof(int32) * max_size);
  cudaMalloc(&barrier_, sizeof(int32) * 1);

  cudaMemset(mem_update_d, 0, sizeof(int32) * max_size);
}

template<typename T>
inline void CudaMergeVector<T>::Free() {
  CudaVector<T>::Free();

  cudaFree(mem_pack_buf_d);
  cudaFree(mem_update_d);
  cudaFree(barrier_);
}

template<typename T>
inline void CudaMergeVector<T>::Swap(CudaMergeVector<T> &v) {
  CudaVector<T>::Swap(v);
  std::swap(mem_update_d, v.mem_update_d);
}

template<typename T>
inline size_t CudaMergeVector<T>::GetCudaMallocBytes() {
  return CudaVector<T>::GetCudaMallocBytes() +
         sizeof(uint32) * (1 + 2 * (2)) + max_size * (sizeof(T) + 
         sizeof(uint64*) + sizeof(int32));
}


template<typename T>
DEVICE inline void CudaMergeVector<T>::StoreDataByPackIdx(
  void* temp_data_buf, int* temp_data_buf_update, int32 buf_size) {
  assert(0);  // haven't implemented
}

// according to the unpack index, copy data from external buf to the inside
// buf; it's used in the 2nd stage of 2-pass atomic token recombination
// Namely, in each frame, we save the token
// information in an array whose size is the number of arcs. This
// ensures there are no write conflicts between threads since each
// arc can be accessed at most once in each frame. After passing
// all tokens, we aggregate survived packed tokens, unpack them
// to get arc indexes, and store token information from the former
// array to token data structures exploiting thread parallelism.
template<>
DEVICE inline void CudaMergeVector<TokenState>::StoreDataByPackIdx(
  void* temp_data_buf, int* temp_data_buf_update, int32 buf_size) {
  int32 tid = threadIdx.x + blockIdx.x * blockDim.x;
  int32 batch = blockDim.x * gridDim.x;
  int32 size = *count_d; // count_d is cleared in Clear() called by InitDecoding()

  for (; tid < size; tid += batch) { // thread parallelism
    uint64* pack_v = mem_pack_buf_d[tid];
    int32 idx = unpack_idx_from_uint64(*pack_v);
    assert(idx < buf_size);
    mem_update_d[(tid + 0)] = temp_data_buf_update[idx];
    if (temp_data_buf_update[idx]) temp_data_buf_update[idx] = 0;
    else continue; // if it isn't updated, just skip storing
    TokenState* to_ts = mem_d + (tid + 0);
    Token* cur_tok = ((Token *)temp_data_buf) + idx;
    Token* to_tok = to_ts->token;
    cuda_store16(to_tok, cur_tok); //memcpy(to_tok,cur_tok,sizeof(T));
  }
}

// check whether the item in index i of the vector is updated in this frame
// call this function after StoreDataByPackIdx()
template<typename T>
DEVICE inline int32 CudaMergeVector<T>::IsUpdated(int32 i) {
  if (i >= *count_d) return 0;
  return mem_update_d[i];
}

// push back function implemented
// by an atomic operation, where the memory is pre-allocated
template<typename T>
DEVICE inline uint32 CudaMergeVector<T>::PushBack(const T &val,
    uint64 *val_pack) {
  uint32 idx = atomicAdd(count_d, 1);
  assert(*count_d < max_size);
  assert(sizeof(val) == 16); // use faster storing
  cuda_store16(&mem_d[idx], &val);
  // store the pack_data pointer in 1st stage
  mem_pack_buf_d[idx] = val_pack; // used in StoreDataByPackIdx() in 2nd stage
  return idx;
}

// LatticePruner Implementation

// Initialize in InitDecoding()
void LatticePruner::Initialize() {
  cudaMemset(arcs_apr_fr_size_d, 0, sizeof(int32) * (prune_interval + 1));
  cudaMemset(arcs_apr_used_d, 0, sizeof(int32));
  cudaMemset(arcs_bpr_used_d, 0, sizeof(int32));
  cudaMemset(toks_bpr_fr_sidx_d, 0, sizeof(int32) * (prune_interval + 1));
  cudaMemset(arcs_bpr_fr_sidx_d, 0, sizeof(int32) * (prune_interval + 1));
}

int32 LatticePruner::Allocate(int32 max_tokens_per_frame,
                              int32 max_lat_arc_per_frame, int32 prune_interval, 
                              int32 max_toks, int32 max_arcs) {
  int32 sz;
  int32 bytes_cuda_malloc = 0;

  //before pruning
  sz = sizeof(Token) * max_toks;
  cudaMalloc((void**)&toks_bpr_d, sz); bytes_cuda_malloc += sz;
  cudaMallocHost((void**)&toks_bpr_h, sz);
  toks_buf_before_pr_size = sz / sizeof(Token);
  sz = sizeof(LatLink) * max_arcs;
  cudaMalloc((void**)&arcs_bpr_d, sz); bytes_cuda_malloc += sz;
  arcs_buf_before_pr_size = sz / sizeof(LatLink);
  sz = sizeof(int32) * (prune_interval + 1);
  cudaMalloc((void**)&toks_bpr_fr_sidx_d, sz); bytes_cuda_malloc += sz;
  cudaMallocHost((void**)&toks_bpr_fr_sidx_h, sz);
  sz = sizeof(int32) * (prune_interval + 1);
  cudaMalloc((void**)&arcs_bpr_fr_sidx_d, sz); bytes_cuda_malloc += sz;

  //after pruning
  sz = sizeof(int32) * (prune_interval + 1);
  cudaMalloc((void**)&arcs_apr_fr_size_d, sz); bytes_cuda_malloc += sz;
  cudaMallocHost((void**)&arcs_apr_fr_size_h, sz);
  sz = kEstimatedPruneRatio * sizeof(LatLink) * max_arcs;
  cudaMalloc((void**)&arcs_apr_d, sz); bytes_cuda_malloc += sz;
  cudaMallocHost((void**)&arcs_apr_h, sz);
  sz = sizeof(int32);
  cudaMalloc((void**)&arcs_apr_used_d, sz); bytes_cuda_malloc += sz;
  cudaMalloc((void**)&arcs_bpr_used_d, sz); bytes_cuda_malloc += sz;
  cudaMallocHost((void**)&arcs_apr_used_h, sz);

  // GPU global memory temp variables
  sz = sizeof(int32);
  cudaMalloc((void**)&barrier_, sz); bytes_cuda_malloc += sz;
  sz = sizeof(int32) * 3;
  cudaMalloc((void**)&modified_d, sz); bytes_cuda_malloc += sz;
  sz = sizeof(int32) * (2);
  cudaMalloc((void**)&count_vec_acc_d, sz); bytes_cuda_malloc += sz;
  this->prune_interval = prune_interval;

  return bytes_cuda_malloc;
}
void LatticePruner::Free() {
  //before pruning
  cudaFree(arcs_bpr_used_d);
  cudaFreeHost(arcs_apr_used_h);
  cudaFree(toks_bpr_d);
  cudaFreeHost(toks_bpr_h);
  cudaFree(arcs_bpr_d);
  cudaFree(toks_bpr_fr_sidx_d);
  cudaFreeHost(toks_bpr_fr_sidx_h);
  cudaFree(arcs_bpr_fr_sidx_d);

  //after pruning
  cudaFree(arcs_apr_fr_size_d);
  cudaFreeHost(arcs_apr_fr_size_h);
  cudaFree(arcs_apr_d);
  cudaFree(arcs_apr_used_d);

  // GPU global memory temp variables
  cudaFree(count_vec_acc_d);
  cudaFree(barrier_);
  cudaFree(modified_d);
  cudaFreeHost(arcs_apr_h);
}

// entry of lattice pruning until this frame
inline DEVICE void LatticePruner::PruneActiveTokens(int32 frame,
    BaseFloat lattice_beam, int32 verbose) {
  int32 rank0 = threadIdx.x == 0 && blockIdx.x == 0 ? 1 : 0;
  if (frame == 0) return;
  if (rank0) *arcs_apr_used_d = 0; // clear buffer index
  __grid_sync_nv_internal(barrier_);
  for (int32 f = frame; f > 0; f--) { //prune each frame in serial
    PruneLatticeForFrame(f, 1, lattice_beam, verbose);
  }
  // by kEstimatedPruneRatio to reduce memory allocation and D2H data transfer
  assert(*arcs_apr_used_d < arcs_buf_before_pr_size * kEstimatedPruneRatio);
  if (verbose > 2 && rank0) 
    CUDA_PRINTF("PRt: %i %i\n", arcs_bpr_fr_sidx_d[frame + 1],
                *arcs_apr_used_d);
}

// collect after each token passing, we store Token data in the sequence of 
// TokenState vector, using continuous memory
inline DEVICE void LatticePruner::CollectToksPerFrame(
  TokenMergeVector& cur_toks_vec, int32 frame) {
  int32 tid = threadIdx.x + blockIdx.x * blockDim.x;
  TokenState* cur_toks = cur_toks_vec.mem_d;
  int32 size = cur_toks_vec.Size();
  if (tid == 0) {
    // Set start index in the buffer of the next frame
    SetNextSidx(toks_bpr_fr_sidx_d, size, frame); 
  }
  for (; tid < size; tid += gridDim.x * blockDim.x) {
    Token* to_tok = GetActiveToken(frame, tid, false);
    cuda_store16(to_tok, cur_toks[tid].token);
    /* 
    // for debug purpose
    assert(cur_toks[tid].token->frame==frame);
    GetActiveToken(frame,tid,true);
    */
  }
}

// collect after each token passing, mainly to update arcs_bpr_fr_sidx_d here
inline DEVICE void LatticePruner::CollectArcsPerFrame(LatLinkVector&
    cur_arc_array,
    int32 frame) {
  int32 tid = threadIdx.x + blockIdx.x * blockDim.x;
  int32 idx = tid;
  int32 rank0 = blockIdx.x == 0 && threadIdx.x == 0 ? 1 : 0;
  int32 batch = blockDim.x * gridDim.x;

  int32 size = cur_arc_array.Size() - *arcs_bpr_used_d; // size of current frame
  __grid_sync_nv_internal(barrier_);
  if (rank0) {
    SetNextSidx(arcs_bpr_fr_sidx_d, size, frame);
    *arcs_bpr_used_d = cur_arc_array.Size();
    // we didn't clear cur_arc_array.count_d until the end of decoding
  }
  /*
  // we share the memory between vector&pruner, so dont need to copy between them
  for(; idx < size; idx += batch) {
    LatLink* to_arc=GetActiveArc(frame,(idx));
    cuda_store32(to_arc, cur_arc_array.mem_d+idx);
    // for debug purpose
    GetActiveToken((cur_arc_array.mem_d+idx)->p1,true,frame);
    GetActiveToken(to_arc->p1,true,frame);
  }
  */
}

// AddArc function implemented
// by an atomic operation, where the memory is pre-allocated
DEVICE int32 LatticePruner::AddArc(LatLink* arc) {
  int32 i = atomicAdd(arcs_apr_used_d, 1);
  cuda_store32(arcs_apr_d + i, arc);
}

// Set start index in the buffer of the next frame
inline DEVICE void LatticePruner::SetNextSidx(int* sidx_buf, int32 size,
    int32 frame) {
  assert(frame >= 0);
  int32 cur_sidx = sidx_buf[(frame)];
  sidx_buf[(frame + 1)] = cur_sidx + size;
}

// Get the active token indexed by a uint64 pair (frame, idx), stored in void* p
// the details of the pair can be referred to LatLink::LatLink()
inline DEVICE Token* LatticePruner::GetActiveToken(void* p, bool check,
    int32 iframe) const {
  int32 frame, id;
  DECODE_TOK_IDX_PAIR(frame, id, (uint64)p);
  if (check) assert(frame == iframe || frame == iframe - 1);
  return GetActiveToken(frame, id, check);
}

// Get the active token indexed by a uint64 pair (frame, idx)
// the details of the pair can be referred to LatLink::LatLink()
inline DEVICE Token* LatticePruner::GetActiveToken(int32 frame, int32 id,
    bool check) const {

  int32 cur_sidx = toks_bpr_fr_sidx_d[frame];
  assert(cur_sidx + id < toks_buf_before_pr_size);
  Token* tok = toks_bpr_d + cur_sidx + id;
  if (check) {
    assert(tok->frame == frame);
  }
  return tok;
}

// Get the active arc indexed by a uint64 pair (frame, idx)
// the vector memory and the start index of each frame are kept in LatticePruner
inline DEVICE LatLink* LatticePruner::GetActiveArc(int32 frame, int32 id) const {
  int32 cur_sidx = arcs_bpr_fr_sidx_d[(frame)];
  assert(cur_sidx + id < arcs_buf_before_pr_size);
  LatLink* arc = arcs_bpr_d + cur_sidx + id;
  return arc;
}

// Size of items in the frame, it is obtained from an accumulate number array
inline DEVICE int32 LatticePruner::GetSize(int* acc_len, int32 frame) const {
  int32 size = acc_len[(frame) + 1] - acc_len[(frame)];
  assert(size >= 0 && size <= arcs_buf_before_pr_size);
  return size;
}

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
// remaining arcs, thus nodes are implicitly pruned. iii) node D2H
// copy is done in each frame asynchronously, which does not introduce overheads.
inline DEVICE void UpdateModifiedBuf(volatile int32 **modified0, volatile int32** modified1,
    volatile int32 ** modified2, int cnt, int32* modified_d) {
  *modified0=modified_d+cnt%3;
  *modified1=modified_d+(cnt+1)%3;
  *modified2=modified_d+(cnt+2)%3;
}
inline DEVICE void LatticePruner::PruneLatticeForFrame(int32 frame,
    bool merge, BaseFloat lattice_beam, int32 verbose) {
  int32 prev_cidx;
  int32 c = 0;
  int32 rank0 = threadIdx.x == 0 && blockIdx.x == 0 ? 1 : 0;
  volatile int32 *modified0;
  volatile int32 *modified1;
  volatile int32 *modified2;
  int32 cnt = 0;
  UpdateModifiedBuf(&modified0, &modified1, &modified2, cnt, modified_d);
  if (rank0 && verbose > 3) CUDA_PRINTF("%i %i\n", c++, GetSize(toks_bpr_fr_sidx_d,
                                          frame - 1)); //size before pruning
  { // initialize
    int32 tid = threadIdx.x + blockIdx.x * blockDim.x;
    int32 size = GetSize(toks_bpr_fr_sidx_d, frame - 1);
    for (; tid < size; tid += gridDim.x * blockDim.x) {
      Token* tok = GetActiveToken(frame - 1, tid, true);
      tok->extra_cost = FLT_MAX;
    }
    if (rank0) {//wait for last iteration(frame+1) finish
      *modified0 = 1;
      *modified1 = 0;
      *modified2 = 0;
      prev_cidx = *arcs_apr_used_d;
    }
    __grid_sync_nv_internal(barrier_);
  }

  //update arc, need a loop here until lattice stop modifying
  while (cnt++ < 10 && *modified0 != 0) {
    //__grid_sync_nv_internal(barrier_); //wait for every thread to enter while
    //cuda_swap(modified0, modified1); // double buffer to eliminate a grid sync after *modified1 = 0;
    UpdateModifiedBuf(&modified0, &modified1, &modified2, cnt, modified_d);
    if (rank0) *modified1 = 0; // till now, threads are using modified0 & modified2, so we clear modified1 here as it won't be used until grid sync below
    
    int32 tid = threadIdx.x + blockIdx.x * blockDim.x;
    int32 size = GetSize(arcs_bpr_fr_sidx_d, frame);
    for (; tid < size; tid += gridDim.x * blockDim.x) {
      LatLink* link = GetActiveArc(frame, tid);
      Token* next_tok = GetActiveToken(link->p1, true, frame);
      Token* tok = GetActiveToken(link->p2, true, frame);
      BaseFloat link_extra_cost = next_tok->extra_cost +
                                  ((tok->cost_ + link->acoustic_cost + link->graph_cost)
                                   - next_tok->cost_);
      if (isnan(link_extra_cost) || link_extra_cost > lattice_beam)
        ; //should be pruned
      else {
        //debug
        if (link_extra_cost < -1) {
          CUDA_PRINTF("%i %f %f %f %f %f\n", frame, next_tok->extra_cost, tok->cost_,
                      link->acoustic_cost, link->graph_cost, next_tok->cost_);
        }
        if (link_extra_cost < tok->extra_cost) {
          atomic_min(&tok->extra_cost, link_extra_cost);
          if (*modified0 == 0) atomicAdd((int32 *)modified0, 1);
        }
      }
    }
    __grid_sync_nv_internal(barrier_);
    if (rank0 && verbose > 3) CUDA_PRINTF("%i %i\n", c++, cnt);
    //if we do this always in 25 frames, we might dont need this
    //some flag to show whether it is changed
  }
  if (rank0 && verbose > 3) CUDA_PRINTF("cnt: %i\n", cnt);
  {
    int32 tid = threadIdx.x + blockIdx.x * blockDim.x;
    int32 size = GetSize(arcs_bpr_fr_sidx_d, frame);
    for (; tid < size; tid += gridDim.x * blockDim.x) {
      LatLink* link = GetActiveArc(frame, tid);
      Token* next_tok = GetActiveToken(link->p1, true, frame);
      Token* tok = GetActiveToken(link->p2, true, frame);
      BaseFloat link_extra_cost = next_tok->extra_cost +
                                  ((tok->cost_ + link->acoustic_cost + link->graph_cost)
                                   - next_tok->cost_);
      if (!isnan(link_extra_cost) && link_extra_cost <= lattice_beam) {
        //shouldn't be pruned
        if (merge) {
          AddArc(link);
          //if have seen, we can delete this
          //link->acoustic_cost=CUDART_NAN_F;
        }
      }
    }
    __grid_sync_nv_internal(barrier_);
    if (rank0 && verbose > 3) CUDA_PRINTF("%i\n", c++);
  }

  /*{
    //update tok
    int32 tid=threadIdx.x+blockIdx.x*blockDim.x;
    int32 size=GetSize(toks_bpr_fr_sidx_d,frame);
    for (;tid<size;tid+=gridDim.x*blockDim.x) {
      Token* tok=GetActiveToken(frame-1,tid);
      if (tok->extra_cost==FLT_MAX)
        tok->tot_cost=CUDART_NAN_F; //prune
    }
  } */

  //get size
  if (merge && rank0) {
    int& size_arc_of_frame = arcs_apr_fr_size_d[frame];
    size_arc_of_frame = *arcs_apr_used_d - prev_cidx;
    if (verbose > 3) CUDA_PRINTF("PR %i %i %i\n", frame,
                                   GetSize(arcs_bpr_fr_sidx_d, frame), size_arc_of_frame);
    //size_tok_of_frame[f-1]=cidx-prev_cidx
    //prev_cidx=cidx
  }
  //__grid_sync_nv_internal(barrier_);
  if (rank0 && verbose > 3) CUDA_PRINTF("%i\n", c++);
}
//#define GET_ARC_BUF_HOST_BY_FRAME(frame) (arcs_apr_h+arcs_apr_h_used)


void LatticePruner::CopyArcsToHost(int32 frame, cudaStream_t st) {
  int32 sz;
  //TODO: optimize out this
  cudaMemcpy(arcs_apr_used_h, arcs_apr_used_d,
             sizeof(int32), cudaMemcpyDeviceToHost);
  //
  sz = sizeof(LatLink) * (*arcs_apr_used_h);
  //sz=sizeof(LatLink)*(arcs_buf_before_pr_size*0.5); //assume 0.5 parts are pruned
  cudaMemcpyAsync(arcs_apr_h, arcs_apr_d,
                  sz, cudaMemcpyDeviceToHost, st);
  //we can call it because currently *arcs_buf_after_pr_size_arr_used_d is static len,
  //which is only used to save size but not any token
  sz = sizeof(int32) * (frame + 1) * (1);
  cudaMemcpyAsync(arcs_apr_fr_size_h, arcs_apr_fr_size_d,
                  sz, cudaMemcpyDeviceToHost, st);
  // clear arcs_apr_used_d in GPU
}
void LatticePruner::CopyToksToHost(int32 frame, cudaStream_t st) {
  int32 sz;
  sz = sizeof(int32) * (frame + 1 + 1) * (1); //include frame 0 & finalsum
  cudaMemcpy(toks_bpr_fr_sidx_h, toks_bpr_fr_sidx_d,
             sz, cudaMemcpyDeviceToHost);
  sz = sizeof(Token) * (toks_bpr_fr_sidx_h[frame + 1]);
  assert(sz);
  cudaMemcpyAsync(toks_bpr_h, toks_bpr_d,
                  sz, cudaMemcpyDeviceToHost, st);
}

void LatticePruner::GetHostData (Token** toks_buf,
                                 int** toks_fr_sidx, LatLink** arcs_buf, int** arcs_fr_size) {
  *toks_fr_sidx = toks_bpr_fr_sidx_h;
  *toks_buf = toks_bpr_h;
  *arcs_fr_size = arcs_apr_fr_size_h; //next prune_interval len
  *arcs_buf = arcs_apr_h; //start of next prune_interval len arcs
}

DEVICE inline void allocateAllTokens_function(TokenLookupElem
    *current_tokens_lookup, int32 numStates,
    CudaLatticeDecoder::TokenAllocator allocator) {
  for (int32 i = blockIdx.x * blockDim.x + threadIdx.x; i < numStates;
       i += blockDim.x * gridDim.x) {
    Token *token = allocator.GetToken(i);
    token->cost_ = INFINITY;
    token->extra_cost = 0;
    token->frame = -1;
    //token->state_id= -1;
    TokenLookupElem elem;
    elem.token = token;
    elem.active = false;
    elem.tokenstate_idx = -1;
    elem.token_pack = pack_cost_idx_into_uint64(-FLT_MAX, 0);
    memcpy(&current_tokens_lookup[i], &elem, sizeof(TokenLookupElem));
  }
}
__global__ void allocateAllTokens(TokenLookupElem *current_tokens_lookup,
                                  int32 numStates,  CudaLatticeDecoder::TokenAllocator allocator, int32 *barrier) {
  allocateAllTokens_function(current_tokens_lookup, numStates, allocator);
  __grid_sync_nv_internal(barrier);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    allocator.AdvanceFront(numStates);
  }
}

DEVICE inline void allocateNewTokens_function(TokenLookupElem
    *current_tokens_lookup, TokenMergeVector cur_toks,
    CudaLatticeDecoder::TokenAllocator allocator) {
  int32 size = cur_toks.Size();
  for (int32 i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    Token *token = allocator.GetToken(i);
    token->cost_ = INFINITY;
    token->extra_cost = 0;
    token->frame = -1;
    //token->state_id= -1;
    //a CPU copy of cur_toks can still be used, cur_toks will be clear in PreProcessTokens
    StateId state = cur_toks[i].state;
    //cur_toks[i].token->arc_index_=-1; // clear here will result in page fault in prefetch
    TokenLookupElem elem;
    elem.token = token;
    elem.active = false;
    elem.tokenstate_idx = -1;
    elem.token_pack = pack_cost_idx_into_uint64(-FLT_MAX, 0);
    memcpy(&current_tokens_lookup[state], &elem, sizeof(TokenLookupElem));
  }
}


void CudaLatticeDecoder::TokenAllocator::PrefetchNextToDevice(
  cudaStream_t stream) {
  PrefetchNextToDevice(stream, prefetch_size);
}

void CudaLatticeDecoder::TokenAllocator::PrefetchNextToDevice(cudaStream_t stream,
    int32 count) {
  int32 front = *front_h;
  //clamp to maximum size
  if (count > size - front)
    count = size - front;

#ifdef MEMADVISE
  cudaMemPrefetchAsync(tokens_allocation + front, sizeof(Token)*count, device,
                       stream);
#endif
}

void CudaLatticeDecoder::TokenAllocator::PrefetchAllocatedToHostForce(
  cudaStream_t stream) {
  if (!*front_h) return;
  cudaMemcpyAsync(tokens_allocation, tokens_allocation, sizeof(Token) * *front_h,
                  cudaMemcpyDeviceToHost, stream);
}

void CudaLatticeDecoder::TokenAllocator::PrefetchAllocatedToHost(
  cudaStream_t stream) {
  nvtxRangePushA("PrefetchAllocatedToHost");
#ifdef MEMADVISE
  if (!*front_h) return;
  cudaMemPrefetchAsync(tokens_allocation, sizeof(Token) * *front_h, cudaCpuDeviceId,
                       stream);
#endif
  nvtxRangePop();
}

size_t CudaLatticeDecoder::TokenAllocator::GetCudaMallocManagedBytes() {
  return bytes_cuda_malloc_managed;
}

void CudaLatticeDecoder::TokenAllocator::Reset() {
  *front_h = 0;
  cudaMemset(front_d, 0, sizeof(int32));
}

void CudaLatticeDecoder::TokenAllocator::Initialize(uint32 size)  {
  cudaGetDevice(&device);
  prefetch_size = 250000;

  this->size = size;

  //managed so getBestPath can easily access this data in the end
  cudaMallocManaged((void**)&tokens_allocation, sizeof(Token)*size);
  bytes_cuda_malloc_managed = sizeof(Token) * size;

  cudaMalloc((void**)&front_d, sizeof(uint32));
  cudaMallocHost((void**)&front_h, sizeof(uint32));

#ifdef MEMADVISE
  //If we do this we get faster perf as long as we don't over subscribe
  cudaMemAdvise(tokens_allocation, sizeof(Token)*size,
                cudaMemAdviseSetPreferredLocation, device);
  cudaMemPrefetchAsync(tokens_allocation, sizeof(Token)*size,
                       device); //force pages to allocate now
#endif

  Reset();
}

void CudaLatticeDecoder::TokenAllocator::Finalize() {
  printf("TokenAllocator::Finalize()\n");
  cudaFree(tokens_allocation);
  cudaFree(front_d);
  cudaFreeHost(front_h);
}

DEVICE inline Token* CudaLatticeDecoder::TokenAllocator::GetToken(
  uint32 offset) {
  int32 idx = *front_d + offset;
  return &tokens_allocation[idx];
}

DEVICE inline void CudaLatticeDecoder::TokenAllocator::AdvanceFront(
  uint32 num) {
  int32 front = *front_d + num;
  assert(front < size);

  *front_d = front;
  *front_h = front;
}


CudaLatticeDecoder::CudaLatticeDecoder(const CudaFst &fst,
                                       const CudaLatticeDecoderConfig &config): config_(config), fst_(fst),
  bytes_cuda_malloc(0), bytes_cuda_malloc_managed(0) {
  printf("CudaLatticeDecoder Constructor\n");
  int32 device;
  cudaGetDevice(&device);
  cudaCheckError();

  if (config_.verbose > 4) cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1e7);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  total_threads = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount *
                  config.gpu_fraction;

  token_allocator_.Initialize(config.max_tokens);
  cudaCheckError();
  bytes_cuda_malloc_managed += token_allocator_.GetCudaMallocManagedBytes();

  cudaEventCreateWithFlags(&event_pt, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&event_ll, cudaEventDisableTiming);

  cudaStreamCreateWithFlags(&stream_comp, cudaStreamNonBlocking);
  for (int32 i = 0; i < LAT_BUF_SIZE; i++)
    cudaStreamCreateWithFlags(&stream_lat[i], cudaStreamNonBlocking);
  cudaStreamCreateWithPriority(&stream_ll, cudaStreamNonBlocking, -1);

  cudaMalloc(&pe_idx_d, sizeof(int32)); bytes_cuda_malloc += sizeof(int32);
  cudaMalloc(&agg_idx_d, sizeof(int32)); bytes_cuda_malloc += sizeof(int32);
  cudaMalloc(&ne_idx_d, sizeof(int32)); bytes_cuda_malloc += sizeof(int32);
  cudaMalloc(&ne_queue_d, sizeof(int32)*config.max_tokens_per_frame);
  bytes_cuda_malloc += sizeof(int32);
  cudaMalloc(&fb_idx_d, sizeof(int32)); bytes_cuda_malloc += sizeof(int32);
  cudaMalloc(&barrier_d, sizeof(int32)); bytes_cuda_malloc += sizeof(int32);

  cudaMemset(pe_idx_d, 0, sizeof(int32));
  cudaMemset(ne_idx_d, 0, sizeof(int32));
  cudaMemset(agg_idx_d, 0, sizeof(int32));
  cudaMemset(fb_idx_d, 0, sizeof(int32));
  cudaMemset(barrier_d, 0, sizeof(int32));
  cudaCheckError();

  cudaMalloc(&cutoff_d, sizeof(CostType)); bytes_cuda_malloc += sizeof(CostType);
  cudaMalloc(&modified_d, sizeof(int32) * 2);
  bytes_cuda_malloc += sizeof(int32) * 2;


  cudaMalloc((void**)&current_tokens_lookup_d,
             sizeof(TokenLookupElem)*fst_.numStates);
  bytes_cuda_malloc += sizeof(TokenLookupElem) * fst_.numStates;

  cudaMallocHost(&loglikelihoods_h, sizeof(BaseFloat) * (fst_.max_ilabel + 1));
  cudaMallocHost(&loglikelihoods_old_h, sizeof(BaseFloat) * (fst_.max_ilabel + 1));

  cudaMalloc((void**)&loglikelihoods_d, sizeof(BaseFloat) * (fst_.max_ilabel + 1));
  bytes_cuda_malloc += sizeof(BaseFloat) * (fst_.max_ilabel + 1);
  cudaMalloc((void**)&loglikelihoods_old_d,
             sizeof(BaseFloat) * (fst_.max_ilabel + 1));
  bytes_cuda_malloc += sizeof(BaseFloat) * (fst_.max_ilabel + 1);

  //for pruning
  bytes_cuda_malloc += lattice_pruner_.Allocate(config.max_tokens_per_frame,
                       config.max_lat_arc_per_frame, config.prune_interval,
                       config.max_tokens, config.max_arcs);

  lat_arcs_buf_.Allocate(
    config.max_arcs,
    NULL,
    NULL,
    NULL,
    lattice_pruner_.GetDeviceArcsBpr());
  bytes_cuda_malloc += lat_arcs_buf_.GetCudaMallocBytes();

  for (int32 j = 0; j < LAT_BUF_SIZE; j++) {
    lat_toks_bufs_[j].Allocate(config.max_tokens_per_frame);
    bytes_cuda_malloc += lat_toks_bufs_[j].GetCudaMallocBytes();
  }

  cudaMalloc((void**)&token_per_arc_d, sizeof(Token)*fst.NumArcs()); //temp solution
  cudaMalloc((void**)&token_per_arc_update_d,
             sizeof(int32)*fst.NumArcs()); //temp solution
  cudaMemset(token_per_arc_update_d, 0,
             sizeof(int32)*fst.NumArcs()); //temp solution
  bytes_cuda_malloc += (sizeof(Token) + sizeof(int32)) * (fst.NumArcs());

  num_frames_decoded_ = 0;
  UpdateTokPointersByFrame(num_frames_decoded_);

  cudaStreamSynchronize(stream_comp);
  cudaStreamSynchronize(stream_lat[0]);
  cudaStreamSynchronize(cudaStreamPerThread);
  //sgemm requires shared memory and we don't want cache config changing.  So set a device wide cache config.
  cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
}

CudaLatticeDecoder::~CudaLatticeDecoder() {

  printf("CUDA LatticeDecoder DESTRUCTOR\n");

  for (int32 j = 0; j < LAT_BUF_SIZE; j++) {
    lat_toks_bufs_[j].Free();
  }
  lat_arcs_buf_.Free(true);

  lattice_pruner_.Free();

  token_allocator_.Finalize();

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
  for (int32 i = 0; i < LAT_BUF_SIZE; i++)
    cudaStreamDestroy(stream_lat[i]);
  cudaStreamDestroy(stream_ll);

}
DEVICE inline Token* FindOrAddTokenArc(processTokens_params& params,
                                       StateId nextstate, CostType total_cost, CostType acoustic_cost,
                                       TokenState* ts, uint32 j, bool add_arc, TokenState** next_ts,
                                       uint64 **token_pack, int* update) {
  //TokenLookupElem lookup_elem;
  //cuda_load16(&lookup_elem, &params.current_tokens_lookup[nextstate]);
  TokenLookupElem& lookup_elem = params.current_tokens_lookup[nextstate];
  Token *cur_tok = lookup_elem.token;
  //check if token is active or not.  Double check the lock.
  if (lookup_elem.active == 0
      && atomicCAS(&lookup_elem.active, 0,
                   1) == 0) {        //grab sentinal to see who gets to add to cur_toks list
    //if havent seen, add into hash
    *update = 1;
    lookup_elem.tokenstate_idx = params.cur_toks.PushBack(TokenState(cur_tok,
                                 nextstate, total_cost),
                                 &lookup_elem.token_pack);
  }
  //need both 2 steps below, to ensure tokenstate_idx won't run into error
  while (lookup_elem.tokenstate_idx == -1);//hasnt pushed
  __threadfence();
  *next_ts = &params.cur_toks[lookup_elem.tokenstate_idx];
  if (add_arc) {
    Token *prev_tok = ts->token;
    int32 ts_id = prev_tok->frame == params.frame ?
                  params.cur_toks.GetIdxFromAddr(ts) :
                  params.prev_toks.GetIdxFromAddr(ts);
    LatLink arc = LatLink(ts_id, prev_tok->frame,
                          lookup_elem.tokenstate_idx, params.frame,
                          params.arc_ilabels[j], params.arc_olabels[j],
                          params.arc_weights[j], acoustic_cost); //duplicate arcs in NE
    int32_t lat_arc_idx = params.lat_arcs_sub_vec.PushBack(arc);
  }
  *token_pack = &lookup_elem.token_pack;
  return cur_tok;
}
__global__ void addOneToken(processTokens_params params, StateId state) {
  TokenState *next_ts = NULL;
  uint64* token_pack;
  int32 j = 0;
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  Token* cur_tok = FindOrAddTokenArc(params, state, 0, //add first token
                                     0, NULL, j, false,  &next_ts,
                                     &token_pack, params.token_per_arc_update + j);
  uint64 new_token_pack = pack_cost_idx_into_uint64(0, j);
  Token* cur_te = params.token_per_arc + j;
  params.token_per_arc_update[j] = 1;
  cuda_store16(cur_te, &(Token(0, params.frame, NULL)));
  atomicMax((unsigned long long *)token_pack, (unsigned long long)new_token_pack);
  params.cur_toks.StoreDataByPackIdx(params.token_per_arc,
                                       params.token_per_arc_update, params.numArcs);
}

namespace CudaLatticeDecoder_kernel {
//putting this into a kernel to avoid extra latency of a memory copy
__global__ void initializeCutoff(CostType *cutoff) {
  *cutoff = INFINITY;
}
}

void CudaLatticeDecoder::InitDecoding() {
  if (config_.verbose > 1 ) printf("CUDA LatticeDecoder InitDecoding\n");
  num_frames_decoded_ = 0;
  for (int32 i = 0; i < LAT_BUF_SIZE; i++) {
    ClearToks(lat_toks_bufs_[i]);
  }
  lat_arcs_buf_.Clear();

  UpdateTokPointersByFrame(num_frames_decoded_);
  lattice_pruner_.Initialize();

  token_allocator_.Reset();
  int32 threads = 64;
  int32 blocks = DIV_ROUND_UP(total_threads, threads);

  //start moving these / allocating them on the device
  token_allocator_.PrefetchNextToDevice(stream_comp, fst_.numStates + 5000);

  allocateAllTokens <<< blocks, threads, 0, stream_comp>>>(current_tokens_lookup_d,
      fst_.numStates, token_allocator_, barrier_d);

  // initialize decoding:
  StateId start_state = fst_.Start();
  KALDI_ASSERT(start_state != fst::kNoStateId);

  cudaCheckError();
  //Token tok(StdWeight::One().Value(), NULL, 0);
  //Token tok(StdWeight::One().Value(),0, NULL, 0);
  processTokens_params params;
  InitParams(params);
  addOneToken <<< 1, 1, 0, stream_comp>>>(params, start_state);

  cudaCheckError();

  CudaLatticeDecoder_kernel::initializeCutoff <<< 1, 1, 0, stream_comp>>>(cutoff_d);
  ProcessNonemitting();
  if (config_.verbose > 1 ) printf("end of CUDA LatticeDecoder InitDecoding\n");
}

void CudaLatticeDecoder::FinalProcessLattice(
  Token** toks_buf, int** toks_fr_sidx, LatLink** arcs_buf, int** arcs_fr_size,
  TokenMergeVector** toks_vec_last_fr) {
  cudaStreamSynchronize(stream_comp); // after fini comp. we can start copy
  // copy unpruned toks to host
  lattice_pruner_.CopyToksToHost(num_frames_decoded_, stream_lat[0]);
  PruneActiveTokens(stream_comp, stream_comp,
                    config_.lat_fraction); // GPU lattice pruning
  // copy the TokenState vector in the last frame, used by ComputeFinalCosts()
  (lat_toks_bufs_[num_frames_decoded_ % LAT_BUF_SIZE]).CopyDataToHost(
    stream_lat[1]);
  *toks_vec_last_fr = &(lat_toks_bufs_[num_frames_decoded_ % LAT_BUF_SIZE]);
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
  for (int32 i = 0; i < cur_toks_->Size(); i++) {
    TokenState ts = (*cur_toks_)[i];

    if (ts.token->cost_ != std::numeric_limits<BaseFloat>::infinity() &&
        fst_.Final(ts.state) != StdWeight::Zero())
      return true;
  }

  return false;
}

// Outputs an FST corresponding to the single best path
// through the lattice.
bool CudaLatticeDecoder::GetBestPath(Lattice *fst_out,
                                     bool use_final_probs) const {
  KALDI_ERR << "we don't have this implementation in lattice decoder";
  return false;
}

void CudaLatticeDecoder::ComputeLogLikelihoods(DecodableInterface *decodable) {
  PUSH_RANGE("ComputeLogLikelihoods", 3)
  int32 frame = num_frames_decoded_;
  //cudaStreamSynchronize(stream_comp); //finish decoding this frame
  std::swap(loglikelihoods_h,
            loglikelihoods_old_h); //double buffering so we don't overwrite loglikelihoods_h before it is copied down
  std::swap(loglikelihoods_d, loglikelihoods_old_d); //double buffer

  //We really only need about 10% of these but finding out which 10% is more expensive then just computing all of them
  //Computing them inline in the next loop leads to lots of redundant computation
  decodable->ComputeLogLikelihoods(loglikelihoods_h, frame, fst_.max_ilabel + 1);

  //copying in another stream to overlap transfer with compute
  cudaMemcpyAsync(loglikelihoods_d, loglikelihoods_h,
                  sizeof(BaseFloat) * (fst_.max_ilabel + 1), cudaMemcpyHostToDevice, stream_ll);

  cudaEventRecord(event_ll,
                  stream_ll); //mark log likelihoods are copied down to the device
  //cudaStreamWaitEvent(stream_comp,event_ll,0); //ensure logliklihoods_d is updated before consuming; wait in ProcessTokens

  POP_RANGE
}

//structs to hold kernel parameters.  Large numbers of parameters can slow down launch latency which matters when we are launching very short kernels

//blockDim.x threads per token
template<int32 blockDimx, int32 blockDimy>
inline DEVICE void findBestCutoff_function(processTokens_params params) {
  auto group = cooperative_groups::tiled_partition<blockDimx>
               (cooperative_groups::this_thread_block());

  CostType local_cutoff = INFINITY;
  int32 size = params.prev_toks.Size();

  //uses dynamically load balanced loop trips.  Tokens are assigned dynamically instead of statically
  while (true) {
    int32 i;
    if (group.thread_rank() == 0) { //thread 0 nominated to get new token
      i = atomicAdd(params.fb_idx, 1);   //get token index
    }
    i = group.shfl(i, 0);        //broadcast token index
    //i=__shfl_sync(0xffffffff,i,0);
    if (i >= size) break; //Work complete

    TokenState ts = params.prev_toks[i];
    Token * tok = ts.token;
    StateId state = ts.state;

    uint32 start = params.e_offsets[state], finish = params.e_offsets[state + 1];

    int32 ilabel, ilabel_next;

    int32 j = start + group.thread_rank();

    if (j < finish) {
      ilabel_next = params.arc_ilabels[j];
    }
    int32 nextj;

    for (j; j < finish; j = nextj) {
      nextj = j + blockDimx;
      ilabel = ilabel_next;
      if (nextj < finish) {
        ilabel_next = params.arc_ilabels[nextj];
      }

      BaseFloat acoustic_cost =
        -params.loglikelihoods[ilabel]; //TODO can I prefetch this?
      CostType weight = params.arc_weights[j];

      CostType total_cost = tok->cost_ + weight + acoustic_cost + params.beam;

      if (total_cost < local_cutoff)
        local_cutoff = total_cost;
    }
  }

  //TODO reduce inside block first?
  if (local_cutoff != INFINITY) {
    atomic_min(params.cutoff, local_cutoff);
  }
}


//blockDim.x threads per token
template<int32 blockDimx, int32 blockDimy>
inline DEVICE void processEmittingTokens_function(processTokens_params& params) {
  auto group = cooperative_groups::tiled_partition<blockDimx>
               (cooperative_groups::this_thread_block());

  CostType cutoff = *params.cutoff;
  int32 size = params.prev_toks.Size();
  //uses dynamically load balanced loop trips.  Tokens are assigned dynamically instead of statically
  while (true) {
    int32 i;
    if (group.thread_rank() == 0) { //thread 0 nominated to get new token
      i = atomicAdd(params.pe_idx, 1);   //get token index
    }
    i = group.shfl(i, 0);        //broadcast token index
    //i=__shfl_sync(0xffffffff,i,0);
    if (i >= size) break;

    TokenState& ts = params.prev_toks[i];
    Token * tok = ts.token;
    StateId state = ts.state;

    uint32 start = params.e_offsets[state], finish = params.e_offsets[state + 1];
    int32 ilabel, ilabel_next;  //prefetch ilabel since it leads to a dependent load

    int32 j = start + group.thread_rank();

    if (j < finish) {
      ilabel_next = params.arc_ilabels[j];
    }
    int32 nextj;

    for (j; j < finish; j = nextj) {
      nextj = j + blockDimx;

      ilabel = ilabel_next;

      if (nextj < finish) {
        ilabel_next =
          params.arc_ilabels[nextj];//prefetch ilabel since it leads to a dependent load
      }
      BaseFloat acoustic_cost =
        -params.loglikelihoods[ilabel];  //TODO can I prefetch this?
      BaseFloat weight = params.arc_weights[j];
      StateId nextstate = params.arc_nextstates[j];

      CostType total_cost = tok->cost_ + weight + acoustic_cost;

      if (total_cost <= cutoff) {
        uint64* token_pack;
        TokenState *next_ts = NULL;
        //get cur_tok&token_pack addr
        Token *cur_tok = FindOrAddTokenArc(params, nextstate, total_cost,
                                           acoustic_cost, &ts, j, true,
                                           &next_ts, &token_pack, params.token_per_arc_update + j);
        //get cur_te&new_token_pack
        uint64 new_token_pack = pack_cost_idx_into_uint64(-total_cost, j);
        uint64 ret = atomicMax((unsigned long long *)token_pack,
                                 (unsigned long long)new_token_pack);
        if (ret < new_token_pack) {
          Token* cur_te = params.token_per_arc + j;
          cuda_store16(cur_te, &(Token(acoustic_cost + weight, params.frame, tok)));
          params.token_per_arc_update[j] = 1;
        }
      } //end total_cost<=cutoff
    } //end arc loop
  } //end token loop
  __grid_sync_nv_internal(params.barrier);
  params.cur_toks.StoreDataByPackIdx(params.token_per_arc,
                                       params.token_per_arc_update, params.numArcs);
}

template<int32 blockDimx, int32 blockDimy>
DEVICE __inline__ void processNonEmittingTokens_function(processTokens_params
    &params, CostType cutoff, uint32 size,  volatile int32 *modified,
    bool aggregate = false) {
  assert(size);
  auto group = cooperative_groups::tiled_partition<blockDimx>
               (cooperative_groups::this_thread_block());
  int* agg_tok_idx = params.agg_idx; // need to make it 0 before enter this func
  int* cur_tok_idx = params.ne_idx; // need to make it 0 before enter this func
  int32 tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (aggregate) {
    for (tid; tid < size; tid += blockDim.x * gridDim.x) {
      if (params.cur_toks.IsUpdated(tid)) {
        int32 i = atomicAdd(agg_tok_idx,
                            1);   //get changed token index for faster NE proc
        if (i >= size) break;
        params.ne_queue[i] = tid;
      }
    }
    __grid_sync_nv_internal(params.barrier); // if we use more modified, we can reduce more grid sync, but will make the program complexer
  }
  if (params.verbose > 3 && threadIdx.x == 0
      && blockIdx.x == 0) CUDA_PRINTF("PNE: %i %i %i\n", params.frame,
                                        params.cur_toks.Size(), *agg_tok_idx);
  //uses dynamically load balanced loop trips.  Tokens are assigned dynamically instead of statically
  while (true) {
    int32 i, j;
    if (group.thread_rank() == 0) { //thread 0 nominated to get new token
      if (aggregate) {
        j = atomicAdd(cur_tok_idx, 1);   //get token index
        if (j >= *agg_tok_idx) i = size; // to exit
        else i = params.ne_queue[j];
      } else {
        i = atomicAdd(cur_tok_idx, 1);   //get token index
      }
    }
    i = group.shfl(i, 0);        //broadcast token index
    if (i >= size) break; //TODO: optimize this to reduce iteration

    TokenState& ts = params.cur_toks[i];
    Token * tok = ts.token;
    StateId state = ts.state;
    assert(params.ne_offsets);
    uint32 start = params.ne_offsets[state], finish = params.ne_offsets[state + 1];
    for (int32 j = start + group.thread_rank(); j < finish; j += blockDimx) {
      BaseFloat weight = params.arc_weights[j];
      StateId nextstate = params.arc_nextstates[j];

      Token next_tok = Token(weight, params.frame, tok);

      CostType total_cost = tok->cost_ + weight;

      if (params.verbose > 4) CUDA_PRINTF("D: %i %i %i %i %i \n", threadIdx.x,
                                            threadIdx.y, j, blockIdx.x, i);
      if (next_tok.cost_ <= cutoff) {
        TokenState *next_ts = NULL;
        uint64* token_pack;
        Token *cur_tok = FindOrAddTokenArc(params, nextstate, total_cost,
                                           0, &ts, j, true, &next_ts,
                                           &token_pack, params.token_per_arc_update + j);

        uint64 new_token_pack = pack_cost_idx_into_uint64(-total_cost, j);
        uint64 ret = atomicMax((unsigned long long *)token_pack,
                                 (unsigned long long)new_token_pack);
        if (ret < new_token_pack) {
          Token* cur_te = params.token_per_arc + j;
          cuda_store16(cur_te, &(Token(weight, params.frame, tok)));
          params.token_per_arc_update[j] = 1;
          (*modified) = true;
        }
      }
    }

  }
  __grid_sync_nv_internal(params.barrier);
  params.cur_toks.StoreDataByPackIdx(params.token_per_arc,
                                       params.token_per_arc_update, params.numArcs);
}

__launch_bounds__(64, 64)
__global__ void processTokens_cg(processTokens_params params,
                                 bool is_init = false) {
  bool rank0 = blockIdx.x == 0 && threadIdx.x == 0;

  if (!is_init) {
    findBestCutoff_function<32, 2>(params);
    __grid_sync_nv_internal(params.barrier);
  }

  volatile int32 *modified0 =
    params.modified;    //modified flag for current iteration
  volatile int32 *modified1 = params.modified +
                              1; //modified flag for next/last iteration
  *modified1 = false;
  CostType cutoff = *params.cutoff;

  if (!is_init) {
    processEmittingTokens_function<32, 2>(params);
    __grid_sync_nv_internal(params.barrier);  //ensure cur_toks size is final
  }

  int32 tok_E;
  int32 itv = params.verbose > 2 ? 1 : 10;
  if (rank0 && params.verbose > 1 && params.frame % itv == 0)
    tok_E = params.cur_toks.Size();

  int32 cnt = 0;
  uint32 size = 0;
  do {
    size = params.cur_toks.Size();
    if (rank0) {
      *params.ne_idx =
        0;  // need to make it 0 before enter processNonEmittingTokens_function
      *params.agg_idx =
        0;  // need to make it 0 before enter processNonEmittingTokens_function
    }
    __grid_sync_nv_internal(
      params.barrier); //wait for everyone to read size and modified0

    //swap buffers
    cuda_swap(modified0,
             modified1); //double buffered to avoid extra sync when resetting modified to false, 3% speedup
    *modified1 = false;
    cnt++;
    bool aggregate = (!is_init) && cnt > 1 ? 1 : 0;

    processNonEmittingTokens_function<32, 2>(params, cutoff, size, modified0,
        aggregate);

    // we have sync in the end of processNonEmittingTokens_function
    // __grid_sync_nv_internal(params.barrier);  //wait for everyone to finish process tokens and writes modified0
  } while ((*modified0) == true);

  if (rank0 && params.verbose > 1 && params.frame % itv == 0)
    CUDA_PRINTF("TK: %i %i %i %f\n", params.frame, tok_E, params.cur_toks.Size(),
                cutoff);


  //proc lattice before allocate new toks to TokenState
  params.lattice_pruner.CollectToksPerFrame(params.cur_toks, params.frame);
  params.lattice_pruner.CollectArcsPerFrame(params.lat_arcs_sub_vec,
      params.frame);


  __grid_sync_nv_internal(params.barrier);

  allocateNewTokens_function(params.current_tokens_lookup, params.cur_toks,
                             params.token_allocator);

  if (rank0) {
    //prepare for next iteration
    //params.prev_toks.Clear(); //change to be done in PreProcessTokens
    *params.cutoff = INFINITY;
    *params.fb_idx = 0;
    *params.pe_idx = 0;
  }
  __grid_sync_nv_internal(params.barrier);  //wait for allocation to finish

  if (rank0) {
    params.token_allocator.AdvanceFront(params.cur_toks.Size());
  }
}

void CudaLatticeDecoder::InitParams(processTokens_params& params) {
  params.prev_toks = (*prev_toks_);
  params.cur_toks = (*cur_toks_);
  params.current_tokens_lookup = current_tokens_lookup_d;
  params.cutoff = cutoff_d;
  params.lat_arcs_sub_vec = lat_arcs_buf_;
  params.token_per_arc = token_per_arc_d;
  params.token_per_arc_update = token_per_arc_update_d;

  params.token_allocator = token_allocator_;
  params.lattice_pruner = lattice_pruner_;

  params.e_offsets = fst_.e_offsets_d;
  params.ne_offsets = fst_.ne_offsets_d;
  params.arc_ilabels = fst_.arc_ilabels_d;
  params.arc_olabels = fst_.arc_olabels_d;
  params.arc_weights = fst_.arc_weights_d;
  params.arc_nextstates = fst_.arc_nextstates_d;

  params.loglikelihoods = loglikelihoods_d;
  params.modified = modified_d;
  params.pe_idx = pe_idx_d;
  params.ne_idx = ne_idx_d;
  params.ne_queue = ne_queue_d;
  params.fb_idx = fb_idx_d;
  params.agg_idx = agg_idx_d;
  params.barrier = barrier_d;

  params.beam = config_.beam;
  params.verbose = config_.verbose;
  params.lattice_beam = config_.lattice_beam;
  params.prune_interval = config_.prune_interval;
  params.numArcs = fst_.NumArcs();
  params.frame = num_frames_decoded_;
}
void CudaLatticeDecoder::ProcessNonemitting() {
  nvtxRangePushA("ProcessNonemitting");

  dim3 threads(64, 1);

  dim3 blocks(DIV_ROUND_UP(total_threads, (threads.x * threads.y)));

  processTokens_params params;
  InitParams(params);
  processTokens_cg <<< blocks, threads, 0, stream_comp>>>(params, true);

  cudaCheckError();
  nvtxRangePop();
}
void CudaLatticeDecoder::UpdateTokPointersByFrame(uint32 frame) {
  cur_toks_ = &lat_toks_bufs_[frame % LAT_BUF_SIZE];
  prev_toks_ = &lat_toks_bufs_[(frame - 1) % LAT_BUF_SIZE];
  //single buffer in lat_arcs_buf_;
}

__launch_bounds__(64, 64)
__global__ void cuda_decoder_lat_prune_active_tokens(processTokens_params
    params) {
//    auto grid = cooperative_groups::this_grid();
  params.lattice_pruner.PruneActiveTokens(params.frame, params.lattice_beam,
                                          params.verbose
                                         );
}

void CudaLatticeDecoder::PruneActiveTokens(cudaStream_t wait_st,
    cudaStream_t run_st, BaseFloat gpu_ratio) {
  processTokens_params params;
  InitParams(params);
  dim3 threads(64, 1);
  dim3 blocks(DIV_ROUND_UP(total_threads * gpu_ratio, (threads.x * threads.y)));
  cudaStreamSynchronize(wait_st);
  if (params.verbose > 1) KALDI_LOG << "PruneActiveTokens, # of blocks: " <<
                                      blocks.x << std::endl;
  cuda_decoder_lat_prune_active_tokens <<< blocks, threads, 0, run_st>>>(params);
  //lattice_pruner_.CopyArcsToHost(num_frames_decoded_, st);
}
void CudaLatticeDecoder::PreProcessTokens() {
  PUSH_RANGE("PreProcessTokens", 1)
  num_frames_decoded_++;
#ifndef MEMADVISE
  //no need to prefetch if we have done a memadvise
  token_allocator_.PrefetchNextToDevice(cudaStreamPerThread);
#endif
  //TODO prefetch here
  UpdateTokPointersByFrame(num_frames_decoded_);
  ClearToks(*cur_toks_);
  //dont need to clear arcs as we directly take the final buffer into this vector

  POP_RANGE
}

void CudaLatticeDecoder::ProcessTokens() {
  PUSH_RANGE("ProcessTokens", 2)
  KALDI_VLOG(4) << num_frames_decoded_ << std::endl;

  processTokens_params params;
  dim3 threads(64, 1);
  dim3 blocks(DIV_ROUND_UP(total_threads, (threads.x * threads.y)));

  InitParams(params);

  if (params.frame == 1) KALDI_VLOG(2) << "# of blocks: " << blocks.x << std::endl;

  cudaStreamWaitEvent(stream_comp, event_ll,
                      0); //make sure log likelihoods are on the device before starting these kernels

  processTokens_cg <<< blocks, threads, 0, stream_comp>>>(params); //doesn't work
  cudaCheckError();

  cudaEventSynchronize(event_pt); //throttle
  cudaEventRecord(event_pt, stream_comp);



  POP_RANGE
}

void CudaLatticeDecoder::ClearToks(TokenMergeVector &toks) {
  //cannot actually delete tokens as they may still be connected to active tokens
  toks.Clear(stream_comp);
}
} // end namespace kaldi.
