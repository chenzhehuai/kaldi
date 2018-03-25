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

#include "cuda-decoder-utils.h"

namespace kaldi {

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
    inline void CudaVector<T>::free(bool create_outside) { 
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


/***************************************CudaFst Implementation*****************************************/
HOST DEVICE inline float CudaFst::Final(StateId state) const {
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
  numArcs=arc_count;

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

} // end namespace kaldi.

