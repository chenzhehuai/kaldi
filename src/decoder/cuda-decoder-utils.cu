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

  DEVICE void load16(void *a, const void *b) {
    const ulong2 *src = reinterpret_cast<const ulong2*>(b);
    ulong2 &dst = *reinterpret_cast<ulong2*>(a);
    asm("ld.global.v2.u64 {%0,%1}, [%2];" : "=l"(dst.x), "=l"(dst.y) : "l"(src));
  }
  
  DEVICE void store16(void *a, const void *b) {
    const ulong2 src = *reinterpret_cast<const ulong2*>(b);
    asm("st.global.v2.u64 [%0], {%1,%2};" :: "l"(a), "l"(src.x), "l"(src.y));
  }

  
  DEVICE void store32(void *a, const void *b) {
    //const ulong4 src = *reinterpret_cast<const ulong4*>(b);
    //asm("st.global.v4.u64 [%0], {%1,%2,%3,%4};" :: "l"(a), "l"(src.x), "l"(src.y),
    //  "l"(src.z), "l"(src.w));
    memcpy(a, b, 32);
  }


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
DEVICE  void __grid_sync_nv_internal(int *barrier)
{
    __gpu_sync_fast((volatile int*)barrier);
}

DEVICE void atomicMin(double *address, double val) {
  unsigned long long *address_ull = (unsigned long long *)address;

  double minval = *address;

  while (val < minval) {  //if my value is less than minimum
    minval = val;         //update the minimum to my value locally
    val = __longlong_as_double(atomicExch(address_ull, __double_as_longlong(val))); //write minimum and read back value
  } //if the new value is < the minimum I wrote I need to try again.
}
DEVICE void atomicMin(float *address, float val) {
  unsigned int *address_ui = (unsigned int  *)address;

  float minval = *address;

  while (val < minval) {  //if my value is less than minimum
    minval = val;         //update the minimum to my value locally
    val = __uint_as_float(atomicExch(address_ui, __float_as_uint(val))); //write minimum and read back value
  } //if the new value is < the minimum I wrote I need to try again.
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

