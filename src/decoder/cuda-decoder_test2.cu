#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include "base/timer.h"

#ifdef __CUDACC__
  #define HOST __host__
  #define DEVICE __device__

#else
  #define HOST
  #define DEVICE
#endif

#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }


DEVICE void acquire_semaphore(volatile int *lock){
  while (atomicCAS((int *)lock, 0, 1) != 0);
  }

DEVICE void release_semaphore(volatile int *lock){
  //*lock = 0;
  atomicExch((unsigned int*)lock,0u);
  __threadfence();
  }

  template<int blockDimx, int blockDimy>
  inline DEVICE void myadd(int *ret, volatile int *mutex) {
    (*(ret+threadIdx.x))++;
    (*(mutex+threadIdx.x*blockIdx.x))++;
  }

  template<int blockDimx, int blockDimy>
  inline DEVICE void myadd2(int *ret, volatile int *mutex) {
    if (threadIdx.x==0) {
    acquire_semaphore((int*)(mutex+threadIdx.x*blockIdx.x));
    (*(ret+threadIdx.x))++;
    release_semaphore((int*)(mutex+threadIdx.x*blockIdx.x));
    }
  }
  template<int blockDimx, int blockDimy>
  inline DEVICE void myadd0(int *ret, volatile int *mutex) {
    acquire_semaphore((int*)(mutex));
    (*(ret))++;
    release_semaphore((int*)(mutex));
  }

  __global__ void callmyadd(int *ret, int *mutex) {
  //myadd2<32,2>(ret, mutex);
  myadd<100,1>(ret, mutex);
  //myadd<32,2>(ret, mutex);
  }
int main() {
  //int blocks=200;
  int blocks=3;
  //int blocks=7;
  int *mutex=0;
  int *ret=0, ret_h=0;
  int n =1e8;
  int *v_man;
  int32_t device;
  kaldi::Timer timer;
  double t1,t2,t0,t3,t2_1,t0_1;

  cudaGetDevice(&device);
  cudaMallocManaged((void**)&v_man,sizeof(int)*n);  
  cudaMallocManaged((void**)&ret,sizeof(int)*n);  
  cudaMemset(v_man, 0,sizeof(int)*n);
  cudaMemset(ret, 0,sizeof(int)*n);
  cudaMemAdvise(v_man,sizeof(int)*n,cudaMemAdviseSetPreferredLocation,device);
  cudaMemAdvise(ret,sizeof(int)*n,cudaMemAdviseSetPreferredLocation,device);
  cudaMemPrefetchAsync(v_man,sizeof(int)*n,device);  //force pages to allocate now

  callmyadd<<<100,320>>>(ret, v_man);
  cudaCheckError();

  //time
  int s0=0;
  timer.Reset();
  for (int i=0;i<n;i++)  s0+=v_man[i];
  t0=timer.Elapsed();
  timer.Reset();
  for (int i=0;i<n;i++)  v_man[i];
  t0_1=timer.Elapsed();

  /*
  //time
  timer.Reset();
  for (int i=0;i<n;i++) int k=v_man[i];
  //time
  timer.Reset();
  for (int i=0;i<n;i++) int k=v_man[i];
 */
  cudaMemPrefetchAsync(v_man, sizeof(int)* n,cudaCpuDeviceId,NULL);  

   //time
  timer.Reset();
  for (int i=0;i<n;i++) int k=v_man[i];
  t1=timer.Elapsed();
 
  callmyadd<<<100,320>>>(ret, v_man);
  cudaCheckError();

  //time
  timer.Reset();
  for (int i=0;i<n;i++) int k=v_man[i];
  t2=timer.Elapsed();

  callmyadd<<<100,320>>>(ret, v_man);
  cudaCheckError();

  //time
  timer.Reset();
  for (int i=0;i<n;i++) int k=v_man[i];
  t2_1=timer.Elapsed();

  callmyadd<<<100,320>>>(ret, v_man);
  cudaCheckError();

  cudaMemPrefetchAsync(v_man, sizeof(int)* n,cudaCpuDeviceId,NULL); 

  //time
  int s=0;
  timer.Reset();
  for (int i=0;i<n;i++)  s+=v_man[i];
  t3=timer.Elapsed();


  std::cout << " nop "<<t0<< " re "<<t0_1<<" p "<<t1<<" mod "<<t2 <<" re "<<t2_1 <<" pf "<<t3<< " "<<ret[0] <<" "<<v_man[0]<<" "<<s0<<" "<<s<<std::endl;
    
  cudaFree(v_man);
}
