// fstext/determinize-star-test.cc

// Copyright 2009-2011  Microsoft Corporation
//           2015       Hainan Xu

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

#include "base/kaldi-math.h"
#include "fstext/fst-test-utils.h"
#include "fstext/kaldi-fst-io.h"
#include "base/timer.h"

#include <sys/mman.h> //for MAP_SHARED
#include <sys/resource.h> // get_mem

#include "ThreadPool.h"
#include "time.h"

namespace fst
{
using namespace kaldi;
using namespace std;

double get_mem() {
  struct rusage ru;
  getrusage(RUSAGE_SELF,&ru);
  return ru.ru_maxrss*1.0/1024;
}
double get_mem2() {
  char buf[256];
  FILE *f;
  int ret=-1;
  /*
  //116551 98709 520 653 0 114831 0
   * 98033 54203 664 808 0 90780 0
                  size       total program size
                             (same as VmSize in /proc/[pid]/status)
                  resident   resident set size
                             (same as VmRSS in /proc/[pid]/status)
                  share      shared pages (from shared mappings)
                  text       text (code)
                  lib        library (unused in Linux 2.6)
                  data       data + stack
                  dt         dirty pages (unused in Linux 2.6)
   */
  sprintf(buf,"/proc/%d/statm",getpid());
  f=fopen(buf,"r");
  if(!f){goto end;}
  int size, resident, share, text, lib, data, dt;
  ret=fscanf(f,"%d %d %d %d %d %d %d",&(size),&(resident),&(share),&(text),&(lib),&(data),&(dt));
  if(ret!=7){ret=-1;goto end;}
  ret=0;
end:
  if(f)
  {
    fclose(f);
  }
  return resident *4.0/1024;
}

template <typename Arc>
size_t LoadFstState(const Fst<Arc>& fst, size_t s, bool change_seq = false) {
  size_t narcs = 0;
  for (ArcIterator<Fst<Arc>> aiter(fst, s); !aiter.Done(); aiter.Next()) {
    const auto &arc = aiter.Value();
    Arc arc_new;
    arc_new.nextstate = arc.nextstate;
    arc_new.weight = arc.weight;
    arc_new.ilabel = arc.ilabel;
    arc_new.olabel = arc.olabel;
    narcs++;
    if (change_seq) {
      for (ArcIterator<Fst<Arc>> aiter(fst, arc_new.nextstate); !aiter.Done(); aiter.Next()) {
        const auto &arc = aiter.Value();
        Arc arc_new;
        arc_new.nextstate = arc.nextstate;
        arc_new.weight = arc.weight;
        arc_new.ilabel = arc.ilabel;
        arc_new.olabel = arc.olabel;
      }
    }
  }
  return narcs;
}
#if 0
template <typename FST>
size_t LoadFst(const FST& fst, bool change_seq = false) {
  typedef typename FST::Arc Arc;
  KALDI_LOG << "main_load "<<time(NULL);
  size_t narcs = 0;
  for (StateIterator<FST> siter(fst); !siter.Done(); siter.Next()) {
    const auto s = siter.Value();
    narcs += LoadFstState(fst, s, change_seq);
  }
  for (int s=0; s<fst.NumStates(); s++) {
    narcs += LoadFstState(fst, s, change_seq);
  }
  return narcs;
}
#else
template <typename Arc>
size_t LoadFst(const Fst<Arc>& fst, bool change_seq = false) {
  size_t narcs = 0;
  KALDI_LOG << "main_load "<<time(NULL);
  for (StateIterator<Fst<Arc>> siter(fst); !siter.Done(); siter.Next()) {
    const auto s = siter.Value();
    for (ArcIterator<Fst<Arc>> aiter(fst, s); !aiter.Done(); aiter.Next()) {
      const auto &arc = aiter.Value();
      Arc arc_new;
      arc_new.nextstate = arc.nextstate;
      arc_new.weight = arc.weight;
      arc_new.ilabel = arc.ilabel;
      arc_new.olabel = arc.olabel;
      narcs++;
      if (change_seq) {
        for (ArcIterator<Fst<Arc>> aiter(fst, arc_new.nextstate); !aiter.Done(); aiter.Next()) {
          const auto &arc = aiter.Value();
          Arc arc_new;
          arc_new.nextstate = arc.nextstate;
          arc_new.weight = arc.weight;
          arc_new.ilabel = arc.ilabel;
          arc_new.olabel = arc.olabel;
        }
      }
    }
  }
  KALDI_LOG << "main_load end "<<time(NULL);
  return narcs;
}
#endif
void PreloadFst(string fst_in_str, size_t start, size_t end) {
  auto *decode_fst = dynamic_cast<ConstFst<StdArc>*>(fst::ReadFstKaldiGeneric(fst_in_str, true, "map", MAP_SHARED));
  KALDI_ASSERT(start>=0);
  KALDI_ASSERT(end <= decode_fst->NumStates());
  KALDI_LOG<<"loading_start ( "<<start<<" , "<<end<<" ) "<<time(NULL);
  size_t narcs = 0;
  for (size_t i = start; i < end; i++) {
    narcs += LoadFstState(*decode_fst, i);
  }
  KALDI_LOG<<"loading_end ( "<<start<<" , "<<end<<" ) "<<narcs<<" "<<time(NULL);
}
void ParaPreloadFst(string fst_in_str, size_t num_states, ThreadPool& pool, int nthreads) {
  int num_states_per_thread = num_states/nthreads+1;
  for(int i = nthreads-1; i < nthreads; ++i) { // TODO
    pool.enqueue(&PreloadFst, fst_in_str, i*num_states_per_thread, std::min(num_states, (size_t)(i+1)*num_states_per_thread));
  }
  return;
}
size_t TestLoadFstSub(string fst_in_str, bool change_seq=false, string map="", int mmap_flags=0, int nthreads=0) {
  Timer timer;
  auto m1 = get_mem2();
  auto *decode_fst = dynamic_cast<ConstFst<StdArc>*>(fst::ReadFstKaldiGeneric(fst_in_str, true, map, mmap_flags));
  auto m2 = get_mem2();
  if (nthreads) {
    ThreadPool pool(nthreads); // try nthreads/2 later
    ParaPreloadFst(fst_in_str, decode_fst->NumStates(), pool, nthreads);
  }
  auto t1 = timer.Elapsed();
  auto r1 = LoadFst(*decode_fst);
  auto t2 = timer.Elapsed();
  auto m3 = get_mem2();
  auto r2 = LoadFst(*decode_fst);
  auto t3 = timer.Elapsed();
  delete decode_fst;

  KALDI_ASSERT(r1==r2);
  KALDI_LOG << "file: "<< map<<mmap_flags<<change_seq <<" "<< fst_in_str << " time: "<< t1 <<" "<<  t2-t1 <<" "<<t3-t2<< " mem: "<<m1<<" "<<m2-m1<<" "<<m3-m2<< " "<<r1;
  return r1;
}


void TestLoadFst(string fst_in_str, bool change_seq = false, int nthreads=0) {
  // TODO: assert align
  FLAGS_v = 0;

  auto r2=TestLoadFstSub(fst_in_str, change_seq, "map", MAP_SHARED, nthreads); // test in this sequence will get the worst result for map + no MAP_POPULATE
  auto r3=TestLoadFstSub(fst_in_str, change_seq, "map", MAP_SHARED|MAP_POPULATE);
  auto r1=TestLoadFstSub(fst_in_str, change_seq, "", 0);

  KALDI_ASSERT(r1 == r2 && r3==r2);
}


} // end namespace fst


int main() {
  using namespace kaldi;
  using namespace fst;

  for (int i = 0;i < 1;i++) {  // We would need more iterations to check
    // this properly.
    //TestLoadFst("/home/resources/zhc00/egs/mini_librispeech/s5/data/lang_test_tgsmall/HCLG.fst.ali");
    //TestLoadFst("/home/resources/zhc00/asr_test/data_ai/lang_sf_prune.graph/HCLG.fst.const2.ali");
    //TestLoadFst("/home/resources/zhc00/asr_test/data_ai/zhc01/kaldi_sf/graph_decode_sf/HCLG.fst.const.ali", true);
    //
    //TestLoadFst("/home/resources/zhc00/egs/mini_librispeech/s5/data/lang_test_tgsmall/HCLG.fst.ali", true);
    //TestLoadFst("/home/resources/zhc00/asr_test/data_ai/lang_sf_prune.graph/HCLG.fst.const2.ali", true);
    
    TestLoadFst("/home/resources/zhc00/asr_test/data_ai/lang_sf_prune.graph/HCLG.fst4.const.ali", false, 3);
    
  }
}
