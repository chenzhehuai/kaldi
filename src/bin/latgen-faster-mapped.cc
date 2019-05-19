// bin/latgen-faster-mapped.cc

// Copyright 2009-2012  Microsoft Corporation, Karel Vesely
//                2013  Johns Hopkins University (author: Daniel Povey)
//                2014  Guoguo Chen

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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "decoder/decodable-matrix.h"
#include "base/timer.h"

#include <sys/mman.h> //for MAP_SHARED

#include "../fstext/ThreadPool.h"
#include "time.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>

using namespace fst;

int is_mincore_page_resident(char p) {
  return p & 0x1;
}
int64_t bytes2pages(int64_t bytes, int64 pagesize) {
  return (bytes+pagesize-1) / pagesize;
}
float page_stat(string fst_in_str) {
  auto fd = open(fst_in_str.c_str(), O_RDONLY, 0);
  struct stat sb;
  auto res = fstat(fd, &sb);
  KALDI_ASSERT(!res);
  int64 len_of_file = sb.st_size;
  auto len_of_range = len_of_file;
  void* mem = mmap(NULL, len_of_range, PROT_READ, MAP_SHARED, fd, 0);
  int64 total_pages_in_core=0;
  auto pagesize = sysconf(_SC_PAGESIZE);
  auto pages_in_range = bytes2pages(len_of_range, pagesize);
  int64_t total_pages=pages_in_range;
  unsigned char*mincore_array = (unsigned char*)malloc(pages_in_range);
  if (mincore(mem, len_of_range, mincore_array)) KALDI_ERR << "";
  for (int i=0; i<pages_in_range; i++) {
    if (is_mincore_page_resident(mincore_array[i])) {
      total_pages_in_core++;
    }
  }
  free(mincore_array);
  if (munmap(mem, len_of_range)) KALDI_ERR<< "unable to munmap file";
  return 1.0f*total_pages_in_core/total_pages;
}


template <typename Arc>
size_t LoadFstState(const Fst<Arc>& fst, size_t s, bool change_seq = false) {
  size_t narcs = 0;
  typename Arc::Weight weight;
  for (ArcIterator<Fst<Arc>> aiter(fst, s); !aiter.Done(); aiter.Next()) {
    auto arc = aiter.Value();
    weight=Times(weight, arc.weight); // in case aiter.Value() is optimized out
    narcs++;
    if (change_seq) {
      KALDI_ASSERT(0);
    }
  }
  return narcs;
}
void PreloadFst2(ConstFst<StdArc> *decode_fst, size_t start, size_t end) {
  KALDI_ASSERT(start>=0);
  KALDI_ASSERT(end <= decode_fst->NumStates());
  KALDI_LOG<<"loading_start ( "<<start<<" , "<<end<<" ) "<<time(NULL);
  size_t narcs = 0;
  for (size_t i = start; i < end; i++) {
    narcs += LoadFstState(*decode_fst, i);
  }
  KALDI_LOG<<"loading_end ( "<<start<<" , "<<end<<" ) "<<narcs<<" "<<time(NULL);
}
void PreloadFstC0(string fst_in_str, size_t start, size_t end) {
  KALDI_LOG<<"loading_start ( "<<start<<" , "<<end<<" ) "<<time(NULL);
  auto *decode_fst = dynamic_cast<ConstFst<StdArc>*>(fst::ReadFstKaldiGeneric(fst_in_str, true, "map", MAP_SHARED));
  KALDI_ASSERT(start>=0);
  KALDI_ASSERT(end <= decode_fst->NumStates());
  size_t narcs = 0;
  for (size_t i = start; i < end; i++) {
    narcs += LoadFstState(*decode_fst, i);
  }
  delete decode_fst;
  KALDI_LOG << "page_stat: " << page_stat(fst_in_str);
  KALDI_LOG<<"loading_end ( "<<start<<" , "<<end<<" ) "<<narcs<<" "<<time(NULL);
}
void PreloadFstC_1(string fst_in_str, size_t start, size_t end) {
  KALDI_LOG<<"loading_start ( "<<start<<" , "<<end<<" ) "<<time(NULL);
  auto *decode_fst = dynamic_cast<ConstFst<StdArc>*>(fst::ReadFstKaldiGeneric(fst_in_str, true, "map", MAP_SHARED));
  KALDI_ASSERT(start>=0);
  KALDI_ASSERT(end <= decode_fst->NumStates());
  size_t narcs = 0;
  for (int64 i = end-1; i >= (int64)start; i--) {
    narcs += LoadFstState(*decode_fst, i);
  }
  delete decode_fst;
  KALDI_LOG << "page_stat: " << page_stat(fst_in_str);
  KALDI_LOG<<"loading_end ( "<<start<<" , "<<end<<" ) "<<narcs<<" "<<time(NULL);
}


void PreloadFstC1(string fst_in_str, size_t start, size_t end) {
  KALDI_LOG<<"loading_start ( "<<start<<" , "<<end<<" ) "<<time(NULL);
  auto *decode_fst = dynamic_cast<ConstFst<StdArc>*>(fst::ReadFstKaldiGeneric(fst_in_str, true, "map", MAP_SHARED|MAP_POPULATE));
  KALDI_ASSERT(decode_fst);
  delete decode_fst;
  KALDI_LOG << "page_stat: " << page_stat(fst_in_str);
  KALDI_LOG<<"loading_end ( "<<start<<" , "<<end<<" ) "<<" "<<time(NULL);
}
void PreloadFstC2(string fst_in_str, size_t start, size_t end) {
  KALDI_LOG<<"loading_start ( "<<start<<" , "<<end<<" ) "<<time(NULL);
  auto pagesize = sysconf(_SC_PAGESIZE);
  auto fd = open(fst_in_str.c_str(), O_RDONLY, 0);

  struct stat sb;
  auto res = fstat(fd, &sb);
  KALDI_ASSERT(!res);
  int64 len_of_file = sb.st_size;
  auto len_of_range = len_of_file;
  auto pages_in_range = (len_of_range+pagesize-1) / pagesize;

  /*
  if (posix_fadvise(fd, 0, len_of_range, POSIX_FADV_SEQUENTIAL)) 
    KALDI_ERR << "fail in posix_fadvise";
    */

  void* mem = mmap(NULL, len_of_range, PROT_READ, MAP_SHARED, fd, 0);
  uint junk_counter=0;
  for (int i=0; i<pages_in_range; i++) {
    junk_counter += ((char*)mem)[i*pagesize];
  }
  if (munmap(mem, len_of_range)) KALDI_ERR<< "unable to munmap file";
  KALDI_LOG << "page_stat: " << page_stat(fst_in_str);
  KALDI_LOG<<"loading_end ( "<<start<<" , "<<end<<" ) "<<junk_counter<<" "<<time(NULL);
}
void PreloadFstC3(string fst_in_str, size_t start, size_t end) {
  auto fd = open(fst_in_str.c_str(), O_RDONLY, 0);
  struct stat sb;
  auto res = fstat(fd, &sb);
  KALDI_ASSERT(!res);
  int64 len_of_file = sb.st_size;
  auto len_of_range = len_of_file;
  if (posix_fadvise(fd, 0, len_of_range, POSIX_FADV_WILLNEED)) 
    KALDI_ERR << "fail in posix_fadvise";
  KALDI_LOG << "page_stat: " << page_stat(fst_in_str);
  KALDI_LOG<<"loading_end ( "<<start<<" , "<<end<<" ) "<<len_of_range<<" "<<time(NULL);
}
void PreloadFstC4(string fst_in_str, size_t start, size_t end) {
  auto fd = open(fst_in_str.c_str(), O_RDONLY, 0);
  struct stat sb;
  auto res = fstat(fd, &sb);
  KALDI_ASSERT(!res);
  int64 len_of_file = sb.st_size;
  auto len_of_range = len_of_file;
  auto pagesize = sysconf(_SC_PAGESIZE);
  for (int64 i=len_of_range-1-pagesize; i >=0; i-=pagesize) {
    if (posix_fadvise(fd, i, i+pagesize+1, POSIX_FADV_WILLNEED)) 
      KALDI_ERR << "fail in posix_fadvise";
  }
  KALDI_LOG << "page_stat: " << page_stat(fst_in_str);
  KALDI_LOG<<"loading_end ( "<<start<<" , "<<end<<" ) "<<len_of_range<<" "<<time(NULL);
}
void PreloadFstC5(string fst_in_str, size_t start, size_t end) {
  auto fd = open(fst_in_str.c_str(), O_RDONLY, 0);
  struct stat sb;
  auto res = fstat(fd, &sb);
  KALDI_ASSERT(!res);
  int64 len_of_file = sb.st_size;
  auto len_of_range = len_of_file;
  auto pagesize = sysconf(_SC_PAGESIZE);
  for (int64 i=0; i < len_of_range; i+=pagesize) {
    if (posix_fadvise(fd, i, i+pagesize, POSIX_FADV_WILLNEED)) 
      KALDI_ERR << "fail in posix_fadvise";
  }
  KALDI_LOG << "page_stat: " << page_stat(fst_in_str);
  KALDI_LOG<<"loading_end ( "<<start<<" , "<<end<<" ) "<<len_of_range<<" "<<time(NULL);
}




void ParaPreloadFst(string fst_in_str, size_t num_states, ThreadPool& pool, int nthreads, int type, float ratio) {
  int num_states_per_thread = (num_states*ratio)/nthreads+1;
  for(int i = 0; i < nthreads; ++i) { // TODO
    if (type==0)
      pool.enqueue(&PreloadFstC0, fst_in_str, i*num_states_per_thread, std::min(num_states, (size_t)(i+1)*num_states_per_thread));
    else if (type == -1)
      pool.enqueue(&PreloadFstC_1, fst_in_str, i*num_states_per_thread, std::min(num_states, (size_t)(i+1)*num_states_per_thread));
    else if (type==1)
      pool.enqueue(&PreloadFstC1, fst_in_str, i*num_states_per_thread, std::min(num_states, (size_t)(i+1)*num_states_per_thread));
    else if (type==2)
      pool.enqueue(&PreloadFstC2, fst_in_str, i*num_states_per_thread, std::min(num_states, (size_t)(i+1)*num_states_per_thread));
    else if (type==3)
      pool.enqueue(&PreloadFstC3, fst_in_str, i*num_states_per_thread, std::min(num_states, (size_t)(i+1)*num_states_per_thread));
    else if (type==4)
      pool.enqueue(&PreloadFstC4, fst_in_str, i*num_states_per_thread, std::min(num_states, (size_t)(i+1)*num_states_per_thread));
    else if (type==5)
      pool.enqueue(&PreloadFstC5, fst_in_str, i*num_states_per_thread, std::min(num_states, (size_t)(i+1)*num_states_per_thread));

  }
  return;
}
int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::Fst;
    using fst::StdArc;

    const char *usage =
        "Generate lattices, reading log-likelihoods as matrices\n"
        " (model is needed only for the integer mappings in its transition-model)\n"
        "Usage: latgen-faster-mapped [options] trans-model-in (fst-in|fsts-rspecifier) loglikes-rspecifier"
        " lattice-wspecifier [ words-wspecifier [alignments-wspecifier] ]\n";
    ParseOptions po(usage);
    Timer timer;
    bool allow_partial = false;
    BaseFloat acoustic_scale = 0.1;
    LatticeFasterDecoderConfig config;

    std::string word_syms_filename;
    int mmap_mode=0;
    int load_mode=0;
    int nthreads=0;
    float load_ratio=1;
    config.Register(&po);
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");

    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial, "If true, produce output even if end state was not reached.");
    po.Register("mmap-mode", &mmap_mode, "0 read 1 mmap(MAP_SHARED) 2 mmap(MAP_SHARED|MAP_POPULATE) ");
    po.Register("load-mode", &load_mode, " ");
    po.Register("load-ratio", &load_ratio, " ");
    po.Register("nthreads", &nthreads, "num threads to load fst");

    po.Read(argc, argv);

    if (po.NumArgs() < 4 || po.NumArgs() > 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        fst_in_str = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        lattice_wspecifier = po.GetArg(4),
        words_wspecifier = po.GetOptArg(5),
        alignment_wspecifier = po.GetOptArg(6);
    bool compress = (fst_in_str.find("gz") !=std::string::npos);

    TransitionModel trans_model;
    ReadKaldiObject(model_in_filename, &trans_model);

    bool determinize = config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
           : lattice_writer.Open(lattice_wspecifier)))
      KALDI_ERR << "Could not open table for writing lattices: "
                 << lattice_wspecifier;

    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;

    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
      SequentialBaseFloatMatrixReader loglike_reader(feature_rspecifier);
      // Input FST is just one FST, not a table of FSTs.
      std::string map="";
      int mmap_flags=0;
      if (mmap_mode == 2) {
        map="map";
        mmap_flags=MAP_SHARED|MAP_POPULATE;
      } else if (mmap_mode == 1) {
        map="map";
        mmap_flags=MAP_SHARED;
      }
      FLAGS_v=1;
      KALDI_LOG << "page_stat: " << page_stat(fst_in_str);
      auto *decode_fst = fst::ReadFstKaldiGeneric(fst_in_str, true, map, mmap_flags, compress);

      ThreadPool pool(nthreads); // try nthreads/2 later
      if (nthreads) {
        auto *decode_fst = dynamic_cast<ExpandedFst<StdArc>*>(fst::ReadFstKaldiGeneric(fst_in_str, true, map, mmap_flags, compress));
        ParaPreloadFst(fst_in_str, decode_fst->NumStates(), pool, nthreads, load_mode, load_ratio); // TODO: no improvement
      }

      timer.Reset();
      {
        LatticeFasterDecoder decoder(*decode_fst, config);

        for (; !loglike_reader.Done(); loglike_reader.Next()) {
          std::string utt = loglike_reader.Key();
          Matrix<BaseFloat> loglikes (loglike_reader.Value());
          loglike_reader.FreeCurrent();
          if (loglikes.NumRows() == 0) {
            KALDI_WARN << "Zero-length utterance: " << utt;
            num_fail++;
            continue;
          }

          DecodableMatrixScaledMapped decodable(trans_model, loglikes, acoustic_scale);

          double like;
          if (DecodeUtteranceLatticeFaster(
                  decoder, decodable, trans_model, word_syms, utt,
                  acoustic_scale, determinize, allow_partial, &alignment_writer,
                  &words_writer, &compact_lattice_writer, &lattice_writer,
                  &like)) {
            tot_like += like;
            frame_count += loglikes.NumRows();
            num_success++;
          } else num_fail++;
        }
      }
      delete decode_fst; // delete this only after decoder goes out of scope.
    } else { // We have different FSTs for different utterances.
      SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_in_str);
      RandomAccessBaseFloatMatrixReader loglike_reader(feature_rspecifier);
      for (; !fst_reader.Done(); fst_reader.Next()) {
        std::string utt = fst_reader.Key();
        if (!loglike_reader.HasKey(utt)) {
          KALDI_WARN << "Not decoding utterance " << utt
                     << " because no loglikes available.";
          num_fail++;
          continue;
        }
        const Matrix<BaseFloat> &loglikes = loglike_reader.Value(utt);
        if (loglikes.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_fail++;
          continue;
        }
        LatticeFasterDecoder decoder(fst_reader.Value(), config);
        DecodableMatrixScaledMapped decodable(trans_model, loglikes, acoustic_scale);
        double like;
        if (DecodeUtteranceLatticeFaster(
                decoder, decodable, trans_model, word_syms, utt, acoustic_scale,
                determinize, allow_partial, &alignment_writer, &words_writer,
                &compact_lattice_writer, &lattice_writer, &like)) {
          tot_like += like;
          frame_count += loglikes.NumRows();
          num_success++;
        } else num_fail++;
      }
    }

    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count) << " over "
              << frame_count<<" frames. "<<time(NULL);
    KALDI_LOG << "page_stat: " << page_stat(fst_in_str);

    delete word_syms;
    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
