// decoder/simple-decoder.h

// Copyright 2009-2013  Microsoft Corporation;  Lukas Burget;
//                      Saarland University (author: Arnab Ghoshal);
//                      Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_DECODER_SIMPLE_DECODER_H_
#define KALDI_DECODER_SIMPLE_DECODER_H_

#include "util/stl-utils.h"
#include "fst/fstlib.h"
#include "lat/kaldi-lattice.h"
#include "itf/decodable-itf.h"
#include "omp.h"

#include "cuda-decoder-utils.h"
namespace kaldi {
  
/** 
 * Simple Cuda Decoder
 */
class CudaDecoder;

struct CudaDecoderConfig {
  BaseFloat beam;
  double gpu_fraction;
  uint32_t max_tokens_per_frame;
  uint32_t max_tokens;
  BaseFloat acoustic_scale;
  int32 chunk_len;
  
  CudaDecoderConfig(): beam(16.0),
                       gpu_fraction(1.0/8.0),
                       max_tokens(300000000),
                       acoustic_scale(0.1), chunk_len(1) {}
  
  void Register(OptionsItf *opts) {
    opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
    opts->Register("gpu-fraction", &gpu_fraction, "Percent of GPU to use for this decoder.  "
                                                  "A single decoding cannot saturate the device.  "
                                                  "Use multiple decoders in parallel for the best performance.");
    opts->Register("max-tokens-allocated", &max_tokens, "Total number of tokens allocated.  This controls how many tokens are allocated to the entire decoding process."
                                                        "  If actual usaged exceeds this the results are undefined.");
    opts->Register("acoustic-scale", &acoustic_scale,
                   "Scaling factor for acoustic likelihoods");
    opts->Register("chunk-len", &chunk_len, "chunk length for loading posteriors.");

  }
  void Check() const {
    KALDI_ASSERT(beam > 0.0 && gpu_fraction>0 && gpu_fraction <= 1 && max_tokens_per_frame > 0 && max_tokens>0);
  }
};

// is mostly read in coalesced accesses
struct InfoToken { // we needed to take StateId out
    BaseFloat cost; // accumulated total cost up to this point.
    int prev_token;
    int arc_idx;
};


class CudaDecoder {

 public:
  typedef fst::StdArc StdArc;
  typedef StdArc::Weight StdWeight;
  typedef StdArc::Label Label;
  typedef StdArc::StateId StateId;
  typedef float CostType;

  CudaDecoder(const CudaFst &fst, const TransitionModel &trans_model, const CudaDecoderConfig &config);  
  ~CudaDecoder();

  inline size_t getCudaMallocBytes() const { return bytes_cudaMalloc; } 
  inline size_t getCudaMallocManagedBytes() const { return bytes_cudaMallocManaged;  }

  /// Decode this utterance.
  /// Returns true if any tokens reached the end of the file (regardless of
  /// whether they are in a final state); query ReachedFinal() after Decode()
  /// to see whether we reached a final state.
  void Decode(MatrixChunker *decodable);
  void DecodeChunk(CuMatrix<BaseFloat> *post_chunk);

  bool ReachedFinal() const;

  // GetBestPath gets the decoding traceback. If "use_final_probs" is true
  // AND we reached a final state, it limits itself to final states;
  // otherwise it gets the most likely token not taking into account final-probs.
  // fst_out will be empty (Start() == kNoStateId) if nothing was available due to
  // search error.
  // If Decode() returned true, it is safe to assume GetBestPath will return true.
  // It returns true if the output lattice was nonempty (i.e. had states in it);
  // using the return value is deprecated.
  bool GetBestPath(Lattice *fst_out, bool use_final_probs = true) const;
  
  /// *** The next functions are from the "new interface". ***
  
  /// FinalRelativeCost() serves the same function as ReachedFinal(), but gives
  /// more information.  It returns the difference between the best (final-cost plus
  /// cost) of any token on the final frame, and the best cost of any token
  /// on the final frame.  If it is infinity it means no final-states were present
  /// on the final frame.  It will usually be nonnegative.
  BaseFloat FinalRelativeCost() const;

  /// InitDecoding initializes the decoding, and should only be used if you
  /// intend to call AdvanceDecoding().  If you call Decode(), you don't need
  /// to call this.  You can call InitDecoding if you have already decoded an
  /// utterance and want to start with a new utterance. 
  void InitDecoding();  


  struct ExpandArcParams {
      StateId *d_q; 
      InfoToken *d_q_info; 

      int *d_q_token_from; 
      int *d_q_token_to;
      int *d_q_token_end;

      int *d_q_token_from_narcs; 
      int *h_q_token_from_narcs;

      int *d_degrees_scan; 

      int *d_q_arc_offset; 
      int *arc_ilabels; 

      BaseFloat *arc_weights; 
      StateId *arc_nextstates; 
      BaseFloat *d_cutoff;
      BaseFloat beam; 

      uint64 *d_lookup;
        
      bool is_emitting;

      int *d_n_CTA_done;

      int *h_q_token_from_size; // to be set at the end

      int *d_curr_token;
      int *d_dbg_tok_num;
      int *barrier;
      int frame;
      uint *d_arc_offsets;
      int* d_block_sums_scan;
      CuMatrixScaledMapper cuda_decodable;
};

    int debug_max_narcs;

  void ExpandArcs(int nthreads, const ExpandArcParams &params);

  void DeviceScan(int *d_degrees, int h_prevTok_size, int *d_degrees_scan);

  void ComputeDegrees(const ExpandArcParams &params);
  void FinalizeDegreesScan();

  /// This will decode until there are no more frames ready in the decodable
  /// object, but if max_num_frames is >= 0 it will decode no more than
  /// that many frames.  If it returns false, then no tokens are alive,
  /// which is a kind of error state.
  
  /// Returns the number of frames already decoded.  
  int32 NumFramesDecoded() const { return num_frames_decoded_; }


  StateId *d_allToken; 
  InfoToken *d_allTokenInfo;

  // Used to detect last CTA alive in some kernels
  int *d_n_CTA_done;

  // At each ProcessToken, we will propagate the queue [from, to[ to [to, end[
  int *d_q_token_from;
  int *d_q_token_to;
  int *d_q_token_end; 

  // Save the offset of currToken of the current frame
  // Used for ProcessEmitting of following frame
  int *d_curr_token;

  // Total number of arcs contained in the [from,to[ queue
  // ie total # of arcs from tok.next_state, where tok is in [from,to[
  // (actually one "valid arcs" are counted, cf ComputeDegrees)
  int *d_q_token_from_narcs, *h_q_token_from_narcs; // TODO
 
  // Host Pinned memory
  // size = to - from, total # of tokens in [from,to[
  int *h_q_token_from_size;

  // Host Pinned memory
  // Total number of arcs contained in the [from,to[ queue
  // ie total # of arcs from tok.next_state, where tok is in [from,to[
  // (actually one "valid arcs" are counted, cf ComputeDegrees)

 
  // Scan of the outgoing arc degrees of tokens in [from,to[
  int *d_degrees_scan;

  // # arcs in the corresponding CTA block
  // Cf Compute degrees
  int *d_block_sums_scan;

  // Cf Compute degrees
  int *d_q_arc_offset;

  int *h_reached_final;

  // TODO remove the d_reversed_path, use only host
  StateId *d_reversed_path, *h_reversed_path;

  int *d_path_size;

  // Lookup table of all the costs
  // d_state_cost[state] -> best cost for that state
  // Resetted between frames
  // Costs is stored as an ordered int representing a float
  uint64 *d_state_cost;

  // Current cutoff for current frame
  BaseFloat *d_cutoff;

  int max_tokens;


  // Streams, overlap likelihoods copies with compute
  cudaStream_t compute_st, stream_ll;
  cudaEvent_t event_ll;

  //pre-computes log likelihoods for the current frame
 
  // ProcessEmitting decodes the frame num_frames_decoded_ of the
  // decodable object, then increments num_frames_decoded_.
  //void ProcessEmitting(DecodableInterface *decodable);

  // Descriptions in .cu file

  void InitLookup();
  void ResetLookup(bool reset = true);
  void NonEmittingLongTail(unsigned int *d_arc_offsets, const ExpandArcParams &params);

  void GetBestCost(BaseFloat *min, int *arg, bool isfinal) const;
  void ProcessEmitting();
  void ProcessNonemitting();

 
  bool ProcessToken(unsigned int *d_arc_offsets, bool is_emitting);

  
  const CudaFst fst_;

  BaseFloat beam_;

  // Keep track of the number of frames decoded in the current file.
  int32 num_frames_decoded_;

  BaseFloat *cutoff;

  int *d_dbg_tok_num;
  int *d_barrier;

  // chunk decoding
  CuMatrixScaledMapper cuda_decodable_;

  const TransitionModel &trans_model_;  // for tid to pdf mapping
  int32* id2pdf_d_;
  const CudaDecoderConfig &config_;


  size_t bytes_cudaMalloc, bytes_cudaMallocManaged;

  KALDI_DISALLOW_COPY_AND_ASSIGN(CudaDecoder);
};


} // end namespace kaldi.


#endif
