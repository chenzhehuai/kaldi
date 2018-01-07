// fstext/toy-fst.h

// Copyright 2009-2011  Microsoft Corporation

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
//
//
// This is a modified file from the OpenFST Library v1.2.7 available at
// http://www.openfst.org and released under the Apache License Version 2.0.
//
//
// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright 2005-2010 Google, Inc.
// Author: allauzen@google.com (Cyril Allauzen)


#ifndef KALDI_FSTEXT_TOY_FST_H_
#define KALDI_FSTEXT_TOY_FST_H_

#include <fst/vector-fst.h>
#include "fstext/fst-test-utils.h"

namespace fst {


template <class A, class S>
class ToyFst :
    public VectorFst<A, S> {
 public:
    using Arc = A;     
    using State = S;
    using Impl = internal::VectorFstImpl<State>;

  ToyFst(const Fst<A> &fst)
      : VectorFst<A, S>(fst) {}

  //ToyFst(const ToyFst<Arc, State> &fst, bool safe = false) 
  //    : VectorFst<A, S>(fst, safe) {}
  //ToyFst(std::shared_ptr<Impl> impl):VectorFst<A, S>(impl) {}
  //ToyFst() : VectorFst<A, S>() {}

 private:
};

namespace internal {

template <class S>
class ToyFst2Impl : public VectorFstImpl<S> {
}

}

template <class A, class S /* = VectorState<A> */>
class ToyFst2 : public ImplToMutableFst<internal::ToyFst2Impl<S>> {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;

  using State = S;
  using Impl = internal::ToyFst2Impl<State>;

  friend class StateIterator<ToyFst2<Arc, State>>;
  friend class ArcIterator<ToyFst2<Arc, State>>;
  friend class MutableArcIterator<ToyFst2<A, S>>;

  template <class F, class G>
  friend void Cast(const F &, G *);

  ToyFst2() : ImplToMutableFst<Impl>(std::make_shared<Impl>()) {}

  explicit ToyFst2(const Fst<Arc> &fst)
      : ImplToMutableFst<Impl>(std::make_shared<Impl>(fst)) {}

  ToyFst2(const ToyFst2<Arc, State> &fst, bool safe = false)
      : ImplToMutableFst<Impl>(fst) {}

  // Get a copy of this ToyFst2. See Fst<>::Copy() for further doc.
  ToyFst2<Arc, State> *Copy(bool safe = false) const override {
    return new ToyFst2<Arc, State>(*this, safe);
  }

  ToyFst2<Arc, State> &operator=(const ToyFst2<Arc, State> &fst) {
    SetImpl(fst.GetSharedImpl());
    return *this;
  }

  ToyFst2<Arc, State> &operator=(const Fst<Arc> &fst) override {
    if (this != &fst) SetImpl(std::make_shared<Impl>(fst));
    return *this;
  }

  // Reads a ToyFst2 from an input stream, returning nullptr on error.
  static ToyFst2<Arc, State> *Read(std::istream &strm,
                                     const FstReadOptions &opts) {
    auto *impl = Impl::Read(strm, opts);
    return impl ? new ToyFst2<Arc, State>(std::shared_ptr<Impl>(impl))
                : nullptr;
  }

  // Read a ToyFst2 from a file, returning nullptr on error; empty filename
  // reads from standard input.
  static ToyFst2<Arc, State> *Read(const string &filename) {
    auto *impl = ImplToExpandedFst<Impl, MutableFst<Arc>>::Read(filename);
    return impl ? new ToyFst2<Arc, State>(std::shared_ptr<Impl>(impl))
                : nullptr;
  }

  bool Write(std::ostream &strm, const FstWriteOptions &opts) const override {
    return WriteFst(*this, strm, opts);
  }

  bool Write(const string &filename) const override {
    return Fst<Arc>::WriteFile(filename);
  }

  template <class FST>
  static bool WriteFst(const FST &fst, std::ostream &strm,
                       const FstWriteOptions &opts);

  void InitStateIterator(StateIteratorData<Arc> *data) const override {
    GetImpl()->InitStateIterator(data);
  }

  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) const override {
    GetImpl()->InitArcIterator(s, data);
  }

  inline void InitMutableArcIterator(StateId s,
                                     MutableArcIteratorData<Arc> *) override;

  using ImplToMutableFst<Impl, MutableFst<Arc>>::ReserveArcs;
  using ImplToMutableFst<Impl, MutableFst<Arc>>::ReserveStates;

 private:
  using ImplToMutableFst<Impl, MutableFst<Arc>>::GetImpl;
  using ImplToMutableFst<Impl, MutableFst<Arc>>::MutateCheck;
  using ImplToMutableFst<Impl, MutableFst<Arc>>::SetImpl;

  explicit ToyFst2(std::shared_ptr<Impl> impl)
      : ImplToMutableFst<Impl>(impl) {}
};

// Writes FST to file in Vector format, potentially with a pass over the machine
// before writing to compute number of states.
template <class Arc, class State>
template <class FST>
bool ToyFst2<Arc, State>::WriteFst(const FST &fst, std::ostream &strm,
                                     const FstWriteOptions &opts) {
  static constexpr int file_version = 2;
  bool update_header = true;
  FstHeader hdr;
  hdr.SetStart(fst.Start());
  hdr.SetNumStates(kNoStateId);
  size_t start_offset = 0;
  if (fst.Properties(kExpanded, false) || opts.stream_write ||
      (start_offset = strm.tellp()) != -1) {
    hdr.SetNumStates(CountStates(fst));
    update_header = false;
  }
  const auto properties =
      fst.Properties(kCopyProperties, false) | Impl::kStaticProperties;
  internal::FstImpl<Arc>::WriteFstHeader(fst, strm, opts, file_version,
                                         "vector", properties, &hdr);
  StateId num_states = 0;
  for (StateIterator<FST> siter(fst); !siter.Done(); siter.Next()) {
    const auto s = siter.Value();
    fst.Final(s).Write(strm);
    const int64 narcs = fst.NumArcs(s);
    WriteType(strm, narcs);
    for (ArcIterator<FST> aiter(fst, s); !aiter.Done(); aiter.Next()) {
      const auto &arc = aiter.Value();
      WriteType(strm, arc.ilabel);
      WriteType(strm, arc.olabel);
      arc.weight.Write(strm);
      WriteType(strm, arc.nextstate);
    }
    ++num_states;
  }
  strm.flush();
  if (!strm) {
    LOG(ERROR) << "ToyFst2::Write: Write failed: " << opts.source;
    return false;
  }
  if (update_header) {
    hdr.SetNumStates(num_states);
    return internal::FstImpl<Arc>::UpdateFstHeader(
        fst, strm, opts, file_version, "vector", properties, &hdr,
        start_offset);
  } else {
    if (num_states != hdr.NumStates()) {
      LOG(ERROR) << "Inconsistent number of states observed during write";
      return false;
    }
  }
  return true;
}

// Specialization for ToyFst2; see generic version in fst.h for sample usage
// (but use the ToyFst2 type instead). This version should inline.
template <class Arc, class State>
class StateIterator<ToyFst2<Arc, State>> {
 public:
  using StateId = typename Arc::StateId;

  explicit StateIterator(const ToyFst2<Arc, State> &fst)
      : nstates_(fst.GetImpl()->NumStates()), s_(0) {}

  bool Done() const { return s_ >= nstates_; }

  StateId Value() const { return s_; }

  void Next() { ++s_; }

  void Reset() { s_ = 0; }

 private:
  const StateId nstates_;
  StateId s_;
};

// Specialization for ToyFst2; see generic version in fst.h for sample usage
// (but use the ToyFst2 type instead). This version should inline.
template <class Arc, class State>
class ArcIterator<ToyFst2<Arc, State>> {
 public:
  using StateId = typename Arc::StateId;

  ArcIterator(const ToyFst2<Arc, State> &fst, StateId s)
      : arcs_(fst.GetImpl()->GetState(s)->Arcs()),
        narcs_(fst.GetImpl()->GetState(s)->NumArcs()),
        i_(0) {}

  bool Done() const { return i_ >= narcs_; }

  const Arc &Value() const { return arcs_[i_]; }

  void Next() { ++i_; }

  void Reset() { i_ = 0; }

  void Seek(size_t a) { i_ = a; }

  size_t Position() const { return i_; }

  constexpr uint32 Flags() const { return kArcValueFlags; }

  void SetFlags(uint32, uint32) {}

 private:
  const Arc *arcs_;
  size_t narcs_;
  size_t i_;
};

// Specialization for ToyFst2; see generic version in mutable-fst.h for sample
// usage (but use the ToyFst2 type instead). This version should inline.
template <class Arc, class State>
class MutableArcIterator<ToyFst2<Arc, State>>
    : public MutableArcIteratorBase<Arc> {
 public:
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  MutableArcIterator(ToyFst2<Arc, State> *fst, StateId s) : i_(0) {
    fst->MutateCheck();
    state_ = fst->GetMutableImpl()->GetState(s);
    properties_ = &fst->GetImpl()->properties_;
  }

  bool Done() const final { return i_ >= state_->NumArcs(); }

  const Arc &Value() const final { return state_->GetArc(i_); }

  void Next() final { ++i_; }

  size_t Position() const final { return i_; }

  void Reset() final { i_ = 0; }

  void Seek(size_t a) final { i_ = a; }

  void SetValue(const Arc &arc) final {
    const auto &oarc = state_->GetArc(i_);
    if (oarc.ilabel != oarc.olabel) *properties_ &= ~kNotAcceptor;
    if (oarc.ilabel == 0) {
      *properties_ &= ~kIEpsilons;
      if (oarc.olabel == 0) *properties_ &= ~kEpsilons;
    }
    if (oarc.olabel == 0) *properties_ &= ~kOEpsilons;
    if (oarc.weight != Weight::Zero() && oarc.weight != Weight::One()) {
      *properties_ &= ~kWeighted;
    }
    state_->SetArc(arc, i_);
    if (arc.ilabel != arc.olabel) {
      *properties_ |= kNotAcceptor;
      *properties_ &= ~kAcceptor;
    }
    if (arc.ilabel == 0) {
      *properties_ |= kIEpsilons;
      *properties_ &= ~kNoIEpsilons;
      if (arc.olabel == 0) {
        *properties_ |= kEpsilons;
        *properties_ &= ~kNoEpsilons;
      }
    }
    if (arc.olabel == 0) {
      *properties_ |= kOEpsilons;
      *properties_ &= ~kNoOEpsilons;
    }
    if (arc.weight != Weight::Zero() && arc.weight != Weight::One()) {
      *properties_ |= kWeighted;
      *properties_ &= ~kUnweighted;
    }
    *properties_ &= kSetArcProperties | kAcceptor | kNotAcceptor | kEpsilons |
                    kNoEpsilons | kIEpsilons | kNoIEpsilons | kOEpsilons |
                    kNoOEpsilons | kWeighted | kUnweighted;
  }

  uint32 Flags() const final { return kArcValueFlags; }

  void SetFlags(uint32, uint32) final {}

 private:
  State *state_;
  uint64 *properties_;
  size_t i_;
};

// Provides information needed for the generic mutable arc iterator.
template <class Arc, class State>
inline void ToyFst2<Arc, State>::InitMutableArcIterator(
    StateId s, MutableArcIteratorData<Arc> *data) {
  data->base = new MutableArcIterator<ToyFst2<Arc, State>>(this, s);
}

// A useful alias when using StdArc.
using StdToyFst2 = ToyFst2<StdArc>;


}  // namespace fst

#endif

