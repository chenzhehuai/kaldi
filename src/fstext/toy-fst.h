// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Simple concrete, mutable FST whose states and arcs are stored in STL vectors.

#ifndef FST_TOY_FST_H_
#define FST_TOY_FST_H_

#include <string>
#include <utility>
#include <vector>

#include <fst/log.h>

#include <fst/fst-decl.h>  // For optional argument declarations
#include <fst/mutable-fst.h>
#include <fst/test-properties.h>
#include <fst/vector-fst.h>

namespace fst {

template <class Arc, class State = VectorState<Arc>>
class ToyFst2;

template <class F, class G>
void Cast(const F &, G *);


template <class A, class S>
class ToyFst :
    public VectorFst<A, S> {
 public:
    using Arc = A;     
    using State = S;
    using Impl = internal::VectorFstImpl<State>;

  ToyFst(const Fst<A> &fst)
      : VectorFst<A, S>(fst) {}

 private:
};


namespace internal {

// This is a ToyFst2BaseImpl container that holds ToyStates and manages FST
// properties.
template <class S>
class ToyFst2Impl : public VectorFstImpl<S> {
 public:
  using State = S;
  using Arc = typename State::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  friend class MutableArcIterator<ToyFst2<Arc, State>>;

  ToyFst2Impl() : VectorFstImpl<S>() {
  }

  explicit ToyFst2Impl(const Fst<Arc> &fst) {  
      SetType("vector");
      SetInputSymbols(fst.InputSymbols());
      SetOutputSymbols(fst.OutputSymbols());
      BaseImpl::SetStart(fst.Start());
      if (fst.Properties(kExpanded, false)) {
        BaseImpl::ReserveStates(CountStates(fst));
      }
      for (StateIterator<Fst<Arc>> siter(fst); !siter.Done(); siter.Next()) {
        const auto state = siter.Value();
        BaseImpl::AddState();
        BaseImpl::SetFinal(state, fst.Final(state));
        ReserveArcs(state, fst.NumArcs(state));
        for (ArcIterator<Fst<Arc>> aiter(fst, state); !aiter.Done(); aiter.Next()) {
          const auto &arc = aiter.Value();
          Arc arc2(arc.ilabel, arc.olabel, arc.weight*0.1, arc.nextstate);
          BaseImpl::AddArc(state, arc2);
        }
      }
      SetProperties(fst.Properties(kCopyProperties, false) | kStaticProperties);
  }
};

}  // namespace internal

// Simple concrete, mutable FST. This class attaches interface to implementation
// and handles reference counting, delegating most methods to ImplToMutableFst.
// Also supports ReserveStates and ReserveArcs methods (cf. STL vector methods).
// The second optional template argument gives the State definition.
template <class A, class S /* = ToyState<A> */>
//class ToyFst2 : public ImplToMutableFst<internal::VectorFstImpl<S>> {
class ToyFst2 : public ImplToMutableFst<internal::ToyFst2Impl<S>> {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;

  using State = S;
  //using Impl = internal::VectorFstImpl<State>;
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

// Writes FST to file in Toy format, potentially with a pass over the machine
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

#endif  // FST_VECTOR_FST_H_

