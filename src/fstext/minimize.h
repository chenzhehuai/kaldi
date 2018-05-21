// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions and classes to minimize an FST.

#ifndef FSTEXT_MINIMIZE_H_
#define FSTEXT_MINIMIZE_H_

#include <cmath>

#include <algorithm>
#include <map>
#include <queue>
#include <utility>
#include <vector>

#include <omp.h>

#include <fst/log.h>

#include <fst/arcsort.h>
#include <fst/connect.h>
#include <fst/dfs-visit.h>
#include <fst/encode.h>
#include <fst/factor-weight.h>
#include <fst/fst.h>
#include <fst/mutable-fst.h>
#include <fst/partition.h>
#include <fst/push.h>
#include <fst/queue.h>
#include <fst/reverse.h>
#include <fst/state-map.h>

#include "base/timer.h"

namespace fst {
namespace internal {

// Computes equivalence classes for cyclic unweighted acceptors. For cyclic
// minimization we use the classic Hopcroft minimization algorithm, which has
// complexity O(E log V) where E is the number of arcs and V is the number of
// states.
//
// For more information, see:
//
//  Hopcroft, J. 1971. An n Log n algorithm for minimizing states in a finite
//  automaton. Ms, Stanford University.
//
// Note: the original presentation of the paper was for a finite automaton (==
// deterministic, unweighted acceptor), but we also apply it to the
// nondeterministic case, where it is also applicable as long as the semiring is
// idempotent (if the semiring is not idempotent, there are some complexities
// in keeping track of the weight when there are multiple arcs to states that
// will be merged, and we don't deal with this).
template <class Arc, class Queue>
class CyclicMinimizerAdv {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using ClassId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using RevArc = ReverseArc<Arc>;

  explicit CyclicMinimizerAdv(const ExpandedFst<Arc> &fst):
  t1(0), t2(0), t3(0), t4(0), t5(0) {
    timer.Reset();
    Initialize(fst);
    t1 = timer.Elapsed();
    Compute(fst);
    t2 = timer.Elapsed()-t1;
  }
  ~CyclicMinimizerAdv() {
    KALDI_LOG<< t1<<" "<<t2 <<" "<< t3<< " "<< t4<<" "<<t5;
  }

  const Partition<StateId> &GetPartition() const { return P_; }

 private:
  // StateILabelHasher is a hashing object that computes a hash-function
  // of an FST state that depends only on the set of ilabels on arcs leaving
  // the state [note: it assumes that the arcs are ilabel-sorted].
  // In order to work correctly for non-deterministic automata, multiple
  // instances of the same ilabel count the same as a single instance.
  class StateILabelHasher {
   public:
    explicit StateILabelHasher(const Fst<Arc> &fst) : fst_(fst) {}

    using Label = typename Arc::Label;
    using StateId = typename Arc::StateId;

    size_t operator()(const StateId s) {
      const size_t p1 = 7603;
      const size_t p2 = 433024223;
      size_t result = p2;
      size_t current_ilabel = kNoLabel;
      for (ArcIterator<Fst<Arc>> aiter(fst_, s); !aiter.Done(); aiter.Next()) {
        Label this_ilabel = aiter.Value().ilabel;
        if (this_ilabel != current_ilabel) {  // Ignores repeats.
          result = p1 * result + this_ilabel;
          current_ilabel = this_ilabel;
        }
      }
      return result;
    }

   private:
    const Fst<Arc> &fst_;
  };

  class ArcIterCompare {
   public:
    explicit ArcIterCompare(const Partition<StateId> &partition)
        : partition_(partition) {}

    ArcIterCompare(const ArcIterCompare &comp) : partition_(comp.partition_) {}

    // Compares two iterators based on their input labels.
    bool operator()(const ArcIterator<Fst<RevArc>> *x,
                    const ArcIterator<Fst<RevArc>> *y) const {
      const auto &xarc = x->Value();
      const auto &yarc = y->Value();
      return xarc.ilabel > yarc.ilabel;
    }

   private:
    const Partition<StateId> &partition_;
  };

  using ArcIterQueue =
      std::priority_queue<ArcIterator<Fst<RevArc>> *,
                          std::vector<ArcIterator<Fst<RevArc>> *>,
                          ArcIterCompare>;

 private:
  // Prepartitions the space into equivalence classes. We ensure that final and
  // non-final states always go into different equivalence classes, and we use
  // class StateILabelHasher to make sure that most of the time, states with
  // different sets of ilabels on arcs leaving them, go to different partitions.
  // Note: for the O(n) guarantees we don't rely on the goodness of this
  // hashing function---it just provides a bonus speedup.
  void PrePartition(const ExpandedFst<Arc> &fst) {
    VLOG(5) << "PrePartition";
    StateId next_class = 0;
    auto num_states = fst.NumStates();
    // Allocates a temporary vector to store the initial class mappings, so that
    // we can allocate the classes all at once.
    std::vector<StateId> state_to_initial_class(num_states);
    {
      // We maintain two maps from hash-value to class---one for final states
      // (final-prob == One()) and one for non-final states
      // (final-prob == Zero()). We are processing unweighted acceptors, so the
      // are the only two possible values.
      using HashToClassMap = std::unordered_map<size_t, StateId>;
      HashToClassMap hash_to_class_nonfinal;
      HashToClassMap hash_to_class_final;
      StateILabelHasher hasher(fst);
      for (StateId s = 0; s < num_states; ++s) {
        size_t hash = hasher(s);
        HashToClassMap &this_map =
            (fst.Final(s) != Weight::Zero() ? hash_to_class_final
                                            : hash_to_class_nonfinal);
        // Avoids two map lookups by using 'insert' instead of 'find'.
        auto p = this_map.insert(std::make_pair(hash, next_class));
        state_to_initial_class[s] = p.second ? next_class++ : p.first->second;
      }
      // Lets the unordered_maps go out of scope before we allocate the classes,
      // to reduce the maximum amount of memory used.
    }
    P_.AllocateClasses(next_class);
    for (StateId s = 0; s < num_states; ++s) {
      P_.Add(s, state_to_initial_class[s]);
    }
    for (StateId c = 0; c < next_class; ++c) L_.Enqueue(c);
    VLOG(5) << "Initial Partition: " << P_.NumClasses();
  }

  // Creates inverse transition Tr_ = rev(fst), loops over states in FST and
  // splits on final, creating two blocks in the partition corresponding to
  // final, non-final.
  void Initialize(const ExpandedFst<Arc> &fst) {
    // Constructs Tr.
    Reverse(fst, &Tr_);
    ILabelCompare<RevArc> ilabel_comp;
    ArcSort(&Tr_, ilabel_comp);
    // Tells the partition how many elements to allocate. The first state in
    // Tr_ is super-final state.
    P_.Initialize(Tr_.NumStates() - 1);
    // Prepares initial partition.
    PrePartition(fst);
    // Allocates arc iterator queue.
    ArcIterCompare comp(P_);
    aiter_queue_.reset(new ArcIterQueue(comp));
  }
  // Partitions all classes with destination C.
  void Split(ClassId C) {
    // Prepares priority queue: opens arc iterator for each state in C, and
    // inserts into priority queue.
    for (PartitionIterator<StateId> siter(P_, C); !siter.Done(); siter.Next()) {
      StateId s = siter.Value();
      if (Tr_.NumArcs(s + 1)) {
        aiter_queue_->push(new ArcIterator<Fst<RevArc>>(Tr_, s + 1));
      }
    }
    // Now pops arc iterator from queue, splits entering equivalence class, and
    // re-inserts updated iterator into queue.
    Label prev_label = -1;
    while (!aiter_queue_->empty()) {
      std::unique_ptr<ArcIterator<Fst<RevArc>>> aiter(aiter_queue_->top());
      aiter_queue_->pop();
      if (aiter->Done()) continue;
      const auto &arc = aiter->Value();
      auto from_state = aiter->Value().nextstate - 1;
      auto from_label = arc.ilabel;
      if (prev_label != from_label) P_.FinalizeSplit(&L_);
      auto from_class = P_.ClassId(from_state);
      if (P_.ClassSize(from_class) > 1) P_.SplitOn(from_state);
      prev_label = from_label;
      aiter->Next();
      if (!aiter->Done()) aiter_queue_->push(aiter.release());
    }
    P_.FinalizeSplit(&L_);
  }

  // Main loop for Hopcroft minimization.
  void Compute(const Fst<Arc> &fst) {
    // Processes active classes (FIFO, or FILO).
    while (!L_.Empty()) {
      const auto C = L_.Head();
      L_.Dequeue();
      Split(C);  // Splits on C, all labels in C.
    }
  }

 private:
  // Partioning of states into equivalence classes.
  Partition<StateId> P_;
  // Set of active classes to be processed in partition P.
  Queue L_;
  // Reverses transition function.
  VectorFst<RevArc> Tr_;
  // Priority queue of open arc iterators for all states in the splitter
  // equivalence class.
  std::unique_ptr<ArcIterQueue> aiter_queue_;
  kaldi::Timer timer;
  double t1,t2,t3,t4,t5;
};

// Given a partition and a Mutable FST, merges states of Fst in place (i.e.,
// destructively). Merging works by taking the first state in a class of the
// partition to be the representative state for the class. Each arc is then
// reconnected to this state. All states in the class are merged by adding
// their arcs to the representative state.
template <class Arc, class B>
void MergeStatesAdv(const Partition<typename Arc::StateId> &partition,
                 VectorFst<Arc, B> *fst) {
  using StateId = typename Arc::StateId;
  std::vector<StateId> state_map(partition.NumClasses());
#pragma omp parallel for
  for (StateId i = 0; i < partition.NumClasses(); ++i) {
    PartitionIterator<StateId> siter(partition, i);
    state_map[i] = siter.Value();  // First state in partition.
  }
  // Relabels destination states.
    atomic_int g_c(0);
    int num_class=partition.NumClasses();
#pragma omp parallel
  {
    while (1) {
      StateId c = atomic_fetch_add(&g_c, 1);
      if (c >= num_class) break;
  //for (StateId c = 0; c < partition.NumClasses(); ++c) {
    for (PartitionIterator<StateId> siter(partition, c); !siter.Done();
         siter.Next()) {
      const auto s = siter.Value();
      for (MutableArcIterator<VectorFst<Arc, B>> aiter(fst, s); !aiter.Done();
           aiter.Next()) {
        auto arc = aiter.Value();
        arc.nextstate = state_map[partition.ClassId(arc.nextstate)];
        if (s == state_map[c]) {  // For the first state, just sets destination.
          aiter.SetValue(arc);
        } else {
          fst->AddArc(state_map[c], arc);
        }
      }
    }
    }
  }
  fst->SetStart(state_map[partition.ClassId(fst->Start())]);
  Connect(fst);
}

template <class Arc, class B>
void AcceptorMinimizeAdv(VectorFst<Arc, B> *fst,
                      bool allow_acyclic_minimization = true) {
  /*
   * if (!(fst->Properties(kAcceptor | kUnweighted, true) ==
        (kAcceptor | kUnweighted))) {
    FSTERROR() << "FST is not an unweighted acceptor";
    fst->SetProperties(kError, kError);
    return;
  }
  */
  kaldi::Timer timer;
  double t1=0,t2=0,t3=0,t4=0;
  // Connects FST before minimization, handles disconnected states.
  // Connect(fst);
  if (fst->NumStates() == 0) return;
  t1=timer.Elapsed();
  if (allow_acyclic_minimization && fst->Properties(kAcyclic, true)) {
    // Acyclic minimization (Revuz).
    VLOG(2) << "Acyclic minimization";
    assert(0);
  } else {
    // Either the FST has cycles, or it's generated from non-deterministic input
    // (which the Revuz algorithm can't handle), so use the cyclic minimization
    // algorithm of Hopcroft.
    VLOG(2) << "Cyclic minimization";
  //using FST = VectorFst<Arc, VectorState<Arc, PoolAllocator<Arc>>>;
  //FST *pool_fst = new FST;
  //Cast(*fst, pool_fst);
  CyclicMinimizerAdv<Arc, LifoQueue<typename Arc::StateId>> minimizer(*fst);
  t2=timer.Elapsed();

  MergeStatesAdv(minimizer.GetPartition(), fst);
  //Cast(*pool_fst, fst);
  //delete pool_fst;
  t3=timer.Elapsed();
  }
  // Merges in appropriate semiring
  ArcUniqueMapper<Arc> mapper(*fst);
  StateMap(fst, mapper);
  t4=timer.Elapsed();
  KALDI_LOG << t1 << " "<< t2-t1 << " " << t3-t2 << " " << t4-t3;
}

}  // namespace internal


}  // namespace fst

#endif  // FSTEXT_MINIMIZE_H_
