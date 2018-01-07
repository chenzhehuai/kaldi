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

namespace fst {

/*

template <class Arc>
struct ToyOptions : CacheOptions {

  ToyOptions(const CacheOptions &opts)
      : CacheOptions(opts) {}

  ToyOptions() {}
};

namespace internal {


// Implementation class for Toy
template <class A, class F>
class ToyFstImpl
    : public CacheImpl<A> {
 public:
  typedef A Arc;

  typedef typename A::Weight Weight;
  typedef typename A::StateId StateId;
  typedef DefaultCacheStore<A> Store;
  typedef typename Store::State State;


  ToyFstImpl(const Fst<A> &fst, const ToyOptions<A> &opts)
      : CacheImpl<A>(opts),
        fst_(fst.Copy()) {
    uint64 props = fst.Properties(kFstProperties, false);

    SetInputSymbols(fst.InputSymbols());
    SetOutputSymbols(fst.OutputSymbols());
  }

  ToyFstImpl(const ToyFstImpl<A, F> &impl)
      : CacheImpl<A>(impl),
        fst_(impl.fst_->Copy(true)),
        delta_(impl.delta_),
        extra_ilabel_(impl.extra_ilabel_),
        extra_olabel_(impl.extra_olabel_) {
    SetType("factor-weight");
    SetProperties(impl.Properties(), kCopyProperties);
    SetInputSymbols(impl.InputSymbols());
    SetOutputSymbols(impl.OutputSymbols());
  }
        

  // Find state corresponding to an element. Create new state
  // if element not found.
  StateId FindState(const Element &e) {
    typename ElementMap::iterator eit = element_map_.find(e);
    if (eit != element_map_.end()) {
      return (*eit).second;
    } else {
      StateId s = elements_.size();
      elements_.push_back(e);
      element_map_.insert(pair<const Element, StateId>(e, s));
      return s;
    }
  }


 private:


  std::unique_ptr<const Fst<A>> fst_;

};

}  // namespace internal
*/

template <class A, class S>
class ToyFst :
    public VectorFst<A, S> {
 public:
    using Arc = A;     
    using State = S;
    using Impl = internal::VectorFstImpl<State>;

  ToyFst() {
      VectorFst<Arc> *fst1 = RandFst<Arc>();
      VectorFst<A, S>::VectorFst(*fst1);
    }

  ToyFst(const Fst<A> &fst)
      : VectorFst<A, S>(fst) {}

  //ToyFst(const ToyFst<Arc, State> &fst, bool safe = false) 
  //    : VectorFst<A, S>(fst, safe) {}
  //ToyFst(std::shared_ptr<Impl> impl):VectorFst<A, S>(impl) {}
  //ToyFst() : VectorFst<A, S>() {}

 private:
};

template <class A, class S>
class ToyFst2 :
    public VectorFst<A, S> : 
    public ImplToMutableFst<internal::VectorFstImpl<S>> {
 public:
    using Arc = A;     
    using State = S;
    using ImplToMutableFst<internal::VectorFstImpl<S>>::impl_;
    //using Impl = internal::VectorFstImpl<State>;

  ToyFst2() {
      VectorFst<Arc> *fst1 = RandFst<Arc>();
      VectorFst<A, S>::VectorFst(*fst1);
    }

  ToyFst2(const Fst<A> &fst)
      : VectorFst<A, S>(fst) {}

  //ToyFst(const ToyFst<Arc, State> &fst, bool safe = false) 
  //    : VectorFst<A, S>(fst, safe) {}
  //ToyFst(std::shared_ptr<Impl> impl):VectorFst<A, S>(impl) {}
  //ToyFst() : VectorFst<A, S>() {}

 private:
};


}  // namespace fst

#endif

