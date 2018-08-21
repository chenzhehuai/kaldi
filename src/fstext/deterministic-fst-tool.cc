#include "deterministic-fst.h"
#include "kaldi-fst-io.h"
#include "string.h"


namespace fst {

template<class Arc>
class BackoffOnDemandFst {
 public:
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;

  explicit BackoffOnDemandFst(const Fst<Arc> &fst, int max_tok) : fst_(fst), max_tok_(max_tok) {
#ifdef KALDI_PARANOID
  KALDI_ASSERT(fst_.Properties(kILabelSorted, true) ==
               (kILabelSorted) &&
               "Input FST is not i-label sorted.");
#endif
}

  StateId Start() { return fst_.Start(); }

  bool GetArcs(StateId s, Label ilabel, int* lm_state_next, float* score, int& ret_num, float extra_weight = 0.0) {
  KALDI_ASSERT(ilabel != 0); //  We don't allow GetArc for epsilon.
  KALDI_VLOG(2) << s<< " " << ilabel;

  SortedMatcher<Fst<Arc> > sm(fst_, MATCH_INPUT, 1);
  sm.SetState(s);
  if (sm.Find(ilabel)) {
    for (; !sm.Done() && ret_num < max_tok_; sm.Next()) {
      auto &arca = sm.Value();
      lm_state_next[ret_num] = arca.nextstate;
      score[ret_num] = arca.weight.Value() + extra_weight;
      ret_num++;
    }
    return true;
  } else {
    // if (s == Start()) return false;
    bool f = false;
    ArcIterator<Fst<Arc> > aiter(fst_, s); // SortedMatcher cannot match ilabel == 0
    for (; !aiter.Done(); aiter.Next()) { // all <eps>
      auto &arca = aiter.Value();
      if (arca.ilabel != 0) break;
      if (arca.nextstate == s) continue; // self loop
    KALDI_VLOG(2) << "back to: " << arca.nextstate;
      f |= GetArcs(arca.nextstate, ilabel, lm_state_next, score, ret_num, arca.weight.Value() + extra_weight);
    }
    return f; // we need to match at least once
  }
}

 private:
  const Fst<Arc> &fst_;
  const int32 max_tok_;
};


  extern "C" {
    typedef fst::StdArc Arc;
    BackoffOnDemandFst<StdArc> *cdfst;
    VectorFst<StdArc> *fst;
    int allocate(char *new_lm_fst_rxfilename, int max_tok) {
      KALDI_VLOG(2) << new_lm_fst_rxfilename;
      fst = CastOrConvertToVectorFst(
            ReadFstKaldiGeneric(new_lm_fst_rxfilename));
      cdfst = new BackoffOnDemandFst<StdArc>(*fst, max_tok);
      if (fst && cdfst) return 0;
      else return 1;
    }
    void free() {
      if (cdfst) delete cdfst; cdfst = NULL;
      if (fst) delete fst; fst = NULL;
    }
    int init() {
      return cdfst->Start();
    }
    int g_idx;
    int get_next(int lm_state, int cid, int* lm_state_next, float* score, int& ret_num) {
      KALDI_VLOG(2) << lm_state << " " << cid;
      
      ret_num = 0;
      bool ans = cdfst->GetArcs(lm_state, cid, lm_state_next, score, ret_num);
      if (!ans) return 1;
      return 0;
    }
  }
}
