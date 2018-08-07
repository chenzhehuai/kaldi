#include "deterministic-fst.h"
#include "kaldi-fst-io.h"
#include "string.h"


namespace fst {
  extern "C" {
    typedef fst::StdArc Arc;
    CacheDeterministicOnDemandFst<StdArc> *cdfst;
    BackoffDeterministicOnDemandFst<StdArc> *dfst;
    VectorFst<StdArc> *fst;
    int allocate(char *new_lm_fst_rxfilename) {
      KALDI_VLOG(2) << new_lm_fst_rxfilename;
      fst = CastOrConvertToVectorFst(
            ReadFstKaldiGeneric(new_lm_fst_rxfilename));
      dfst = new BackoffDeterministicOnDemandFst<StdArc>(*fst);
      cdfst = new CacheDeterministicOnDemandFst<StdArc>(dfst);
      if (dfst && fst && cdfst) return 0;
      else return 1;
    }
    void free() {
      if (cdfst) delete cdfst; cdfst = NULL;
      if (dfst) delete dfst; dfst = NULL;
      if (fst) delete fst; fst = NULL;
    }
    int init() {
      return cdfst->Start();
    }
    int get_next(int lm_state, int cid, int& lm_state_next, float& score) {
      KALDI_VLOG(2) << lm_state << " " << cid;
      Arc lm_arc;
      bool ans = cdfst->GetArc(lm_state, cid, &lm_arc);
      if (!ans) return 1;
      lm_state_next = lm_arc.nextstate;
      score = lm_arc.weight.Value();
      return 0;
    }
  }
}
