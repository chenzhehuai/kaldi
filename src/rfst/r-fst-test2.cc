#include "util/kaldi-io.h"
#include <fst/fstlib.h>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "r-fst.h"
#include "base/timer.h"

int main(int argc, char *argv[]) {
    using namespace kaldi;
    using namespace fst;
    const char *usage = "";
    typedef StdArc Arc;
    ParseOptions po(usage);
    po.Read(argc, argv);
    std::string f = po.GetArg(1);
    auto *fst= fst::ReadFstKaldiGeneric(f, true);
    Fst<Arc>& wrap_wfst = *fst;
    Timer timer;
    typename Arc::Weight weight;
    int narc=0;
    for (StateIterator<Fst<Arc>> siter(wrap_wfst); !siter.Done(); siter.Next()) {
      auto s = siter.Value();
      ArcIterator<Fst<Arc>> aiter(wrap_wfst, s);
      for (; !aiter.Done(); aiter.Next()) {
        auto arc = aiter.Value();
        weight=Times(weight, arc.weight); // in case aiter.Value() is optimized out
        narc++;
      }
    }
    KALDI_LOG << narc << " " << weight.Value() << " " << timer.Elapsed();
    return 0;
}
