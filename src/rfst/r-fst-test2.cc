#include "util/kaldi-io.h"
#include <fst/fstlib.h>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "r-fst.h"
#include "../rfstfast/r-fst.h"
#include "base/timer.h"

template <typename FST> 
void test(FST &fst) {
  using namespace kaldi;
  using namespace fst;
  Timer timer;
  typename FST::Arc::Weight weight;
  int narc=0;
  for (StateIterator<FST> siter(fst); !siter.Done(); siter.Next()) {
    auto s = siter.Value();
    ArcIterator<FST> aiter(fst, s);
    for (; !aiter.Done(); aiter.Next()) {
      auto arc = aiter.Value();
      weight=Times(weight, arc.weight); // in case aiter.Value() is optimized out
      narc++;
    }
  }
  KALDI_LOG << narc << " " << weight.Value() << " " << timer.Elapsed();
}
int main(int argc, char *argv[]) {
    using namespace kaldi;
    using namespace fst;
    const char *usage = "";
    ParseOptions po(usage);
    po.Read(argc, argv);
    std::string f = po.GetArg(1);
    auto *fst= fst::ReadFstKaldiGeneric(f, true);
    auto *wrap_wfst = dynamic_cast<StdRFst*>(fst);
    auto *wrap_wfst2 = dynamic_cast<StdRFstFast*>(fst);
    if (wrap_wfst) test(*wrap_wfst);
    else if (wrap_wfst2) test(*wrap_wfst2);
    else test(*fst);
    return 0;
}
