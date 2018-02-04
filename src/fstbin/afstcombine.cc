// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Performs the dynamic replacement of arcs in one FST with another FST,
// allowing for the definition of FSTs analogous to RTNs.

#include <cstring>

#include <string>
#include <vector>
#include <fst/flags.h>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/kaldi-fst-io.h"

using LabelFstPair = std::pair<int, const fst::Fst<fst::StdArc> *>;

int main(int argc, char **argv) {
  try {
    using namespace kaldi;
    using namespace fst;
    
    bool del_disambig_sym = true;
    string disambig_wxfilename;
    const char *usage = "Recursively replaces FST arcs with other FST(s).\n\n"
                   "  Usage: "
                   "afstcombine root.fst labelmap 364 713 \
                    [rule1.fst label1 labelmap  ...] [out.fst]\n";

    ParseOptions po(usage);
    AfstCombineOptions opts;
    po.Register("connect", &opts.connect, "If true, trim FST before output.");
    po.Register("del-disambig-sym", &opts.del_disambig_sym, "If true, delete 
                disambig symbols after concatenation.");
    po.Register("write-disambig-syms", &disambig_wxfilename,
                "List of disambiguation symbols on input of out.fst");
    po.Read(argc, argv);
    if (po.NumArgs() < 8) {
      po.PrintUsage();
      exit(1);
    }

    AFSTCombine<StdArc> afst_combine_data(opts);

    const string fst_name = po.GetArg(1);
    const string disam_map_name = po.GetArg(2);
    const int32 disambig_sym_start_ = atoi(po.GetArg(3).c_str());
    const int32 disambig_sym_end_ = atoi(po.GetArg(4).c_str());
    const string fst_out_str =  po.GetArg(po.NumArgs());
    if (afst_combine_data.InitHfst(fst_name, disam_map_name)) return 1;
    for (auto i = 3; i < po.NumArgs(); i += 3) {
      const string fst_name = po.GetArg(i);
      const auto label = atoll(po.GetArg(i+1).c_str());
      const string disam_map_name = po.GetArg(i+2);
      if (afst_combine_data.InitSingleAfst(fst_name, label, disam_map_name)) return 1;
    }

    if (afst_combine_data.CombineMain()) return 1;
    afst_combine_data.WriteCombineResult(fst_out_str);

    if (disambig_wxfilename != "") {
      assert(0);
      //if (afst_combine_data.WriteDisamSym(disambig_wxfilename)) return 1;
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
