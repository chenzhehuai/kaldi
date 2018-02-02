// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Performs the dynamic replacement of arcs in one FST with another FST,
// allowing for the definition of FSTs analogous to RTNs.

#include <cstring>

#include <string>
#include <vector>

#include <fst/flags.h>

#include "fst/replace-util.h"
#include "fst/replace-afst.h"

DECLARE_string(call_arc_labeling);
DECLARE_string(return_arc_labeling);
DECLARE_bool(epsilon_on_replace);

void Cleanup(std::vector<fst::script::LabelFstClassPair> *pairs) {
  for (const auto &pair : *pairs) {
    delete pair.second;
  }
  pairs->clear();
}

static bool GetReplaceLabelType(const string &str, bool epsilon_on_replace,
                         ReplaceLabelType *rlt) {
  if (epsilon_on_replace || str == "neither") {
    *rlt = REPLACE_LABEL_NEITHER;
  } else if (str == "input") {
    *rlt = REPLACE_LABEL_INPUT;
  } else if (str == "output") {
    *rlt = REPLACE_LABEL_OUTPUT;
  } else if (str == "both") {
    *rlt = REPLACE_LABEL_BOTH;
  } else {
    return false;
  }
  return true;
}


int main(int argc, char **argv) {
  try {
    using namespace kaldi;
    using namespace fst;
    using fst::ReplaceLabelType;

    std::string call_arc_labeling = "input", disambig_wxfilename;
    std::string return_arc_labeling = "neither";
    int64 return_label = 0;
    bool epsilon_on_replace = true;

    string usage = 
    const char *usage = "Recursively replaces FST arcs with other FST(s).\n\n"
                   "  Usage: "
                   "afstreplace root.fst rootlabel [rule1.fst label1 ...] [out.fst]\n";

    ParseOptions po(usage);
    po.Register("call_arc_labeling", &call_arc_labeling, "Which labels to make non-epsilon on the call arc. ");
    po.Register("return_arc_labeling", &return_arc_labeling, "Which labels to make non-epsilon on the return arc. ");
    po.Register("return_label", &return_label, "Label to put on return arc. ");
    po.Register("epsilon_on_replace", &epsilon_on_replace, "Call/return arcs are epsilon arcs?");
    po.Register("write-disambig-syms", &disambig_wxfilename,
                "List of (only) AFST disambiguation symbols on input of out.fst");

    po.Read(argc, argv);

    if (po.NumArgs() < 4) {
      po.PrintUsage();
      exit(1);
    }
    

    const string in_name = po.GetArg(1);
    const string out_name = po.NumArgs() % 2 == 1 ? po.GetArg(po.NumArgs()) : "";

    //auto *ifst = FstClass::Read(in_name);
    VectorFst<StdArc> *ifst = ReadFstKaldi(in_name);
    
    if (!ifst) return 1;

    std::vector<std::pair<int64, const Fst *>> pairs;
    // Note that if the root label is beyond the range of the underlying FST's
    // labels, truncation will occur.
    const auto root = atoll(argv[2]);
    pairs.emplace_back(root, ifst);

    for (auto i = 3; i < po.NumArgs(); i += 2) {
      VectorFst<StdArc> *ifst = ReadFstKaldi(po.GetArg(i));
      if (!ifst) {
        Cleanup(&pairs);
        return 1;
      }
      // Note that if the root label is beyond the range of the underlying FST's
      // labels, truncation will occur.
      const auto label = atoll(po.GetArg(i+1));
      pairs.emplace_back(label, ifst);
    }

    ReplaceLabelType call_label_type;
    if (!GetReplaceLabelType(call_arc_labeling, epsilon_on_replace,
                                &call_label_type)) {
      KALDI_ERR << argv[0] << ": Unknown or unsupported call arc replace "
                 << "label type: " << call_arc_labeling;
    }
    ReplaceLabelType return_label_type;
    if (!GetReplaceLabelType(return_arc_labeling,
                                epsilon_on_replace, &return_label_type)) {
      KALDI_ERR << argv[0] << ": Unknown or unsupported return arc replace "
                 << "label type: " << return_arc_labeling;
    }
    ReplaceFstOptions opts(root, call_label_type, return_label_type, 
                           return_label);
    VectorFst<StdArc> ofst;
    Replace(pairs, &ofst, opts);
    WriteFstKaldi(ofst, fst_out_str);
    std::std::vector<uint64> ilabel_disambig_out_vec;
    if (disambig_wxfilename != "") {
      GetIlabelDisambigOut(ofst, ilabel_disambig_out_vec);
      if (!WriteIntegerVectorSimple(disambig_wxfilename, ilabel_disambig_out_vec)) {
          KALDI_ERR << "Could not write disambiguation symbols to "
                    << PrintableWxfilename(disambig_wxfilename) << '\n';
          return 1;
      }
    }
    Cleanup(&pairs);
    return 0;
  }
}
