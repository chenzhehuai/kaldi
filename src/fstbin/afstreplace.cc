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

#include <fst/fstlib.h>
#include <fst/fst-decl.h>
#include <fst/script/print-impl.h>
#include "fstext/kaldi-fst-io.h"

#include "fst/replace-util.h"
#include "fstext/replace-afst.h"

//namespace fst {
//fst::VectorFst<afst::StdArc> *ReadFstKaldi(std::string rxfilename);
//
//
//void WriteFstKaldi(const fst::VectorFst<afst::StdArc> &fst,
//                   std::string wxfilename);
//}

using LabelFstPair = std::pair<typename afst::StdArc::Label, const fst::Fst<afst::StdArc> *>;

static void Cleanup(std::vector<LabelFstPair> *pairs) {
  for (const auto &pair : *pairs) {
    delete pair.second;
  }
  pairs->clear();
}

static bool GetReplaceLabelType(const string &str, bool epsilon_on_replace,
                         fst::ReplaceLabelType *rlt) {
  using namespace fst;
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

template <class Int>
static bool WriteIntVectorSimple(std::string wxfilename,
                              const std::vector<Int> &list) {
  kaldi::Output ko;
  // false, false is: text-mode, no Kaldi header.
  if (!ko.Open(wxfilename, false, false)) return false;
  for (size_t i = 0; i < list.size(); i++) ko.Stream() << list[i] << '\n';
  return ko.Close();
}

int main(int argc, char **argv) {
  try {
    using namespace kaldi;
    using namespace fst;
    using fst::ReplaceLabelType;

    std::string call_arc_labeling = "input", disambig_wxfilename;
    std::string return_arc_labeling = "neither";
    int32 return_label = 0;
    bool epsilon_on_replace = true;

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
    const string fst_out_str = po.NumArgs() % 2 == 1 ? po.GetArg(po.NumArgs()) : "";

    //auto *ifst = FstClass::Read(in_name);
    VectorFst<afst::StdArc>* ifst=new VectorFst<afst::StdArc>();
    kaldi::Input ki(in_name);
    ReadFstKaldi(ki.Stream(), false, ifst); //because of format problem, use text

    std::vector<LabelFstPair> pairs;
    // Note that if the root label is beyond the range of the underlying FST's
    // labels, truncation will occur.
    const auto root = atoll(argv[2]);
    pairs.emplace_back(root, ifst);

    for (auto i = 3; i < po.NumArgs(); i += 2) {
    VectorFst<afst::StdArc>* ifst=new VectorFst<afst::StdArc>();
    kaldi::Input ki(po.GetArg(i));
    ReadFstKaldi(ki.Stream(), false, ifst);
      if (!ifst) {
        Cleanup(&pairs);
        return 1;
      }
      // Note that if the root label is beyond the range of the underlying FST's
      // labels, truncation will occur.
      const auto label = atoll(po.GetArg(i+1).c_str());
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
    AFSTReplaceFstOptions<afst::StdArc> opts(root, call_label_type, return_label_type, 
                           return_label);
    VectorFst<afst::StdArc> ofst;

    if (disambig_wxfilename != "") {
      std::vector<uint64> ilabel_disambig_out_vec;
      AFSTReplace(pairs, &ofst, ilabel_disambig_out_vec, opts);
      if (!WriteIntVectorSimple(disambig_wxfilename, ilabel_disambig_out_vec)) {
          KALDI_ERR << "Could not write disambiguation symbols to "
                    << PrintableWxfilename(disambig_wxfilename) << '\n';
          return 1;
      }
    } else {
      AFSTReplace(pairs, &ofst, opts);
    }
    kaldi::Output ko(fst_out_str, false, false);
    WriteFstKaldi(ko.Stream(), false, ofst);
    Cleanup(&pairs);
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
