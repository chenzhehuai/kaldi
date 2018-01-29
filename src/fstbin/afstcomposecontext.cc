// fstbin/afstcomposecontextafst.cc

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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/simple-io-funcs.h"
#include "fst/fstlib.h"
#include "fstext/context-afst.h"
#include "fstext/fstext-utils.h"
#include "fstext/kaldi-fst-io.h"

/*
  A couple of test examples:

  pushd ~/tmpdir
  # (1) with no disambig syms.
  ( echo "0 1 1 1"; echo "1 2 2 2"; echo "2 3 3 3"; echo "3 0" ) | fstcompile | afstcomposecontextafst ilabels.sym > tmp.fst
  ( echo "<eps> 0"; echo "a 1"; echo "b 2"; echo "c 3" ) > phones.txt
  fstmakecontextsyms phones.txt ilabels.sym > context.txt
  fstprint --isymbols=context.txt --osymbols=phones.txt tmp.fst
#  0    1    <eps>/<eps>/a    a
#  1    2    <eps>/a/b    b
#  2    3    a/b/c    c
#  3


  # (2) with disambig syms:
  ( echo 4; echo 5) > disambig.list
  ( echo "<eps> 0"; echo "a 1"; echo "b 2"; echo "c 3" ) > phones.txt
  ( echo "0 1 1 1"; echo "1 2 2 2"; echo " 2 3 4 4"; echo "3 4 3 3"; echo "4 5 5 5"; echo "5 0" ) | fstcompile > in.fst
  afstcomposecontextafst --disambig-syms=disambig.list ilabels.sym in.fst tmp.fst
  fstmakecontextsyms --disambig-syms=disambig.list phones.txt ilabels.sym > context.txt
  cp phones.txt phones_disambig.txt;  ( echo "#0 4"; echo "#1 5" ) >> phones_disambig.txt
  fstprint --isymbols=context.txt --osymbols=phones_disambig.txt tmp.fst

#  0    1    <eps>/<eps>/a    a
#  1    2    <eps>/a/b    b
#  2    3    #0    #0
#  3    4    a/b/c    c
#  4    5    #1    #1
#  5

*/

static void ReadSymMap(std::string phone_map_rxfilename,
                  std::vector<int32> *phone_map) {
  phone_map->clear();
  // phone map file has format e.g.:
  // 1 1
  // 2 1
  // 3 2
  // 4 2
  std::vector<std::vector<int32> > vec;  // vector of vectors, each with two elements
  // (if file has right format). first is old phone, second is new phone
  if (!kaldi::ReadIntegerVectorVectorSimple(phone_map_rxfilename, &vec))
    KALDI_ERR << "Error reading phone map from " <<
        phone_map_rxfilename;
  for (size_t i = 0; i < vec.size(); i++) {
    if (vec[i].size() != 2 || vec[i][0]<=0  ||
       (vec[i][0]<static_cast<int32>(phone_map->size()) &&
        (*phone_map)[vec[i][0]] != -1))
      KALDI_ERR << "Error reading phone map from "
                <<   phone_map_rxfilename
                << " (bad line " << i << ")";
    if (vec[i][0]>=static_cast<int32>(phone_map->size()))
      phone_map->resize(vec[i][0]+1, -1);
    KALDI_ASSERT((*phone_map)[vec[i][0]] == -1);
    (*phone_map)[vec[i][0]] = vec[i][1];
  }
  if (phone_map->empty()) {
    KALDI_ERR << "Read empty phone map from "
              << phone_map_rxfilename;
  }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;
    /*
        # afstcomposecontextafst composes efficiently with a context fst
        # that it generates.  Without --disambig-syms specified, it
        # assumes that all input symbols of in.fst are phones.
        # It adds the subsequential symbol itself (it does not
        # appear in the output so doesn't need to be specified by the user).
        # the disambig.list is a list of disambiguation symbols on the LHS
        # of in.fst.  The symbols on the LHS of out.fst are indexes into
        # the ilabels.list file, which is a kaldi-format file containing a
        # vector<vector<int32> >, which specifies what the labels mean in
        # terms of windows of symbols.
        afstcomposecontextafst  ilabels.sym  [ in.fst [ out.fst ] ]
         --disambig-syms=disambig.list
         --context-size=3
         --central-position=1
         --binary=false
    */

    const char *usage =
        "Composes on the left with a dynamically created context FST\n"
        "\n"
        "Usage:  afstcomposecontextafst <disambig_afst_rxfilename> <ilabels-output-file>  [<in.fst> [<out.fst>] ]\n"
        "E.g:  afstcomposecontextafst dis.map ilabels.sym < LG.fst > CLG.fst\n";
    

    ParseOptions po(usage);
    bool binary = true;
    std::string disambig_rxfilename,
        disambig_wxfilename, disambig_afst_wxfilename;
    int32 N = 3, P = 1;
    po.Register("binary", &binary,
                "If true, output ilabels-output-file in binary format");
    po.Register("read-disambig-syms", &disambig_rxfilename,
                "List of disambiguation symbols on input of in.fst");
    po.Register("write-disambig-syms", &disambig_wxfilename,
                "List of disambiguation symbols on input of out.fst");
    po.Register("write-disambig-afst-syms", &disambig_afst_wxfilename,
                "List of disambiguation symbols on input of out.fst, including begin & end.");
    po.Register("context-size", &N, "Size of phone context window");
    po.Register("central-position", &P,
                "Designated central position in context window");

    po.Read(argc, argv);

    if (po.NumArgs() < 1 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string disambig_afst_rxfilename = po.GetOptArg(1),
        ilabels_out_filename = po.GetArg(2),
        fst_in_filename = po.GetOptArg(3),
        fst_out_filename = po.GetOptArg(4);

    VectorFst<StdArc> *fst = ReadFstKaldi(fst_in_filename);

    if ( (disambig_wxfilename != "") && (disambig_rxfilename == "") )
      KALDI_ERR << "afstcomposecontextafst: cannot specify --write-disambig-syms if "
          "not specifying --read-disambig-syms\n";

    std::vector<int32> disambig_in;
    if (disambig_rxfilename != "")
      if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_in))
        KALDI_ERR << "afstcomposecontextafst: Could not read disambiguation symbols from "
                  << PrintableRxfilename(disambig_rxfilename);

    if (disambig_in.empty()) {
      KALDI_WARN << "Disambiguation symbols list is empty; this likely "
                 << "indicates an error in data preparation.";
    }

    std::vector<int32> dis2phone_map;
    ReadSymMap(disambig_afst_rxfilename,
                 &dis2phone_map);
    
    std::vector<std::vector<int32> > ilabels;
    VectorFst<StdArc> composed_fst;

    // Work gets done here (see context-fst.h)
    ComposeContext(disambig_in, dis2phone_map, N, P, fst, &composed_fst, &ilabels);

    WriteILabelInfo(Output(ilabels_out_filename, binary).Stream(),
                    binary, ilabels);

    if (disambig_wxfilename != "") {
      std::vector<int32> disambig_out;
      for (size_t i = 0; i < ilabels.size(); i++)
        if (ilabels[i].size() == 1 && ilabels[i][0] <= 0) //including #PHN
          disambig_out.push_back(static_cast<int32>(i));
      if (!WriteIntegerVectorSimple(disambig_wxfilename, disambig_out)) {
        std::cerr << "afstcomposecontextafst: Could not write disambiguation symbols to "
                  << PrintableWxfilename(disambig_wxfilename) << '\n';
        return 1;
      }
    }

    if (disambig_afst_wxfilename != "") {
      std::vector<int32> disambig_afst_out;
      for (size_t i = 0; i < ilabels.size(); i++)
        if (ilabels[i].size() == 1 && ilabels[i][0] <= disambig_in[0])
          disambig_afst_out.push_back(static_cast<int32>(i));
      if (!WriteIntegerVectorSimple(disambig_afst_wxfilename, disambig_afst_out)) {
        std::cerr << "afstcomposecontextafst: Could not write disambiguation symbols to "
                  << PrintableWxfilename(disambig_afst_wxfilename) << '\n';
        return 1;
      }
    }


    WriteFstKaldi(composed_fst, fst_out_filename);
    delete fst;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

