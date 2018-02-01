#!/bin/bash

dir=data_afst/toy.2b
stage=1
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
LANG=data/lang_nosp/
#LANG=data/lang_chain

. path.sh
. cmd.sh

afstdir=$dir

set -x

if [ $stage -le 0 ]; then
rm -rf $dir
mkdir -p $dir
fi

#get a unk word filling task example
if [ $stage -le 1 ]; then
    mkdir -p $dir/oov $dir/oovadd
    cp data/lang_nosp_test_tgsmall/words.txt $dir/
    awk '{for (i=2;i<=NF;i++){d[$i]++;c++}}END{for (i in d){print i,d[i],d[i]/c} }' data/dev_clean_2/text | sort -k 2nr | awk 'NR>10&&NR<40{print;c+=$3}END{}' > $dir/words.oov #around 17%
    #awk 'NR==FNR{d[$1]="#OOV"}NR!=FNR{$1="";for (i=2;i<=NF;i++){if (d[$i]!=""){$i=d[$i]}}print}' $dir/words.oov data/dev_clean_2/text > $dir/text.oov
    #to make it extremely simple, we just make OOV words  totally empty here
    awk 'NR==FNR{d[$1]="#OOV"}NR!=FNR{$1="";for (i=2;i<=NF;i++){if (d[$i]!=""){$i=""}}print}' $dir/words.oov data/dev_clean_2/text > $dir/text.oov
    ngram-count   -lm $dir/oov/lm  -text $dir/text.oov -order  3 #-vocab $dir/words.txt 
fi

if [ $stage -le 2 ]; then
    echo compile our LM ref: ../s5_otf/local/format_lms.sh

  cat $dir/oov/lm | \
    arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$dir/words.txt - $dir/oov/G.fst
  awk -v weight=5 '{print 0,1,$1,$1}END{print 0,1,"<eps>","<eps>";print 1, weight}' $dir/words.oov \
      | fstcompile --isymbols=$dir/words.txt --osymbols=$dir/words.txt  - \
      | fstarcsort --sort_type=ilabel - \
      >$dir/oovadd/G.fst
  for i in oov oovadd
  do
  cp -a -n $LANG/* $dir/$i/
  utils/validate_lang.pl --skip-determinization-check $dir/$i/ || exit 1;
    done
fi

#baseline 0
if [ $stage -le 3 ]; then
    cp -a $dir/oov/* $dir/baseline0.1
    cp -a $dir/oovadd/* $dir/baseline0.2

    for i in baseline0.1 baseline0.2
    do
utils/mkgraph.sh \
    --self-loop-scale 1.0 $dir/$i \
    $tree_dir $dir/$i #|| exit 1;
done

fi

#baseline 1 oov_gcat_oovadd
if [ $stage -le 4 ]; then
    cp -a $dir/oov/* $dir/baseline1
    mv $dir/baseline1/G.fst $dir/baseline1/G.fst.p
    fstconcat $dir/baseline1/G.fst.p $dir/oovadd/G.fst \
        | fstconcat - $dir/baseline1/G.fst.p \
        | fstdeterminize \
        | fstarcsort --sort_type=ilabel - \
    > $dir/baseline1/G.fst 

    utils/mkgraph_1b.sh \
    --self-loop-scale 1.0 $dir/baseline1 \
    $tree_dir $dir/baseline1 #|| exit 1;

fi

#false && \
    {
#baseline 2
if [ $stage -le 5 ]; then
    mkdir $dir/baseline2
    tdir=$dir/baseline2/
    cp $dir/baseline0.1/{words.txt} $dir/baseline2/
    fstconcat $dir/{baseline0.1,baseline0.2}/tmp/CLG_2_1.fst  \
        | fstconcat - $dir/baseline0.1/tmp/CLG_2_1.fst  \
        | fstarcsort --sort_type=ilabel - \
    > $dir/baseline2/CLG_2_1.fst
  make-h-transducer --disambig-syms-out=$tdir/disambig_tid.int \
    --transition-scale=1.0 $dir/baseline0.1/tmp/ilabels_2_1 $tree_dir/tree $tree_dir/final.mdl \
     > $tdir/Ha.fst
  fsttablecompose $tdir/Ha.fst $tdir/CLG_2_1.fst \
    > $tdir/tmp.fst
  fstrmsymbols $tdir/disambig_tid.int $tdir/tmp.fst| \
 fstrmepslocal | \
     fstminimizeencoded > $tdir/HCLGa.fst || exit 1;
  fstisstochastic $tdir/HCLGa.fst || echo "HCLGa is not stochastic"
  add-self-loops --self-loop-scale=1.0 --reorder=true \
    $tree_dir/final.mdl< $tdir/HCLGa.fst | fstconvert --fst_type=const > $tdir/HCLG.fst || exit 1;
fi

}
#false && \
    {
#proposed1
if [ $stage -le 6 ]; then
mkdir -p $dir/proposed1
cp -a $dir/oov/* $dir/proposed1/oov
cp -a $dir/oovadd/* $dir/proposed1/oovadd
for tdir in $dir/proposed1/oovadd $dir/proposed1/oov
do
#false && \
    {
bash script/get_afst_1a.sh $tdir $LANG
}
#process ilabel to make two afst consistent
#TODO: because diff ilabels, first compose H; after that, we can concat AFSTs; make Ha.fst respectively

make-h-transducer --disambig-syms-out=$tdir/disambig_tid.int \
    --transition-scale=1.0 $tdir/ilabels.2 $tree_dir/tree $tree_dir/final.mdl \
    | fstarcsort --sort_type=olabel - \
     > $tdir/Ha.fst
#NOTICE: weight pushing is important becuase afstconcat will ignore the weight after EOA symbols
cat $tdir/CLG.afst \
| fstarcsort --sort_type=ilabel -  \
|  fsttablecompose $tdir/Ha.fst - \
| fstdeterminizestar --use-log=true \
| fstpushspecial \
    > $tdir/hclga.fst

done

tdir=$dir/proposed1/
#use the larger disambig_tid.int for removal
cp $dir/proposed1/oov/disambig_tid.int $tdir/disambig_tid.int
afstconcat $tdir/disambig_tid.int $dir/proposed1/oov/hclga.fst $dir/proposed1/oovadd/hclga.fst \
    | afstconcat $tdir/disambig_tid.int  - $dir/proposed1/oov/hclga.fst \
    > $tdir/hclga.fst
#normal procedure
cat $tdir/hclga.fst \
|  fstrmsymbols $tdir/disambig_tid.int - | \
 fstrmepslocal | \
     fstminimizeencoded > $tdir/HCLGa.fst || exit 1;
  fstisstochastic $tdir/HCLGa.fst || echo "HCLGa is not stochastic"
  add-self-loops --self-loop-scale=1.0 --reorder=true \
    $tree_dir/final.mdl< $tdir/HCLGa.fst | fstconvert --fst_type=const > $tdir/HCLG.fst || exit 1;
cp $dir/proposed1/oov/words.txt $tdir/

fi
}

#proposed method 
if [ $stage -le 7 ]; then

mkdir -p $dir/proposed2
cp -a $dir/{oov,oovadd} $dir/proposed2/
cp -a $dir/oov $dir/proposed2/oovh

for tdir in $dir/proposed2/oovadd $dir/proposed2/oov
do
#false && \
    {
bash script/get_afst_1b.sh $tdir $LANG $tree_dir
}
#process ilabel to make two afst consistent
#because diff ilabels, first compose H; after that, we can concat AFSTs; make Ha.fst respectively
done
bash script/get_afst_1b.sh --composeaddin "--read-ilabel-info=$dir/proposed2/oov/ilabels.2 " --nohead true $dir/proposed2/oovh  $LANG $tree_dir 

tdir=$dir/proposed2/
#use the larger disambig_tid.int for removal
cp $dir/proposed2/oov/disambig_tid.int $tdir/disambig_tid.int
fstarcsort --sort_type=olabel $dir/proposed2/oovh/hclga.fst \
| afstconcat $tdir/disambig_tid.int - $dir/proposed2/oovadd/hclga.fst \
| fstarcsort --sort_type=olabel  \
    | afstconcat $tdir/disambig_tid.int  - $dir/proposed2/oov/hclga.fst \
    > $tdir/hclga.fst
#normal procedure
cat $tdir/hclga.fst \
| fstdeterminize \
|  fstrmsymbols $tdir/disambig_tid.int - | \
 fstrmepslocal | \
     fstminimizeencoded > $tdir/HCLGa.fst || exit 1;
  fstisstochastic $tdir/HCLGa.fst || echo "HCLGa is not stochastic"
  add-self-loops --self-loop-scale=1.0 --reorder=true \
    $tree_dir/final.mdl< $tdir/HCLGa.fst | fstconvert --fst_type=const > $tdir/HCLG.fst || exit 1;
cp $dir/proposed2/oov/words.txt $tdir/

fi


#concat 2 AFSTs to get the decodable search space
if [ $stage -le 8 ]; then

chunk_left_context=0
chunk_right_context=0
chunk_width=140,100,160
frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
nnet3_affix=
tree_affix=
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
data=dev_clean_2
nspk=$(wc -l <data/${data}_hires/spk2utt)
affix=1e   # affix for the TDNN directory name
dir=exp/chain${nnet3_affix}/tdnn${affix}_sp

for graphdir in $afstdir/{proposed2,proposed1,baseline0.1,baseline0.2,baseline1,baseline2} #baseline1 #$afstdir/proposed2  #$afstdir/baseline1 $afstdir/proposed2  $afstdir/proposed1 #  #{baseline1,proposed1}
do
#false && \
    {
      steps/nnet3/decode.sh \
          --skip_diagnostics true \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context $chunk_left_context \
          --extra-right-context $chunk_right_context \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 1 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $graphdir data/${data}_hires ${dir}/decode_afst.`basename $graphdir`_${data} \
          || exit 1
      }&
done
      wait

fi


