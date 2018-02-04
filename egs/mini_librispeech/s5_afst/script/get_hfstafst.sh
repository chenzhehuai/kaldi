#!/bin/bash

dir=data_afst/toy.3a
stage=3
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
LANG=data/lang_nosp/
#LANG=data/lang_chain
AFST_names="afst.1 afst.2"
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
    for i in $AFST_names hfst; do mkdir -p $dir/$i; done
    awk '{for (i=2;i<=NF;i++){d[$i]++;c++}}END{for (i in d){print i,d[i],d[i]/c} }' data/dev_clean_2/text | sort -k 2nr | awk 'NR>10&&NR<20{print;c+=$3}END{}' > $dir/words.afst.1 
    awk '{for (i=2;i<=NF;i++){d[$i]++;c++}}END{for (i in d){print i,d[i],d[i]/c} }' data/dev_clean_2/text | sort -k 2nr | awk 'NR>20&&NR<40{print;c+=$3}END{}' > $dir/words.afst.2 #total around 17%
    cp data/dev_clean_2/text $dir/text.hfst
    cp data/lang_nosp_test_tgsmall/words.txt $dir/
    for i in $AFST_names; do
    awk -v na=$i 'NR==FNR{d[$1]=na}NR!=FNR{$1="";for (i=2;i<=NF;i++){if (d[$i]!=""){$i=d[$i]}}print}' $dir/words.$i $dir/text.hfst > $dir/text.hfst.$$
    awk -v na=$i '{print}END{print na,$2+1}' $dir/words.txt > $dir/words.txt.$$
    mv $dir/words.txt.$$  $dir/words.txt
    mv $dir/text.hfst.$$ $dir/text.hfst
    done
    ngram-count   -lm $dir/hfst/lm  -text $dir/text.hfst -order  3 #-vocab $dir/words.txt 
fi


if [ $stage -le 2 ]; then
    echo compile our LM ref: ../s5_otf/local/format_lms.sh

    disam_st=`awk '$0~"#"{print $2;exit}' $LANG/phones.txt`
    dis_num=`awk '$0~"#"{c++}END{print c}' $LANG/phones.txt`
    #6 is SPN, need to add disambig symbols
    echo $AFST_names | sym2int.pl $dir/words.txt - | awk -v dis_num=$dis_num -v dst=$disam_st '{isym=6;dis=dst;st=1;for (i=1;i<=NF;i++){print 0,st,6,$i;print st,0,dis,0;st++;if (i%dis_num==0){dis=dst;isym++;if (isym==11){print "FAIL because of TOO many AFSTs; please re-generate disam symbols";exit}}else{dis++}}{print 0}}' | fstcompile - \
        | fstunion $LANG/L_disambig.fst - $dir/L_disambig.fst

  for i in $AFST_names hfst
  do
  cp $dir/words.txt  $dir/$i/
  cp -a -n $LANG/* $dir/$i/
  if [ $i = hfst ]; then
  cat $dir/$i/lm | \
    arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$dir/words.txt - $dir/$i/G.fst
  else
  awk -v weight=5 '{print 0,1,$1,$1}END{print 0,1,"<eps>","<eps>";print 1, weight}' $dir/words.$i \
      | fstcompile --isymbols=$dir/words.txt --osymbols=$dir/words.txt  - \
      | fstarcsort --sort_type=ilabel - \
      >$dir/$i/G.fst
  fi
  utils/validate_lang.pl --skip-determinization-check $dir/$i/ || exit 1;

  fsttablecompose $dir/L_disambig.fst $dir/$i/G.fst \
| fstdeterminizestar --use-log=true  \
|    fstminimizeencoded \
| fstpushspecial \
|    fstarcsort --sort_type=ilabel  \
> $dir/$i/LG.fst
 done
fi


if [ $stage -le 3 ]; then
tdir=$dir/hfst
bash script/get_hfst_1c.sh $tdir $LANG $tree_dir "$AFST_names"
fi

if [ $stage -le 4 ]; then

for na in $AFST_names
do
  tdir=$dir/$na
  cp $dir/hfst/{phones.afst.txt,LG.fst.dis.map,SOA.int,EOA.fst} $tdir/
bash script/get_afst_1c.sh $tdir $LANG $tree_dir
  bash script/get_afst_1d.sh  --nohead false $tdir  $LANG $tree_dir 
done
fi

exit

if [ $stage -le 5 ]; then
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


