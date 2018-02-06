#!/bin/bash

dir=data_afst/toy.3a
stage=2
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
lanum=`fstprint $LANG/L_disambig.fst | awk 'END{print $1}'`
    echo $AFST_names | sym2int.pl $dir/words.txt - | awk -v st=$lanum -v dis_num=$dis_num -v dst=$disam_st '{isym=6;dis=dst;for (i=1;i<=NF;i++){st++;print 1,st,6,$i;print st,1,dis,0;if (i%dis_num==0){dis=dst;isym++;if (isym==11){print "FAIL because of TOO many AFSTs; please re-generate disam symbols";exit}}else{dis++}}}' > $dir/tmp.fst.txt
#    echo $AFST_names | sym2int.pl $dir/words.txt - | awk -v st=$lanum -v dis_num=$dis_num -v dst=$disam_st '{isym=6;dis=dst;st++;for (i=1;i<=NF;i++){print 1,st,6,$i;print st,1,dis,0;if (i%dis_num==0){dis=dst;isym++;if (isym==11){print "FAIL because of TOO many AFSTs; please re-generate disam symbols";exit}}else{dis++}}}' > $dir/tmp.fst.txt
    fstprint $LANG/L_disambig.fst | awk '{print}' - $dir/tmp.fst.txt | fstcompile | fstarcsort --sort_type=olabel - $dir/L_disambig.fst 
    rm $dir/tmp.fst.txt

  for i in $AFST_names hfst
  do
  cp $dir/words.txt  $dir/$i/
  cp -a -n $LANG/* $dir/$i/
  if [ $i = hfst ]; then
  cat $dir/$i/lm | \
    arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$dir/words.txt - $dir/$i/G.fst
  else
  awk -v weight=5 '{print 0,1,$1,$1}END{print 1, weight}' $dir/words.$i \
      | fstcompile --isymbols=$dir/words.txt --osymbols=$dir/words.txt  - \
      | fstarcsort --sort_type=ilabel - \
      >$dir/$i/G.fst
  fi
  #utils/validate_lang.pl --skip-determinization-check $dir/$i/ || exit 1;

  echo "6" > $dir/$i/rm.sym
  fsttablecompose $dir/L_disambig.fst $dir/$i/G.fst \
| fstdeterminizestar --use-log=true  \
|    fstminimizeencoded \
| fstpushspecial \
| fstrmsymbols $dir/$i/rm.sym - \
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
  cp $dir/hfst/{phones.afst.txt,LG.fst.dis.map,SOA.int} $tdir/
  bash script/get_afst_1d.sh $tdir $LANG $tree_dir
done
fi

if [ $stage -le 5 ]; then
disam_begin=`awk '$1!=$2{print $1;exit}' $dir/hfst/LG.fst.dis.map`
disam_end=`awk 'END{print $1}' $dir/hfst/LG.fst.dis.map`
combine_cmd=""
i=0
for na in $AFST_names
do
  ((i++))
  combine_cmd=$combine_cmd" "$dir/$na/hclga.fst" "$i" "$dir/$na/hclga.fst.disam.map
done
  afstcombine $dir/hfst/hclga.fst $dir/hfst/hclga.fst.disam.map $disam_begin $disam_end  $combine_cmd $dir/hclga.afst
fi

if [ $stage -le 6 ]; then
  tdir=$dir
  fstarcsort --sort_type=olabel $dir/hclga.afst \
  > $tdir/HCLGa.fst || exit 1;
  fstisstochastic $tdir/HCLGa.fst || echo "HCLGa is not stochastic"
  add-self-loops --self-loop-scale=1.0 --reorder=true \
    $tree_dir/final.mdl< $tdir/HCLGa.fst | fstconvert --fst_type=const > $tdir/HCLG.fst || exit 1;

fi

false && \
  {


if [ $stage -le 7 ]; then
    echo compile our LM ref: ../s5_otf/local/format_lms.sh

    tdir=$dir/baseline3
    mkdir -p $tdir
    awk '{$1="";print }' data/dev_clean_2/text  > $tdir/text
    ngram-count   -lm $tdir/lm  -text $tdir/text -order  3 #-vocab $dir/words.txt 
  cat $tdir/lm | \
    arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$dir/words.txt - $tdir/G.fst
  cp -a -n $LANG/* $tdir/
  utils/validate_lang.pl --skip-determinization-check $tdir  || exit 1;
utils/mkgraph.sh \
    --self-loop-scale 1.0 $tdir \
    $tree_dir $tdir #|| exit 1;

fi

#baseline 1
if [ $stage -le 8 ]; then
    tdir=$dir/baseline1/
    rm -rf $tdir
    mkdir $tdir
    cp $dir/hfst/words.txt $tdir
  afst_cmds=""
  for i in $AFST_names; do
  afst_id=`echo $i | sym2int.pl $tdir/words.txt - | awk '{print $1}'`
  afst_cmds=$afst_cmds" $dir/$i/G.fst $afst_id " 
  done
  unused_id=`awk 'END{print $2+1}' $tdir/words.txt`
    fstreplace  --epsilon_on_replace $dir/hfst/G.fst $unused_id $afst_cmds  \
      | fstarcsort --sort_type=ilabel - \
      $tdir/G.fst.$$
    mv $tdir/G.fst.$$ $tdir/G.fst

    cp -a -n $LANG/* $tdir/
  utils/validate_lang.pl --skip-determinization-check $tdir  || exit 1;
utils/mkgraph.sh \
    --self-loop-scale 1.0 $tdir \
    $tree_dir $tdir #|| exit 1;

fi

}

#concat 2 AFSTs to get the decodable search space
if [ $stage -le 20 ]; then

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

for graphdir in  $afstdir/ #$afstdir/baseline2 #baseline3 #. #$afstdir/{proposed2,proposed1,baseline0.1,baseline0.2,baseline1,baseline2} #baseline1 #$afstdir/proposed2  #$afstdir/baseline1 $afstdir/proposed2  $afstdir/proposed1 #  #{baseline1,proposed1}
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


