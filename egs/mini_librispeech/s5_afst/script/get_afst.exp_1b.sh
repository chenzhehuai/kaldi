#!/bin/bash

dir=data_afst/toy.1b
stage=2

. path.sh

set -x

if [ $stage -le 0 ]; then
rm -rf $dir
mkdir -p $dir
fi

if [ $stage -le 1 ]; then
#new phones.txt including EOA & SOA symbols
awk '$0!~"#"{d[$2]=$1;f=$2}{print;fa=$2}END{for (i=1;i<=f;i++){print "#"d[i],i+fa};print "#EOA",i+fa}' data/lang_test_tgsmall/phones.txt > $dir/phones.txt 

#the last one is always the common #EOA
awk '$0!~"#"{d[$2]=$1;f=$2}$0!~"<eps>"{print $2,$2;fa=$2}END{for (i=1;i<=f;i++){print i+fa,i};print i+fa,-2}' data/lang_test_tgsmall/phones.txt > $dir/dis.map

for i in end singleton nonword begin 
do
    awk -v pos=$i '$2==pos{print "#"$1,"#"$1,$1,$1}' data/lang_test_tgsmall/phones/word_boundary.txt | sym2int.pl -f 2 $dir/phones.txt - | sym2int.pl -f 4 $dir/phones.txt -  > $dir/$i.dis.int
done

#only these positions can be word-end boundary
cat $dir/{end,singleton,nonword}.dis.int  | awk '{print 0,1,$2,0}END{print 1}'  | fstcompile - $dir/SOA.fst

#use a common EOA symbol
tail -n 1 $dir/dis.map | awk '{print 0,1,$1,0;print 1}' | fstcompile - $dir/EOA.fst 

cat << EOF  > $dir/toy.fst.txt
0 1 2 1
0 2 19 2
0 3 14 3
0 4 18 4
1 2 9 0
2 3 8 0
3 5 347 0
4 5 348 0
5 1.5 
EOF

#e.g. only word_end symbols can be connected to the start of AFST
cat $dir/toy.fst.txt \
| fstcompile \
| fstconcat $dir/SOA.fst - \
| fstconcat - $dir/EOA.fst \
 > $dir/toy.afst

#call old fstcomposecontext; can't process the left-context correctly
cat $dir/toy.afst \
| fstcomposecontext --binary=false --context-size=2 --central-position=1 --read-disambig-syms=data/lang_test_tgsmall/phones/disambig.int --write-disambig-syms=$dir/disambig_ilabels_2_1.int.1  $dir/ilabels.1 - \
| fstprint \
> $dir/ctoy.afst.1

#call new fstcomposecontext; process left-context correctly; #EOA & #SOA are consistent
cat $dir/toy.afst \
| afstcomposecontext --binary=false --context-size=2 --central-position=1 --read-disambig-syms=data/lang_test_tgsmall/phones/disambig.int --write-disambig-syms=$dir/disambig_ilabels_2_1.int.2 --write-disambig-afst-syms=$dir/disambig_ilabels_afst.int $dir/dis.map $dir/ilabels.2 - \
| fstprint \
> $dir/ctoy.afst.2
fi

#concat 2 AFSTs to get the decodable search space
if [ $stage -le 2 ]; then
fstcompile $dir/ctoy.afst.2 $dir/ctoy.afst
afstconcat $dir/disambig_ilabels_afst.int $dir/ctoy.afst $dir/ctoy.afst \
| fstprint \
    > $dir/doublectoy.afst.txt
fi

