
nohead=false
composeaddin=""
stage=0

. path.sh
. ./utils/parse_options.sh

set -x

tdir=$1
lang=$2
tree_dir=$3
AFST_names="$4"

if [ $stage -le 1 ];then
for i in end singleton nonword begin 
do
    awk -v pos=$i '$2==pos{print $1,$1}' $lang/phones/word_boundary.txt | sym2int.pl -f 2 $lang/phones.txt -   > $tdir/$i.dis.int
done

#only these positions can be word-end boundary
cat $tdir/{end,singleton,nonword}.dis.int > $tdir/SOA.int

#new phones.txt including EOA & SOA symbols
#the last one is always the common #EOA
awk  'NR==FNR{di[$1]=$2}NR!=FNR&&$0!~"#"{d[$2]=$1;f=$2}NR!=FNR&&$0!~"<eps>"{print $1,$2,$2;fa=$2}END{c=fa+1;{for (i in di){print "#S."i,c,di[i];c++;print "#E."i,c,di[i];c++;}};print "#EOA",c,-2;print "#INIT",c+1,-3}'  $tdir/SOA.int $lang/phones.txt \
    | awk -v tdir=$tdir 'BEGIN{printf"">tdir"/phones.afst.txt";printf"">tdir"/LG.fst.dis.map"}{print $1,$2 >>tdir"/phones.afst.txt";print $2,$3>>tdir"/LG.fst.dis.map" }' 

#use a common EOA symbol
echo "#EOA #INIT" | sym2int.pl $tdir/phones.afst.txt -  | awk '{print 0,1,$1,0;print 1,2,$2,0;print 2}' | fstcompile - $tdir/EOA.fst 

#| afstcomposecontext $composeaddin --binary=false --context-size=2 --central-position=1 --read-disambig-syms=$lang/phones/disambig.int --write-disambig-syms=$tdir/disambig_ilabels.int --write-disambig-afst-syms=$tdir/disambig_ilabels_afst.int $tdir/dis.map $tdir/ilabels.int - \

#e.g. only word_end symbols can be connected to the start of AFST
# NOTICE: dif from AFST, here, we use EOA-SOA
unused_id=`awk 'END{print $2+1}' $tdir/words.txt`
cat $tdir/SOA.int \
    | awk  'NR==FNR{d[$1]=$2}NR!=FNR{print 0,1,d["#S."$1],0}END{print 1}' $tdir/phones.afst.txt -  | fstcompile - \
|fstconcat  $tdir/EOA.fst - \
 > $tdir/SOAEOA.fst #its really EOA-SOA
fi

if [ $stage -le 2 ];then
afst_cmds=""
for i in $AFST_names; do
afst_id=`echo $i | sym2int.pl $tdir/words.txt - | awk '{print $1}'`
afst_cmds=$afst_cmds" $tdir/SOAEOA.fst $afst_id " 
done

afstreplace  --write-disambig-syms=$tdir/LG.afst.disam.int --epsilon_on_replace $tdir/LG.fst $unused_id $afst_cmds  $tdir/LG.afst.$$
mv $tdir/LG.afst.$$ $tdir/LG.afst

#gen dis.map.LG.afst
#awk 'NR==FNR{d[$1]=$2;print}NR!=FNR{did=$1%65536;if (d[did]==""){print "ERROR in "did; exit}print $1,d[did]}' $tdir/LG.fst.dis.map $tdir/LG.afst.disam.int > $tdir/LG.afst.dis.map

#call new fstcomposecontext; process left-context correctly; #EOA & #SOA are consistent
cat $tdir/LG.afst \
| afstcomposecontext $composeaddin --binary=false --context-size=2 --central-position=1 --read-disambig-syms=$lang/phones/disambig.int --write-disambig-syms=$tdir/CLG.afst.disambig_ilabels_2_1.int --write-disambig-afst-syms=$tdir/CLG.afst.disambig_ilabels_afst.int $tdir/LG.fst.dis.map $tdir/CLG.afst.ilabels - \
> $tdir/CLG.afst
fi

if [ $stage -le 3 ];then

make-h-transducer --disambig-syms-out=$tdir/disambig_tid.int \
    --disambig-syms-map-out=$tdir/hclga.fst.disam.map \
    --transition-scale=1.0 $tdir/CLG.afst.ilabels $tree_dir/tree $tree_dir/final.mdl \
    | fstarcsort --sort_type=olabel - \
     > $tdir/Ha.fst
#NOTICE: weight pushing is important becuase afstconcat will ignore the weight after EOA symbols
cat $tdir/CLG.afst \
| fstarcsort --sort_type=ilabel -  \
|  fsttablecompose $tdir/Ha.fst - \
| fstdeterminizestar --use-log=true \
|    fstminimizeencoded \
|    fstarcsort --sort_type=ilabel  \
    > $tdir/hclga.fst

fi
