
nohead=false
composeaddin=""

. path.sh
. ./utils/parse_options.sh

set -x

tdir=$1
lang=$2
tree_dir=$3
AFST_names="$4"

#new phones.txt including EOA & SOA symbols
#the last one is always the common #EOA
awk -v nas="$AFST_names" '$0!~"#"{d[$2]=$1;f=$2}{print $1,$2,$2;fa=$2}END{split(nas,naa," ");c=fa+1;for (na in naa){for (i=1;i<=f;i++){print "#"d[i]"#"naa[na],c,i;c++}};print "#EOA",c,-2}' $lang/phones.txt \
    | awk -v tdir=$tdir 'BEGIN{printf"">tdir"/phones.afst.txt";printf"">tdir"/dis.map"}{print $1,$2 >>tdir"/phones.afst.txt";print $2,$3>>tdir"/dis.map" }' 


for i in end singleton nonword begin 
do
    awk -v pos=$i '$2==pos{print $1,$1}' $lang/phones/word_boundary.txt | sym2int.pl -f 2 $tdir/phones.afst.txt -   > $tdir/$i.dis.int
done

#only these positions can be word-end boundary
cat $tdir/{end,singleton,nonword}.dis.int > $tdir/SOA.int

#use a common EOA symbol
echo "#EOA" | sym2int.pl $tdir/phones.afst.txt -  | awk '{print 0,1,$1,0;print 1}' | fstcompile - $tdir/EOA.fst 

#| afstcomposecontext $composeaddin --binary=false --context-size=2 --central-position=1 --read-disambig-syms=$lang/phones/disambig.int --write-disambig-syms=$tdir/disambig_ilabels.int --write-disambig-afst-syms=$tdir/disambig_ilabels_afst.int $tdir/dis.map $tdir/ilabels.int - \

cp $tdir/LG.fst $tdir/LG.afst
for i in  $AFST_names; do
#e.g. only word_end symbols can be connected to the start of AFST
unused_id=`awk 'END{print $2+1}' $tdir/words.txt`
cat $tdir/SOA.int \
    | awk -v na=$i 'NR==FNR{d[$1]=$2}NR!=FNR{print 0,1,d["#"$1"#"na],0}END{print 1}' $tdir/phones.afst.txt -  | fstcompile - \
|fstconcat - $tdir/EOA.fst \
 > $tdir/SOAEOA.fst

afst_id=`echo $i | sym2int.pl $tdir/words.txt - `
fstreplace  --epsilon_on_replace $tdir/LG.afst $unused_id $tdir/SOAEOA.fst $afst_id  $tdir/LG.afst.$$
mv $tdir/LG.afst.$$ $tdir/LG.afst
done

exit

#call new fstcomposecontext; process left-context correctly; #EOA & #SOA are consistent
cat $tdir/LG.afst \
| afstcomposecontext $composeaddin --binary=false --context-size=2 --central-position=1 --read-disambig-syms=$lang/phones/disambig.int --write-disambig-syms=$tdir/disambig_ilabels_2_1.int.2 --write-disambig-afst-syms=$tdir/disambig_ilabels_afst.int $tdir/dis.map $tdir/ilabels.2 - \
> $tdir/CLG.afst

make-h-transducer --disambig-syms-out=$tdir/disambig_tid.int \
    --transition-scale=1.0 $tdir/ilabels.2 $tree_dir/tree $tree_dir/final.mdl \
    | fstarcsort --sort_type=olabel - \
     > $tdir/Ha.fst
#NOTICE: weight pushing is important becuase afstconcat will ignore the weight after EOA symbols
cat $tdir/CLG.afst \
| fstarcsort --sort_type=ilabel -  \
|  fsttablecompose $tdir/Ha.fst - \
| fstdeterminizestar --use-log=true \
    > $tdir/hclga.fst


