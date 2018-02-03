
nohead=false
composeaddin=""

. path.sh
. ./utils/parse_options.sh

set -x

tdir=$1
lang=$2
tree_dir=$3

#use hfst results below:
#cp $dir/hfst/{phones.afst.txt,LG.fst.dis.map,SOA.int,EOA.fst} $tdir/

if [ $nohead = false ]; then
cat $tdir/SOA.int \
    | awk  'NR==FNR{d[$1]=$2}NR!=FNR{print 0,1,d["#"$1],0}END{print 1}' $tdir/phones.afst.txt -  | fstcompile - \
| fstconcat - $tdir/LG.fst $tdir/tmp.fst
else
cp  $tdir/LG.fst $tdir/tmp.fst
fi
cat $tdir/tmp.fst \
| fstconcat - $tdir/EOA.fst \
| fstdeterminizestar --use-log=true  \
|    fstminimizeencoded \
|    fstarcsort --sort_type=ilabel  \
 > $tdir/LG.afst
#| fstpushspecial  \

#call new fstcomposecontext; process left-context correctly; #EOA & #SOA are consistent
cat $tdir/LG.afst \
| afstcomposecontext $composeaddin --binary=false --context-size=2 --central-position=1 --read-disambig-syms=$lang/phones/disambig.int --write-disambig-syms=$tdir/CLG.afst.disambig_ilabels_2_1.int --write-disambig-afst-syms=$tdir/CLG.afst.disambig_ilabels_afst.int $tdir/LG.fst.dis.map $tdir/CLG.afst.ilabels  - \
> $tdir/CLG.afst

make-h-transducer --disambig-syms-out=$tdir/disambig_tid.int \
    --disambig-syms-map-out=$tdir/hclga.fst.disam.map \
    --transition-scale=1.0 $tdir/CLG.afst.ilabels  $tree_dir/tree $tree_dir/final.mdl \
    | fstarcsort --sort_type=olabel - \
     > $tdir/Ha.fst

cat $tdir/CLG.afst \
| fstarcsort --sort_type=ilabel -  \
|  fsttablecompose $tdir/Ha.fst - \
| fstdeterminizestar --use-log=true \
|    fstminimizeencoded \
|    fstarcsort --sort_type=ilabel  \
> $tdir/hclga.fst


