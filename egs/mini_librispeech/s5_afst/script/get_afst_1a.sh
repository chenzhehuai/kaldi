. path.sh

set -x

tdir=$1
lang=$2

#new phones.txt including EOA & SOA symbols
awk '$0!~"#"{d[$2]=$1;f=$2}{print;fa=$2}END{for (i=1;i<=f;i++){print "#"d[i],i+fa};print "#EOA",i+fa}' $lang/phones.txt > $tdir/phones.txt 

#the last one is always the common #EOA
awk '$0!~"#"{d[$2]=$1;f=$2}$0!~"<eps>"{print $2,$2;fa=$2}END{for (i=1;i<=f;i++){print i+fa,i};print i+fa,-2}' $lang//phones.txt > $tdir/dis.map

for i in end singleton nonword begin 
do
    awk -v pos=$i '$2==pos{print "#"$1,"#"$1,$1,$1}' $lang/phones/word_boundary.txt | sym2int.pl -f 2 $tdir/phones.txt - | sym2int.pl -f 4 $tdir/phones.txt -  > $tdir/$i.dis.int
done

#only these positions can be word-end boundary
cat $tdir/{end,singleton,nonword}.dis.int  | awk '{print 0,1,$2,0}END{print 1}'  | fstcompile - $tdir/SOA.fst

#use a common EOA symbol
tail -n 1 $tdir/dis.map | awk '{print 0,1,$1,0;print 1}' | fstcompile - $tdir/EOA.fst 

#e.g. only word_end symbols can be connected to the start of AFST
fsttablecompose $lang/L_disambig.fst $tdir/G.fst \
| fstconcat $tdir/SOA.fst - \
| fstconcat - $tdir/EOA.fst \
 > $tdir/LG.afst

#call new fstcomposecontext; process left-context correctly; #EOA & #SOA are consistent
cat $tdir/LG.afst \
| afstcomposecontext --binary=false --context-size=2 --central-position=1 --read-disambig-syms=$lang/phones/disambig.int --write-disambig-syms=$tdir/disambig_ilabels_2_1.int.2 --write-disambig-afst-syms=$tdir/disambig_ilabels_afst.int $tdir/dis.map $tdir/ilabels.2 - \
> $tdir/CLG.afst

