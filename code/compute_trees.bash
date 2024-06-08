for f in `ls ../data/cognate_classes_phylip`
do
    db=`basename $f .phy`
    raxml-ng --msa ../data/cognate_classes_phylip/$f --model BIN --prefix ../data/cognate_classes_mltree/$db.phy --redo
    cp ../data/cognate_classes_mltree/$db.phy.raxml.bestTree ../data/cognate_classes_mltree/$db"_ml.tre"
done


for f in `ls ../data/correspondences_phylip`
do
    db=`basename $f .phy`
    raxml-ng --msa ../data/correspondences_phylip/$f --model BIN --prefix ../data/correspondences_mltree/$db.phy --redo
    cp ../datas/correspondences_mltree/$db.phy.raxml.bestTree ../data/correspondences_mltree/$db"_ml.tre"
done


