
# Extract constraint features of a set of .smt2 files listing in a filelist.
python PathConstraintClassifier/feature.py -f ./output/small_smt.filelist -o ./output/ -extract

# Sometimes, extract features of a large amount of .smt2 files could result 
# in memory problem. (insufficient memory due to too many constrint stored in 
# pysmt.formula.FormulaManager object)
# 
# If you have this problem, you can try this to extract features by partitions.

mkdir ./output/feature_partition
python ./PathConstraintClassifier/feature.py -f ./output/small_smt.filelist -p 0 -o ./output/feature_partition -extract


