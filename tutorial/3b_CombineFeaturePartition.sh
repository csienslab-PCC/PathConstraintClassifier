
# Generate feature filelist for first.
./PathConstraintClassifier/script/DataProcess/gen_filelist.sh ./output/feature_partition ./output/feature_partition/small_smt feature

# Combine feature partitions by a feature filelist.
python ./PathConstraintClassifier/combine.py -f ./output/feature_partition/small_smt.featurelist -partition-size 100000 -o ./output/feature_partition/ -combine-features
