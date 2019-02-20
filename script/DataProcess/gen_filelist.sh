
if [ $# -ne 3 ]; then
    echo "usage: [target_dir] [output_name] [file_type: smt | ans | feature]"
    exit 1
fi

path=$(realpath $1)
echo "Generate filelist to ${path}"

if [ $3 == 'smt' ]; then
    find $path -name '*.smt2' | sort > $2"_smt.filelist"
elif [ $3 == 'ans' ]; then
    find $path -name '*.result' | sort > $2"_ans.filelist"
elif [ $3 == 'feature' ]; then
    find $path -name '*.features' | sort > $2".featurelist"
fi
