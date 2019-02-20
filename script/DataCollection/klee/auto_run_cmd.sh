
# Usage:
#   $ ./auto_run_cmd.sh [target_bc_filelist] [script_to_be_execute]
#
# Example:
#   $ ./auto_run_cmd.sh ./test.filelist ./run_klee.py
#   or
#   $ ./auto_run_cmd.sh ./test.filelist ./extract_query.py
#

cat $1 | xargs -n1 $2
