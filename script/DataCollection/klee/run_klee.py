#!/usr/bin/env python2

import os
import sys
import config

from datetime import datetime

def main():
    
    if config.log_path:
        log_file = open(config.log_path, 'a')
    else:
        log_file = sys.stdout

    default_args = "--sym-args 0 1 10 --sym-args 0 2 2 --sym-files 1 8 --sym-stdin 8 --sym-stdout"
    sym_args_table = {
       "dd": "--sym-args 0 3 10 --sym-files 1 8 --sym-stdin 8 --sym-stdout",
       "dircolors": "--sym-args 0 3 10 --sym-files 2 12 --sym-stdin 12 --sym-stdout",
       "echo": "--sym-args 0 4 300 --sym-files 2 30 --sym-stdin 30 --sym-stdout",
       "expr": "--sym-args 0 1 10 --sym-args 0 3 2 --sym-stdout",
       "mknod": "--sym-args 0 1 10 --sym-args 0 3 2 --sym-files 1 8 --sym-stdin 8 --sym-stdout",
       "od": "--sym-args 0 3 10 --sym-files 2 12 --sym-stdin 12 --sym-stdout",
       "pathchk": "--sym-args 0 1 2 --sym-args 0 1 300 --sym-files 1 8 --sym-stdin 8 --sym-stdout",
       "printf": "--sym-args 0 3 10 --sym-files 2 12 --sym-stdin 12 --sym-stdout",
    }

    default_command = "klee --simplify-sym-indices -use-query-log=solver:smt2 --disable-inlining --optimize --use-forked-solver --libc=uclibc --posix-runtime --allow-external-sym-calls --run-in=/tmp/sandbox --max-sym-array-size=4096 --max-instruction-time=30. --watchdog --max-memory-inhibit=false --max-static-cpfork-pct=1 --switch-type=internal --search=random-path --search=nurs:covnew --use-batching-search --batch-instructions=10000 -min-query-time-to-log={min_log_time} -no-output -max-solver-time={max_solver_time} --max-time={max_time} --output-dir={output} {target_bc} {sym_args}"

    min_log_time = 500
    max_solver_time = 100
    target_bc = sys.argv[1]
    target_bc_path = "{}/{}.bc".format(config.bc_file_dir, sys.argv[1])
    time = config.execution_time
    output = "{}/{}-{}".format(config.klee_data_output_dir, target_bc, time)
#    sym_args = sym_args_table[target]
    sym_args = sym_args_table.get(target_bc, default_args)


    command = default_command.format(
        min_log_time=min_log_time,
        max_solver_time=max_solver_time,
        max_time=time, 
        output=output, 
        target_bc=target_bc_path,
        sym_args=sym_args
    )

    log_file.write("\n[{}] Start execute command: \n{}\n".format(str(datetime.now()), command))
    os.system(command)
    log_file.write("\n[{}] Done.\n".format(str(datetime.now())))

if __name__ == '__main__':

    main()
