#!/usr/bin/env python2

import os
import sys
import config

from IPython import embed
from datetime import datetime


def main():

    target_bin = sys.argv[1]
    klee_data_dir = config.klee_data_output_dir
    time = config.execution_time

    target_bc_dir = "{}/{}-{}".format(klee_data_dir, target_bin, time)

    input_file  = os.path.join(target_bin_dir, "solver-queries.smt2")
    if not os.path.exists(input_file):
        return

    if config.query_data_output_dir:
        output_dir = config.query_data_output_dir
    else:
        output_dir = "./extracted_query_data_output/"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(input_file, 'r') as f:
        d = f.read()

    queries = d.split('\n\n')
 
    count = 0
    for i, q in enumerate(queries):
        
        if not q:
            continue

        output_file = os.path.join(output_dir, "query-{}.smt2".format("%05d" % i))

        with open(output_file, "w") as o:
            
            for cmd in q.split('\n'):
                
                if 'get-value' in cmd:
                    cmd = '; ' + cmd
                o.write(cmd + '\n')
        
        print('output: {}'.format(output_file))

    return


if __name__ == '__main__':
    
    main()
