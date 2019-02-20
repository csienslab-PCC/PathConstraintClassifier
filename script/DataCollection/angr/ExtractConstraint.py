import os
import sys
import angr
import logging
import claripy
from z3 import Solver
from claripy.backends.backend_z3 import BackendZ3

from IPython import embed

solver = Solver()
bk_z3 = BackendZ3()

def call_exit():
    exit()

def main():

    target_bin = sys.argv[1]
    target_bin_name = os.path.basename(target_bin)
    if (len(sys.argv) > 2):
        output_dir = sys.argv[2]
    else:
        output_dir = None

    print output_dir

#    args = claripy.BVS('args', 8 * 16)
    arg1 = claripy.BVS('args', 8 * 100)
    arg2 = claripy.BVS('args', 8 * 100)
    arg3 = claripy.BVS('args', 8 * 100)
    arg4 = claripy.BVS('args', 8 * 300)
    proj = angr.Project(target_bin)
    state = proj.factory.entry_state(args=[arg1, arg2, arg3, arg4])
    simgr = proj.factory.simgr(state)

    level = 0
    while simgr.active:

        print 'level:', level
        level += 1

        print 'state_num:', len(simgr.active)
        print simgr.active
        print ''
        count = 0
        for s in simgr.active:

            solver = Solver()
            bsolver = BackendZ3()
            constr = s.solver.constraints
            print constr

            if constr:
                z3_constr = bsolver.convert_list(constr)
                solver.add(z3_constr)

                if (output_dir):
                    filename = "{}-{}-{}".format(target_bin_name, level, count)
                    output_path = os.path.join(output_dir, filename)
                    with open(output_path, "w") as f:
                        f.write(solver.to_smt2())
    #                logging.info("Output constraint to: " + output_path + "\n")
                    print "Output constraint to: " + output_path + "\n"
                else:
                    print sovler.to_smt2()
                count += 1

        simgr.step()

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    main()
