
import sys
import time
import json

from six.moves import cStringIO
from pysmt.shortcuts import Solver
from pysmt.smtlib.parser import SmtLibParser


if __name__ == '__main__':

    filename = sys.argv[1]
    solver_name = sys.argv[2]
    smt_file = open(filename, 'r').read().replace('\r', "")

    parser = SmtLibParser()
    script = parser.get_script(cStringIO(smt_file))

#    with Solver(name=solver_name) as solver:

    solver = Solver(name=solver_name)
    error = False
    s = time.time()
    try:
        log = script.evaluate(solver)
        e = time.time()
    except:
        e = time.time()
        error = True    
        log = []
        
    """
    print json.dumps({
        'time' : e - s,
        'log' : log,
        'error' : error
    })
    """
    sys.stdout.write(json.dumps({
        'time' : e - s,
        'log' : log,
        'error' : error
    }))
    sys.stdout.flush()
    
    """
    with open('/tmp/temp', 'w') as f:
        f.write(json.dumps({
            'time' : e - s,
            'log' : log,
            'error' : error
        }))
        f.flush()
            
    time.sleep(1)
    """
