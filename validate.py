
import os
import sys

import numpy as np

from IPython import embed
from feature import FeatureProcessor


def length_check(data, smt, ans):

    print len(data['x']), len(data['y']), len(data['z']), len(data['m'])
    print len(smt)
    print len(ans)
    assert len(smt) == len(ans)
    assert len(ans) == len(data['x'])
    assert len(data['x']) == len(data['y'])
    assert len(data['y']) == len(data['z'])
    assert len(data['z']) == len(data['m'])
    print 'length ok.'
    return  

def answer_check(data, smt, ans):

    length = len(ans)
    for i in range(length):


        ans1 = [float(x.split(' ')[-2]) for x in open(ans[i]).read().strip().split('\n')]
        ans2 = [float(x) for x in data['z'][i]]
        
        print "[{} / {}] ".format(i, length),
#        print ans1, ans2
        try:
            assert ans1 == ans2
            if int(sum(ans1)) != 7000:
                assert ans2.index(min(ans2)) == int(data['y'][i])
            print 'ok.'
        except:
            print ans1
            print ans2
            embed()

    print 'answer ok.'
    return

def validate(data):

    FE = FeatureProcessor()

    new_m = [None for _ in range(len(data['m']))]
    new_x = FE.process(data['x'])

    for i, v in enumerate(data['m']):
        new_m[i] = [v]
        
    new_m = np.array(new_m)

    np.savez(os.path.basename(sys.argv[1]), x=new_x, y=data['y'], z=data['z'], m=new_m)

    return

def main():

    data = np.load(open(sys.argv[1], 'rb'))
    smt = open(sys.argv[2], 'rb').read().strip().split('\n')
    ans = open(sys.argv[3], 'rb').read().strip().split('\n')

    length_check(data, smt, ans)
#    answer_check(data, smt, ans)
    validate(data)

#    embed()



    return

if __name__ == '__main__':
    
    main()
