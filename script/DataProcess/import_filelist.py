#!/usr/bin/env python

import os
import sys
import json


def main():

    if len(sys.argv) < 3:
        print "usage: {} [filelist_file] [filelist_name]".format(sys.argv[0])
        exit()

    if len(sys.argv) == 4:
        output = sys.argv[3]
    else:
        output = 'filelist.json'

    print output
    filelist = json.load(open(output, 'r'))
    import_filelist = open(sys.argv[1], 'r').read().split('\n')[:-1]
    filelist[sys.argv[2]] = import_filelist

    with open(output, 'w') as output:
        output.write(json.dumps(filelist, indent=4))
    
    return

if __name__ == '__main__':

    main()
