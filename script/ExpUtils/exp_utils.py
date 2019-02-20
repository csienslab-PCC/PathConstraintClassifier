#!/usr/bin/env python

import os
import sys
import json

from IPython import embed

sys.path.append('../../../')

import numpy as np

from argparse import ArgumentParser

from PathConstraintClassifier.model import *
#from PathConstraintClassifier.train import TrainingManager
from PathConstraintClassifier.partition import DataPartitioner, DataGrouper
from PathConstraintClassifier.evaluate import PredictionEvaluator
import PathConstraintClassifier.feature_group as FI

import PathConstraintClassifier.config as pcc_config

import gspread_tools as GST

def compute_solvable_rate(x):
 
    if 'total_count' not in x or 'solvable_count' not in x:
        return 0

    return 100 * float(x['solvable_count']) / x['total_count'] if x['total_count'] != 0 else 0.0

def upload_result(data):

    url = "https://docs.google.com/spreadsheets/d/1szNjv5h-M2pqbrzCUDzwgyd94FZbtjyAxFLBkRzLUPo/edit#gid=1367631075"
    sh = GST.get_spreadsheet(url, "thesis.json")
    ws = GST.get_worksheet(sh, "upload_result")[0]


    g_list = [
        "solvable_count", "solvable_rate", "timeout_count", "error_count", "nonsolvable_count", "total_count",
        "activate_backup_by_timeout", "activate_backup_by_error", "activate_backup_by_noanswer", "activate_backup_count",
        "solving_time", "timeout_time", "error_time", "nonsolving_time", 
        "activate_backup_by_timeout_time", "activate_backup_by_error_time", "activate_backup_by_noanswer_time", "activate_backup_time",
        "total_time"
    ]
    p_list = [
        "activate_backup_by_timeout", "activate_backup_by_error", "activate_backup_by_noanswer", "activate_backup_count",
        "activate_backup_by_timeout_time", "activate_backup_by_error_time", "activate_backup_by_noanswer_time", "activate_backup_time",
    ]

    b_list = [
        "solvable_count", "solvable_rate", "timeout_count", "error_count", "nonsolvable_count", "total_count",
        "solving_time", "timeout_time", "error_time", "nonsolving_time", 
        "total_time"
    ]

    o_list = [
        "predict", "avg_predict", "logic_selection", "avg_logic_selection"
    ]

    to_upload = []
    for p_name, p_data in data['data'].iteritems(): 

        p_data['general']['solvable_rate'] = compute_solvable_rate(p_data['general'])
        p_data['backup']['solvable_rate'] = compute_solvable_rate(p_data['general'])
        general = [p_data['general'].get(x ,"-") for x in g_list]
        pre = [p_data['preback'].get(x, "-") for x in p_list]
        bak = [p_data['backup'].get(x, "-") for x in b_list]
        overhead = [data['overhead'][k] for k in o_list]


        merge = general + overhead + pre + bak

        to_upload.append([
            data['model_name'],
            p_name,
            data['logic_selection'],
            data['backup_solver'],
            data['activate_bks_time'],
        ] + merge)

    GST.upload_data(ws, to_upload, mode="append", tail_space=1)

    return 

def jprint(x):
    print json.dumps(x, indent=4)

def load(x):
    return json.load(open(x))

def load_data(x):
    return np.load(open(x))

def remove_dummy(data):

    indices = np.array([])
    for i, v in enumerate(data['y']):

        if v == 'dummy' or int(float(v)) == 7:
#        if int(float(v)) == 7:
            indices = np.append(indices, i)
#            print data['z'][i]
 
#            zz = [eval(x) for x in data['z'][i]]

#            assert(int(sum([float(x[1] if len(x) > 1 else x) for x in zz])) == 1000 * len(zz))
#            print int(sum([float(x) for x in data['z'][i]])),  1000*len(data['z'][i])
    
    ret = {
        'x': np.delete(data['x'], indices.astype(int), axis=0),
        'y': np.delete(data['y'], indices.astype(int)),
        'z': np.delete(data['z'], indices.astype(int), axis=0),
        'm': np.delete(data['m'], indices.astype(int), axis=0),
        't': np.delete(data['t'], indices.astype(int), axis=0)
    }
    return ret



def _gen(config, output):

    if os.path.exists(output):
        ans = raw_input("output exists, continue?\n >> ")
        if ans != 'y':
            print 'exit'
            exit()
    else:
        os.mkdir(output)

    config = load(config)
    DP = DataPartitioner()
    data = DP.load(config)
    train, test = DP.partition(data, output=output)
    
    return



def gen(config_path, output, extra):

#    output = sys.argv[2]
    if os.path.exists(output):
        print 'output exists.'
#        if '-f' not in sys.argv:
#            return
    else:
        os.mkdir(output)

 #   config = load(os.path.join(output, 'config.json'))
    config = load(config_path)
#    DP = DataPartitioner()
    DP = DataGrouper(config)

    
#    train, test = DP.partition()
    train, test = DP.partition_by_key(partition_key=[
        ['angr', 'dd'],
        ['angr', 'expr'],
        ['klee', 'mknod'],
        ['klee', 'dd'],
        ['klee', 'printf'],
    ])
    if 'remove-dummy' == extra:

        print "length: ", len(train['x']), len(test['x'])
        train = remove_dummy(train)
        test = remove_dummy(test)
        print "after remove: ", len(train['x']), len(test['x'])

    DP.output(train, os.path.join(output, 'train'))
    DP.output(test, os.path.join(output, 'test'))
    
    return   

def delete_features(data):
    
    return data

    new_x = []
    for xx in data['x']:
        xx[[FI.UNIQUE_SYMBOLS, FI.THEORY]] = [0, 0]
        new_x.append(xx)       

#    embed()

    ret = {}
    for l in data.keys():
        ret[l] = data[l]

    ret['x'] = np.array(new_x)

    return ret



def train(input_file, model_info):
    
    model_list = ["DNN", "DNN-alpha", "DNN-beta", "DNN-gamma", "RandomForest" , "CostSensitive"]

#    input_dir = sys.argv[2]
    model_id, model_dir = model_info.split(':')
    model_name = model_list[int(model_id)]
    model_path = os.path.join(model_dir, model_name + '.model')

    print input_file
    print model_path

    train = delete_features(load_data(input_file))

    print 'ok'

    MR = ModelRegister()
    model = MR.get_model(model_name)
    model.train(train)
    model.save(model_path)
    
    return

def test_upload(input_file, model_info, logic_selection, activate_backup_solver_timeout, solver_id):

    model_id, model_path = model_info.split(':')

    model_list = ["DNN", "DNN-alpha", "DNN-beta", "DNN-gamma", "Solver", "Best"]
    model_name = model_list[int(model_id)]


    print input_file
    print model_name, model_path

#    test = load_data(os.path.join(input_dir, 'test.train'))
    test = delete_features(load_data(input_file))

    MR = ModelRegister()
    print model_name

    if model_name != "Best":
        model = MR.get_model(model_name)
        if model_name == 'Solver':
#            solver_id = raw_input("solver id: ")
            model.solver_id = int(solver_id)
            model_name = pcc_config.solver_list[int(solver_id)]
        else:
#            model.load(os.path.join(input_dir, model_name + '.model'))
            model.load(model_path)

        if logic_selection:
            print 'enable logic_selection'
            model = EnhancedModel(model)

    activate_time = activate_backup_solver_timeout
    if activate_backup_solver_timeout > 0:
        enable_backup_solver = True
        print 'enable backup_solver'
    else:
        enable_backup_solver = False
    print 'ok'

    PE = PredictionEvaluator()

    store_ans_file = "answer_{}_{}_b{}_l{}.json".format(model_name, str(len(test['x'])), enable_backup_solver, logic_selection)
    if os.path.exists(store_ans_file) and False:
        ans = json.load(open(store_ans_file))
        print 'use local ans file.'
    else:
        if model_name == 'Best':
            ans = PE.get_answer(test['z'])
        else:
#            embed()
            ans = list(model.predict(test))
        json.dump(ans, open(store_ans_file, "w"))

    if model_name != 'Best': 
        prediction_time = model.prediction_time
        avg_prediction_time = model.avg_prediction_time
    else:
        prediction_time = 0
        avg_prediction_time = 0

    logic_selection_time = model.logic_selection_time if logic_selection else 0
    avg_logic_selection_time = model.avg_logic_selection_time if logic_selection else 0

    time_p = PE.time_partition(
        test, 
        ans, 
        enable_backup_solver=enable_backup_solver, 
        backup_solver_id=1, 
        activate_backup_solver_timeout=activate_time
    )
 
    jprint(time_p)

    upload = {
        "model_name": model_name,
        "logic_selection": logic_selection,
        "backup_solver": enable_backup_solver,
        "activate_bks_time": activate_time,
        "data": time_p,
        "overhead": {
            "predict": prediction_time,
            "avg_predict": avg_prediction_time,
            "logic_selection": logic_selection_time,
            "avg_logic_selection": avg_logic_selection_time
        }
    }

#    json.dump(upload, "mdl:{}-bks:{}-ls{}-bkst:{}".format(model_name, enable_backup_solver, logic_selection, activate_time), indent=4)

    upload_result(upload)
    return 



def test(input_dir, model_id, logic_selection, activate_backup_solver_timeout, solver=None):
 
    # "RandomForest" , "CostSensitive",
    model_list = ["DNN", "DNN-alpha", "DNN-beta", "DNN-gamma", "Solver"]
    
#    input_dir = sys.argv[2]
    model_name = model_list[int(model_id)]

    print input_dir
    print model_name

    train = load_data(os.path.join(input_dir, 'train.train'))
    test = load_data(os.path.join(input_dir, 'test.train'))

    MR = ModelRegister()
    print model_name

    model = MR.get_model(model_name)
    if model_name == 'Solver':
#        solver_id = raw_input("solver id: ")
        solver_id = int(solver)
        model.solver_id = solver_id
    else:
        model.load(os.path.join(input_dir, model_name + '.model'))

    if logic_selection:
        print 'enable logic_selection'
        model = EnhancedModel(model)

    activate_time = activate_backup_solver_timeout
    if activate_backup_solver_timeout > 0:
        enable_backup_solver = True
        print 'enable backup_solver'
    else:
        enable_backup_solver = False

    print 'ok'

#    ans1 = model.predict(train)
#    print 'predict train done.'
    ans2 = model.predict(test)
    print 'predict test done.'

    PE = PredictionEvaluator()

#    acc1 = PE.accuracy(train['y'], ans1)
#    acc2 = PE.accuracy(test['y'], ans2)

#    time_p1 = PE.time_partition(train, ans1, enable_backup_solver=enable_backup_solver, backup_solver_id=6, activate_backup_solver_timeout=10)
#    timr_p2 = PE.time_partition(test, ans2, enable_backup_solver=enable_backup_solver, backup_solver_id=6, activate_backup_solver_timeout=10)


#    time1 = PE.time(train, ans1, enable_backup_solver=enable_backup_solver, backup_solver_id=6, activate_backup_solver_timeout=activate_time)
#    time1_p = PE.time_partition(train, ans1, enable_backup_solver=enable_backup_solver, backup_solver_id=6, activate_backup_solver_timeout=activate_time)
#    best_time1 = PE.time(train['z'], train['y'])

#    best1 = PE.get_answer(train['z'])
#    best_time1 = PE.time(train, best1, enable_backup_solver=enable_backup_solver, backup_solver_id=6, activate_backup_solver_timeout=activate_time) 
#    best_time1_p = PE.time_partition(train, best1, enable_backup_solver=enable_backup_solver, backup_solver_id=6, activate_backup_solver_timeout=activate_time) 
#    print len(ans1)
#    print len(best1)

     
#    time2 = PE.time(test, ans2, enable_backup_solver=enable_backup_solver, backup_solver_id=6, activate_backup_solver_timeout=activate_time)
    time2_p = PE.time_partition(test, ans2, enable_backup_solver=enable_backup_solver, backup_solver_id=6, activate_backup_solver_timeout=activate_time)
#    best_time2 = PE.time(test['z'], test['y'])

#    best2 = PE.get_answer(test['z'])
#    best_time2 = PE.time(test, best2, enable_backup_solver=enable_backup_solver, backup_solver_id=6, activate_backup_solver_timeout=activate_time)
#    best_time2_p = PE.time_partition(test, best2, enable_backup_solver=enable_backup_solver, backup_solver_id=6, activate_backup_solver_timeout=activate_time)
#    print len(ans2)
#    print len(best2)
   
        
    upload = {
        "model_name": model_name,
        "logic_selection": logic_selection,
        "backup_solver": enable_backup_solver,
        "activate_bks_time": activate_time,
        "data": time2_p
    }

    upload_result(upload)
    return 

    print "acc1:", acc1
    print "acc2:", acc2
    print ''

    """
    del time1['best']
    del best_time1['best']
    del time2['best']
    del best_time2['best']
    """
    print "time1:", 
    jprint(time1)
    print "best_time1:",  
    jprint(best_time1)
    print ''
    jprint(time1_p)
    print ''
    jprint(best_time1_p)

    print "time2:",
    jprint(time2)
    print "best_time2:", 
    jprint(best_time2)
    print ''
    jprint(time2_p)
    print ''
    jprint(best_time2_p)

    count, double = PE.distribution(ans1, train['y']) 
    print "dist1", count
    for i in double:
        print "{}".format(i), double[i]

    print "best_dist1", PE.distribution(train['y'], train['y'])
    print ''

    count, double = PE.distribution(ans2, test['y']) 
    print "dist2", count
    for i in double:
        print "{}".format(i), double[i]

    print "best_dist2", PE.distribution(test['y'], test['y'])

#    logic_dist = PE.logic_distribution(ans1, train)
#    print "logic1:" 
#    print json.dumps(dict([(x, "%.4f (%4d)" % (logic_dist[x]['rate'], logic_dist[x]['total'])) for x in logic_dist]), indent=4)
#    for logic in logic_dist:
#        print logic, logic_dist[logic]['rate']


#    logic_dist = PE.logic_distribution(ans2, test)
#    print "logic2:",
#    print json.dumps(dict([(x, "%.4f (%4d)" % (logic_dist[x]['rate'], logic_dist[x]['total'])) for x in logic_dist]), indent=4)
#    for logic in logic_dist:
#        print logic, logic_dist[logic]['rate']

    return

def evaluate():

    solver_list = config.solver_list

    f = np.load(sys.argv[2])
    if sys.argv[3] in solver_list:
        idx = solver_list.index(sys.argv[3])
        ans = [idx for _ in range(len(f['y']))]
    else:
        ans = open(sys.argv[3]).read().strip().split('\n')
        ans = ans[:len(f['y'])]

    PE = PredictionEvaluator()
    
    acc = PE.accuracy(f['y'], ans)
    time = PE.time(f['z'], ans)
    best_time = PE.time(f['z'], f['y'])
    
    print "acc:", acc
    print "time:", time
    print "best_time", best_time
    print PE.distribution(ans, f['y'])
    print PE.distribution(f['y'], f['y'])

    return


def main():

    parser = ArgumentParser()    

    parser.add_argument("-a", "--action", dest="action", default="")
    parser.add_argument("-c", "--config", dest="config", default="")
    parser.add_argument("-o", "--output", dest="output", default="")
    parser.add_argument("-i", "--input", dest="input", default="")
    parser.add_argument("-m", "--model", dest="model", default="")
    parser.add_argument("-s", "--solver", dest="solver", default=0)
    parser.add_argument("-e", "--extra", dest="extra_options", default="")
    parser.add_argument("-logslct", "--logic-selection", dest="logic_selection", default=False)
    parser.add_argument("-backup", "--backup-solver", dest="backup_solver", default=False)

    args = parser.parse_args()
    action = args.action
    config = args.config
    input_path = args.input
    output_path = args.output
    model_name = args.model
    solver = args.solver
    extra = args.extra_options
    logic_selection = bool(args.logic_selection)
    backup_solver = int(args.backup_solver)

    print 'enable logic-selection:', logic_selection
    print 'enable backup-solver:', backup_solver
    
    if action == 'gen':
        gen(config, output_path, extra)
    elif action == 'train':
        train(input_path, model_name)
    elif action == 'test':
        test(input_path, model_name, logic_selection, backup_solver, solver)
    elif action == 'eval':
        evaluate()
    elif action == 'test_upload':
        test_upload(input_path, model_name, logic_selection, backup_solver, solver)

    return



if __name__ == '__main__':

    main()
