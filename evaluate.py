#!/usr/bin/env python2

import os
import sys
import time
import json
import signal
import logging
import threading

import config

from IPython import embed
from easyprocess import EasyProcess

from datetime import datetime
from multiprocessing import Queue

from six.moves import cStringIO
from pysmt.shortcuts import Solver
from pysmt.smtlib.parser import SmtLibParser
from pysmt.smtlib.script import evaluate_command
from pysmt.smtlib.commands import CHECK_SAT, GET_VALUE


TIMEOUT = False

def job(script, solver, q):

    log = script.evaluate(solver)
    q.put(log)
    return 
    
def handler(signum, frame):

    global TIMEOUT
    TIMEOUT = True

    print "(Time out!)", 
    raise Exception("End of time")

class SolverEvaluator(object):

    timeout = 100

    @property
    def logger(self):
        name = self.__class__.__name__
        return logging.getLogger(name)

    def __init__(self, timeout=100):
    
        self.timeout = timeout
        self.solver_list= config.solver_list
        self.supported_logic = config.supported_logic
        self.solve_script_path = '/home/enhancing-se/enhancing-se/PathConstraintClassifier/solve.py'

        return

    def _evaluate(self, solver_name=None, logic=None, filename=None): #script=None):

        global TIMEOUT

        result = {}
#        if solver_name == None or script == None:
        if solver_name == None or filename == None:
            return result
        
        s = time.time()
        result = EasyProcess(
            'python {} {} {}'.format(
                self.solve_script_path,
                filename, 
                solver_name
            )
        ).call(timeout=self.timeout + 2).stdout
        e = time.time()

        if len(result) == 0:

#            result = json.load(open('/tmp/temp'))
            if len(result) == 0:
                result = {
                    'time': e - s,
                    'log': [],
                    'error': False
                }

#            os.remove('/tmp/temp')

        else:
            result = json.loads(result)

        """
        with Solver(name=solver_name) as solver:
       
            try:
                TIMEOUT = False
                signal.alarm(self.timeout)
                s = time.time()
                log = script.evaluate(solver)
                e = time.time()
                signal.alarm(0)
            except Exception as exc:
    
                signal.alarm(0)
                e = time.time()
                if TIMEOUT:
                    s = 0
                    e = self.timeout
                log = []
         """

#        result['time'] = e - s
        for cmd, value in result['log']:
            if cmd == CHECK_SAT:
                result['status'] = 'sat' if value else 'unsat'
                print '(%s)' % result['status'],
                break
        """
        if result.get('status') == None:
            if result['error']:
                result['status'] = 'error'
        """
        return result

    def load(self, output_file):

        if os.path.exists(output_file):
            return json.load(open(output_file, 'r'))
        return {}

    def save(self, output_file, next_index, total_result):

        output = json.load(open(output_file, 'r')) if os.path.exists(output_file) else {}
        output['next'] = next_index
        if 'result' not in output:
            output['result'] = []
        output['result'] += total_result
        
        with open(output_file, 'w') as o:
            o.write(json.dumps(output, indent=4))
        return

    def evaluate(self, filelist=None, dataset_name="default_dataset_name", output_file_name=None):

        filelist = [] if filelist == None else filelist
        if output_file_name == None:
            output_file_name = datetime.now().strftime("%Y-%m-%d_%H%M%S.answer")
#        else:
#            output_file_name = "{}.answer".format(dataset_name)

        signal.signal(signal.SIGALRM, handler)

        parser = SmtLibParser()
        length = len(filelist)
        output = self.load(output_file_name)
        next_index = output.get('next', 0)
        try:
            total_result = output.get('result', [])
            total_file_num = len(filelist)
            for index, file_path in enumerate(filelist):

                print '[{}] [{} / {}]'.format(
                    datetime.now().strftime("%H:%M:%S"), 
                    index+1, 
                    total_file_num
                ), 

                if index < next_index:
                    print ''
                    continue

#                smt_file = open(file_path, 'r').read().replace("\r", "")
#                script = parser.get_script(cStringIO(smt_file))
                
                solving_result = {}
                for solver in self.solver_list:
                    try:
                        print solver,
                        solving_result[solver] = self._evaluate(
                            solver_name=solver,
                            logic=None,
#                            script=script
                            filename=file_path
                        )
                    except Exception as exc:
                        self.logger.error(
                            "[{}] Evaluate fail. (solver: {})".format(str(index), solver)
                        )
                        solving_result[solver] = {}
                        print (exc)
#                        raise
                        
                total_result.append({
                    "id": index,
                    "file_path": file_path,
#                    "result": solving_result
#                    "result": [str([
                    "result": [{
                        "error": solving_result[s].get("error", ''),
                        "answer": solving_result[s].get("status", ''), 
                        "time": solving_result[s].get('time', -1)
                        } for s in self.solver_list
                    ]
                })
                print ''

                if len(total_result) % 5 == 0 and len(total_result) != 1:
                    self.logger.info(
                        "Backup results. [process: {} / {}]".format(index+1, length)
                    )
                    self.save(output_file_name, index+1, total_result)
                    total_result = []
                    self.logger.info("save done.")
        except:
            self.logger.info("Error occurs, stop..\n")

        self.save(output_file_name, total_result[-1]["id"] + 1, total_result)
        self.logger.info("save done.")
        return output_file_name


class AnswerEvaluator(object):

    def __init__(self):

        return

    def transform(self, ans):

        ans['result']  = [[eval(y) for y in x['result']] for x in ans['result']]

        return ans

    def analysis(self, ans, tag=None):

        res = ans['result']
        if tag == None:
            all_solving_time = [[y for y in x['result']] for x in res]
        else:
            all_solving_time = [[y for y in x['result']] for x in res if x['file_path'].find(tag) != -1]
        
        error = [0 for i in range(7)]
        unknown = [0 for i in range(7)]
        solvable = [0 for i in range(7)]
        best = [0 for i in range(7)]
        effective_solving_time = [[] for i in range(7)]
        solving_time = [[] for i in range(7)]
        timeout = [0 for i in range(7)]
        best_time = []
        non_solvable_count = 0

        for time_vector in all_solving_time:
            
            for i in range(7):

                solving_time[i].append(time_vector[i]['time'])
                if time_vector[i]['answer']:
                    solvable[i] += 1
                    effective_solving_time[i].append(time_vector[i]['time'])
                else:
                    
                    if time_vector[i]['error']: 
                        error[i] += 1
                    elif time_vector[i]['time'] >= 100.0:
                        timeout[i] += 1
                    else:
                        unknown[i] += 1

            _solvable = [(i, time_vector[i]['time']) for i in range(7) if time_vector[i]['answer']]
            _solvable.sort(key=lambda x: x[1])
            
            if _solvable:
                best[_solvable[0][0]] += 1
                best_time.append(time_vector[_solvable[0][0]]['time'])
            else:
                non_solvable_count += 1
                print json.dumps(time_vector, indent=4)

#            print best, solvable, solving_time_list

        return (
            best, 
            solvable, 
            error, 
            timeout, 
            effective_solving_time, 
            solving_time, 
            all_solving_time,
            best_time,
            non_solvable_count
        )


class PredictionEvaluator(object):

    @property
    def logger(self):
        name = self.__class__.__name__
        return logging.getLogger(name)

    def __init__(self):
 
        self.solver_list = config.solver_list

        return

    def accuracy(self, answer, predict):

        acc_count = sum([1 for x in zip(answer, predict) if int(x[0]) == int(x[1])]) 

        return float(acc_count) / len(predict)
    
    def get_answer(self, solver_result):

        best = []
        for i, sr in enumerate(solver_result):

            solvable = sorted([
                (j, s['time']) for j, s in enumerate(solver_result[i]) if s['answer']
            ], key=lambda x: x[1])

            if len(solvable) != 0:
                best.append(solvable[0][0])
            else:
                support = sorted([
                    (j, s['time']) for j, s in enumerate(solver_result[i]) if not s['error']
                ])
                if len(support) != 0:
                    best.append(support[0][0])
                else:
                    print 'something happen.'
    
        assert(len(best) == len(solver_result))
        return best

    def update_evaluated_result(self, res1, res2):
 
        ret = res1.copy()
        for k in res2:  
            
            if k in ret:

                if isinstance(ret[k], int) or isinstance(ret[k], float):
                    ret[k] += res2[k]
                elif isinstance(ret[k], list):
                    for x in res2[k]:
                        if x not in ret[k]:
                            ret[k].append(x)
                elif isinstance(ret[k], dict):
                    ret[k] = self.update_evaluated_result(ret[k], res2[k])
                else:
                    print "unknown data type: ", type(res2[k])       
            else:
#                print k, "not in ret"
                if isinstance(res2[k], dict):
                    ret[k] = res2[k].copy()
                else:
                    ret[k] = res2[k]

        return ret      

    def evaluate_prebackup_solver_result(self, sol_res, activate_time):

        ret = {
            "timeout_count": 0,
            "solvable_count": 0,
            "nonsolvable_count": 0,
            "unknown_count": 0, 
            "error_count": 0,
            "total_count": 0,

            "possible_solve_result": [],

            "timeout_time": 0.0,
            "error_time": 0.0,
            "solving_time": 0.0,
            "nonsolving_time": 0.0,
            "total_time": 0.0,
            "activate_backup_by_error": 0,
            "activate_backup_by_timeout": 0,
            "activate_backup_by_noanswer": 0,
            "activate_backup_by_error_time": 0.0,
            "activate_backup_by_timeout_time": 0.0,
            "activate_backup_by_noanswer_time": 0.0,
        }

        if sol_res['error']:    
            ret["activate_backup_by_error"] += 1
            ret["activate_backup_by_error_time"] += sol_res['time']
        elif sol_res['time'] < activate_time:
            ret["activate_backup_by_noanswer"] += 1
            ret["activate_backup_by_noanswer_time"] += sol_res['time']
        else:
            ret["activate_backup_by_timeout"] += 1
            ret["activate_backup_by_timeout_time"] += activate_time

        ret["activate_backup_count"] = ret["activate_backup_by_timeout"] + ret["activate_backup_by_error"] + ret["activate_backup_by_noanswer"]
        ret["activate_backup_time"] = ret["activate_backup_by_timeout_time"] + ret["activate_backup_by_error_time"] + ret["activate_backup_by_noanswer_time"]
        ret["total_time"] = ret["activate_backup_time"]
        return ret       

    def evaluate_solver_result(self, sol_res, timeout, is_backup_solver=False):

        ret = {
            "timeout_count": 0,
            "solvable_count": 0,
            "nonsolvable_count": 0,
            "unknown_count": 0, 
            "error_count": 0,
            "total_count": 0,

            "possible_solve_result": [],

            "timeout_time": 0.0,
            "error_time": 0.0,
            "solving_time": 0.0,
            "nonsolving_time": 0.0,
            "total_time": 0.0,
        }

        if sol_res['error']:
            ret["nonsolvable_count"] += 1
            ret["nonsolving_time"] += sol_res['time']
            ret["error_time"] += sol_res['time']
            ret["error_count"] += 1
        else:
            
            if sol_res['answer']:
                ret["solvable_count"] += 1
                ret["solving_time"] += sol_res['time']

                if sol_res['answer'] not in ret["possible_solve_result"]:
                    ret["possible_solve_result"].append(sol_res['answer'])
            else:
                
                ret["nonsolvable_count"] += 1
                ret["nonsolving_time"] += sol_res['time']
                ret["timeout_time"] += sol_res['time']
                if sol_res['time'] >= timeout:
                    ret["timeout_count"] += 1

                else:
                    ret["unknown_count"] += 1

        ret["total_time"] += ret["solving_time"] + ret["nonsolving_time"]
        ret["total_count"] += ret["solvable_count"] + ret["nonsolvable_count"] 
#        ret["solvable_rate"] = float(ret["solvable_count"]) / ret["total_count"] if ret["total_count"] != 0 else 0
        return ret

    def time(self, data, predict, timeout = 100, 
             enable_backup_solver=False, backup_solver_id=6, activate_backup_solver_timeout=1, group_index=None):

        print "enable_backup_solver:", enable_backup_solver
        print "backup_solver_id:", backup_solver_id
        print "activate_backup_solver_timeout:", activate_backup_solver_timeout

        solver_result = data['z']

        total_count = len(predict)
#        best = self.get_answer(solver_result)
        
        general_result = {}
        prebackup_result = {}
        backup_result = {}
        no_answer = 0

        group_result = {}
        do_partition = False if group_index == None else True
        for i, solver_id in enumerate(predict):

            if do_partition:
                if i not in group_index:
                    continue

            solve_count = sum([1 for sr in solver_result[i] if sr['answer']])
            if solve_count == 0:
                no_answer += 1
                continue

            s_id = int(solver_id)
            res = solver_result[i][s_id]
            
            activate_backup_solver = False

            if enable_backup_solver:
                
                if  res['error'] or \
                   (res['time'] < activate_backup_solver_timeout and len(res['answer']) == 0) or \
                    res['time'] >= activate_backup_solver_timeout:
                    activate_backup_solver = True

                    print "##################"
                    print "activate backup solver!!!!"
                    print solver_result[i], res

                    if s_id == backup_solver_id:
                        activate_backup_solver = False

                        print ">>> Cancel activate!!"

                if activate_backup_solver:

                    bs_res = solver_result[i][backup_solver_id]
                    if bs_res['time'] > timeout:
                        print "pre:", res
                        print "backup:", bs_res
                        print solver_result[i]
                        print data['t'][i]

                    e_res = self.evaluate_prebackup_solver_result(res, activate_backup_solver_timeout)
                    e_bs_res = self.evaluate_solver_result(bs_res, timeout, is_backup_solver=True)

                    prebackup_result = self.update_evaluated_result(
                        prebackup_result, e_res
                    )
                    
#                    print json.dumps(general_result, indent=4)

                    general_result = self.update_evaluated_result(
                        general_result, e_res
                    )
                    
                    """
                    print json.dumps(general_result, indent=4)
                    raw_input('press...')
                    """

                    backup_result = self.update_evaluated_result(
                        backup_result, e_bs_res
                    )  
                    general_result = self.update_evaluated_result(
                        general_result, e_bs_res
                    )
                    
            if not activate_backup_solver:
                general_result = self.update_evaluated_result(
                    general_result, 
                    self.evaluate_solver_result(res, timeout, is_backup_solver=False)
                )

        return {
            "general": general_result,
            "backup": backup_result,
            "preback": prebackup_result,
            "no_answer": no_answer
        }

    def time_partition(self, data, predict, timeout = 100, 
             enable_backup_solver=False, backup_solver_id=6, activate_backup_solver_timeout=1,
             partition_target=None):

        save_file = 'index_file_{}.json'.format(str(len(data['y'])))
        if os.path.exists(save_file):
            partition = json.load(open(save_file))
            print "use local index file."
        else:

            if partition_target == None:
                partition_target = {'':"benchmark", 'angr':'angr', 'klee':'klee'}

            partition = dict([(v, []) for k, v  in partition_target.iteritems()])
            for i, tags in enumerate(data['t']):
                
                for p_tag, p_name in partition_target.iteritems():
                    if tags[0] == p_tag:
                        partition[p_name].append(i)
            
            json.dump(partition, open(save_file, 'w'))

        group_result = {}
        for p_name, index in partition.iteritems():
            
            group_result[p_name] = self.time(
                data, 
                predict, 
                timeout, 
                enable_backup_solver, 
                backup_solver_id, 
                activate_backup_solver_timeout, 
                index
            )

#        print group_result


#        print json.dumps(group_result, indent=4)
#        raw_input('press...')

        aggregate = {}
        for k, v in group_result.iteritems():
            aggregate = self.update_evaluated_result(aggregate, v)
        group_result['all'] = aggregate

#        print group_result
#        raw_input("press...")
        return group_result
                    
    def distribution(self, predict, ans):
    
        count = {}
        double_count = dict([(x , {}) for x in range(7)])
        for i, solver_id in enumerate(predict):
            count[int(solver_id)] = count.get(int(solver_id), 0) + 1
            
            if solver_id not in double_count[ans[i]]:
                double_count[ans[i]][solver_id] = 0
            double_count[ans[i]][solver_id] += 1
            

        return count, double_count


    def logic_distribution(self, predict, data):


        count = {}
        for i, solver_id in enumerate(predict):

            logic = data['m'][i][0]
            if logic not in count:
                count[logic] = {'accurate': 0, 'total': 0}
            if int(solver_id) == int(data['y'][i]):
                count[logic]['accurate'] += 1
            count[logic]['total'] += 1
            
        for logic in count:
            count[logic]['rate'] = float(count[logic]['accurate']) / count[logic]['total']

        return count


def main():

    logging.basicConfig(level=logging.INFO)

    filelist = open(sys.argv[1], 'r').read().strip().split('\n')
    dataset_name = os.path.basename(sys.argv[1]).split('.')[0]
    evaluator = SolverEvaluator()
    output_file_name = "{}.answer".format(sys.argv[2])
    evaluator.evaluate(filelist, dataset_name, output_file_name)

    return

if __name__ == '__main__':

	main()

