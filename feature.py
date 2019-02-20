#!/usr/bin/env python2

import os
import sys
import time
import math
import signal
import logging
import numpy as np

from argparse import ArgumentParser

import config
from utils import VariableGraph
#from feature_group import FeatureIndex as FI
import feature_group as FI

from IPython import embed
from datetime import datetime
from types import FunctionType
from six.moves import cStringIO

import pysmt.operators as pysmt_op

from pysmt import fnode
from pysmt import typing
from pysmt.formula import TemplateFormulaManager
from pysmt.environment import reset_env
from pysmt.environment import get_env
from pysmt.smtlib.parser import SmtLibParser
from pysmt.smtlib.script import evaluate_command
from pysmt.oracles import SizeOracle, AtomsOracle

class ExtractError(Exception):
    pass

class CacheError(Exception):
    pass

def handler(signum, frame):
    print "Time out!"
    raise Exception("end of time")


time_list = [0, 0, 0, 0, 0]
consecutive_time = []

class FeatureProcessor(object):

    """ Rescale value of features """

    @property
    def logger(self):
        name = self.__class__.__name__
        return logging.getLogger(name)      

    def __init__(self):
    
        self.inf_replace_value = np.finfo('float64').max

        return 

    def process_one(self, f):

        f[ f == np.inf ] = self.inf_replace_value
        
        for i, ff in enumerate(f):
            f[i] = math.log(ff + 1)

#        print f
#        f = np.log(np.array(f, dtype=np.float64) + 1)
        return f

    def process(self, all_features):

        for i, f in enumerate(all_features):
            f = self.process_one(f)
            all_features[i] = f

        return all_features

class FeatureCacheManager(object):

    """ Store features cache to reduce extraction time """

    @property
    def logger(self):
        name = self.__class__.__name__
        return logging.getLogger(name)

    def __init__(self, clean_threshold=100000):

        self._tree_cache = {}
        self._node_cache = {}
        self._variable_cache = {}

        self.formula_manager = get_env().formula_manager

        self.get_cache_key_count = {'tree':0, 'node':0}
        self.get_cache_key_time = {'tree':0, 'node':0}
        self.get_cache_key_by_attr = 0

        """
        self.clean_threshold = clean_threshold
        self._cache = {}
        self._hits = {}
        self._node_hits = {}
        
        self._size = 0
        self._node_size = 0
        self._time = 0
        """

        return

    def time(self):
        self._time += 1
        return self._time 

    def get_cache_key(self, key, cache_name):
        
        self.get_cache_key_count[cache_name] += 1

        a = time.time()
        is_symbol = False
        if key.is_symbol():
            content = key._content
            cache_key = "symbol:{}-{}".format(content[0], content[2][-1])
            is_symbol = True
        elif key.is_int_constant():

            if key.is_one():
                cache_key = "int_constant:1"
            elif key.is_zero():
                cache_key = "int_constant:0"
            else:
                cache_key = "int_constant:others"
        else:
            if cache_name == 'node':
                cache_key = key.node_type()
            else:
                cache_key = key

        self.get_cache_key_time[cache_name] += time.time() - a
#        key.cache_key_pair = (cache_key, is_symbol)
#        return (cache_key, is_symbol)
        return cache_key
    
    def get_variable_cache_key(self, fnode):

        return fnode

    """
    def set(self, key, value):

        cache_key = self.get_cache_key(key, 'tree')
        self._cache[cache_key] = value
        self._hits[cache_key] = 1
        self._size += 1

        return 

    def get(self, key):
        
        cache_key = self.get_cache_key(key, 'tree')
        ret = self._cache.get(cache_key)
        if isinstance(ret, np.ndarray):
#            self._hits[cache_key] += 1
            return ret, True

        return ret, False

    def set_tree_cache(self, key, value):
        return self.set(key, value)

    def get_tree_cache(self, key):
        return self.get(key)

    def in_tree_cache(self, key):
        cache_key, is_symbol = self.get_cache_key(key, 'tree')
        return cache_key in self._cache

    def set_node_cache(self, key, value):

        cache_key = self.get_cache_key(key, 'node')
        if cache_key in self._node_cache:
            print 'GGGG'
            raw_input()

        self._node_cache[cache_key] = value
        self._node_hits[cache_key] = 1
        self._node_size += 1
        
        return

    def get_node_cache(self, key):

        cache_key = self.get_cache_key(key, 'node')
        ret = self._node_cache.get(cache_key)
        if isinstance(ret, np.ndarray):
            self._node_hits[cache_key] += 1
            return ret, True

        return ret, False

    def in_node_cache(self, key):
        cache_key, is_symbol = self.get_cache_key(key, 'node')
        return cache_key in self._node_cache
    """

    def size(self):
#        return self._size + self._node_size;
        return len(self._tree_cache) + len(self._node_cache)

    def clean(self):
        
#        print 'clean cache {}'.format(self.size()), 

        """
        median = np.median([h for k, h in self._last_hit.items()])
        for k, h in self._last_hit.items():
            
            if h < median:
                del self._cache[k]
                del self._last_hit[k]

            if self._time - h[1] > (self.clean_threshold / 2):
                del self._hits[k]
                del self._cache[k]

        self._size = len(self._cache)
        print ' -> {}'.format(self.size())
        """
#        embed()
        return


class FeatureExtracter(object):

    exclude_method = ['is_literal', 'is_lira_op']
    max_extract_depth = 900

    @property
    def logger(self):
        name = self.__class__.__name__
        return logging.getLogger(name)

    def __init__(self):

        self.methods = []
        self.feature_cache_manager = FeatureCacheManager()
        self.feature_processor = FeatureProcessor()
        
        fnode_methods = fnode.FNode.__dict__.items()
        for i, j in fnode_methods:
            if type(j) == FunctionType:
                if 'is' in i and i not in self.exclude_method:
                    self.methods.append((i, int))
        self.basic_feature_num = len(self.methods)
        self.variable_features = [
            ('UNIQUE_SYMBOLS', frozenset), ('THEORY', frozenset)
        ]
        self.aggregable_features = self.methods + [
            ('LEAVES', int), ('TREE_NODES', int), ('DEPTH', int),
        ] + self.variable_features
        
        self.aggregable_feature_num = len(self.aggregable_features)
        self.feature_process_index = range(self.basic_feature_num) + [FI.LEAVES, FI.TREE_NODES]

        self.syntax_tree_feature_name = [
            'leaves',  'tree', 'symbols', 'depth', 'dag_nodes',
            'bool_dag_nodes', 'theory_num', 
            'clause_num', 'max_clause_size', 'min_clause_size', 'avg_clause_size'
#            'max_variable_degree', 'min_variable_degree', 'avg_variable_degree'
        ]
        
        self.f_id = dict([(name, i) for i, name in enumerate(self.syntax_tree_feature_name)])
        self.syntax_tree_feature_num = len(self.syntax_tree_feature_name)

        self.feature_dim = self.basic_feature_num + self.syntax_tree_feature_num

        self.logic_enum = config.logic_enum
        self.logic_num = len(self.logic_enum)

        self.size_oracle = SizeOracle()
        self.atom_oracle = AtomsOracle()

        """ Reset env's FormulaManager to custom class 'TemplateFormulaManager' """
#        env = get_env()
#        env.formula_manager = TemplateFormulaManager(env)

        return

    def do_parse(self, smt_file):

        parser = SmtLibParser()
        smt = open(smt_file, 'r')
        smt_raw = smt.read()
        smt.close()

        try:
            smt_script = parser.get_script(cStringIO(smt_raw))
        except Exception as exc:
            self.logger.error("parse(): {}".format(exc))
            return None

        return smt_script

    def init_features(self):

       return np.array([x[1]() for x in self.aggregable_features])
#        return np.zeros(self.basic_feature_num)

    def init_variable_features(self):

        return np.array([x[1]() for x in self.variable_features])

    def _aggregate_parent_child_features(self, node_features, child_feature_list, flag=0):

        tree_features = node_features.copy()
        if len(child_feature_list) == 0:
            return tree_features

#        a = time.time()

        max_depth = 0
        for cf in child_feature_list:

            tree_features[:self.basic_feature_num] += cf[:self.basic_feature_num]
            next_idx = self.basic_feature_num

            tree_features[FI.LEAVES] += cf[FI.LEAVES]
            tree_features[FI.TREE_NODES] += cf[FI.TREE_NODES]

            if cf[FI.DEPTH] > max_depth: 
                max_depth = cf[FI.DEPTH]

#            tree_features[FI.UNIQUE_SYMBOLS] |= cf[FI.UNIQUE_SYMBOLS] 
#            if flag != 0:
#                tree_features[FI.THEORY] |= cf[FI.THEORY] 

        tree_features[FI.DEPTH] = 1 + max_depth

#        b = time.time()
#        time_list[3] += b - a
        return tree_features

    def _extract_variable_feature(self, node, child_feature_list):

        features = self.init_variable_features()
        
        if node.is_symbol():
            features[0] = frozenset([node])

        node_type = node.node_type()
        if node_type in pysmt_op.RELATIONS:
            features[1] = frozenset([node])

        elif node_type in pysmt_op.BOOL_CONNECTIVES or node_type in pysmt_op.QUANTIFIERS:
            features[1] = frozenset(x for a in child_feature_list for x in a[FI.THEORY])

        elif node_type == pysmt_op.ITE:
            if any(a[FI.THEORY] is None for a in child_feature_list):
                features[1] = None
            else:
                features[1] = frozenset(x for a in child_feature_list for x in a[FI.THEORY])
        elif node.is_symbol(typing.BOOL):
            features[1] = frozenset([node])

        elif node.is_bool_constant():
            features[1] = frozenset()
          
        elif node_type == pysmt_op.FUNCTION:
            if node.function_name().symbol_type().return_type.is_bool_type():
                features[1] = frozenset([node])
            else:
                features[1] = None
        else:
            features[1] = None

        return features
       
    def _extract_node_feature(self, node):

        node_features = self.init_features()

        """ BASIC_FEATURES """
        for i, m in enumerate(self.methods):
            node_features[i] = int(getattr(node, m[0])())
        next_idx = self.basic_feature_num

        node_features[FI.LEAVES] = 1 if len(node.args()) == 0 else 0
        node_features[FI.TREE_NODES] = 1
        node_features[FI.DEPTH] = 1

        """
        if node.is_symbol():
            node_features[FI.UNIQUE_SYMBOLS] = frozenset([node])

        if node.node_type() in pysmt_op.RELATIONS:
            node_features[FI.THEORY] = frozenset([node])
        """

        return node_features

    def _extract_by_iteration(self, root):

        # For speed up #
        fcm = self.feature_cache_manager
        tree_cache_table = fcm._tree_cache
        node_cache_table = fcm._node_cache
        variable_cache_table = fcm._variable_cache

        root_tree_cache_key = fcm.get_cache_key(root, 'tree')
        tree_features = tree_cache_table.get(root_tree_cache_key)
        if isinstance(tree_features, np.ndarray):
            return tree_features

        process_stack = [(False, root, root_tree_cache_key, None)]
        push = process_stack.append
        pop = process_stack.pop
        while process_stack:

            (child_result_ready, fnode, tree_cache_key, child_tree_cache_keys) = pop()
            if not child_result_ready:

                child_tree_cache_keys = []
                push((True, fnode, tree_cache_key, child_tree_cache_keys))
                for child in fnode.args():
                    child_tree_cache_key = fcm.get_cache_key(child, 'tree')
                    child_tree_cache_keys.append(child_tree_cache_key)
                    if child_tree_cache_key not in tree_cache_table:
                        push((False, child, child_tree_cache_key, None))
            else:
                if tree_cache_key in tree_cache_table:
                    continue

#                node_features = self.complete_node_features(node, fcm)
                node_cache_key = fcm.get_cache_key(fnode, 'node')
                node_features = node_cache_table.get(node_cache_key)
                if not isinstance(node_features, np.ndarray):
                    node_features = self._extract_node_feature(fnode)
                    node_cache_table[node_cache_key] = node_features

                child_feature_list = [
                    tree_cache_table[cache_key] for cache_key in child_tree_cache_keys
                ]

                """
                variable_cache_key = fcm.get_variable_cache_key(fnode)
                variable_features = variable_cache_table.get(variable_cache_key)
                if not isinstance(variable_features, np.ndarray):
                    variable_features = self._extract_variable_feature(fnode, child_feature_list)
                    variable_cache_table[variable_cache_key] = variable_features
                node_features[[FI.UNIQUE_SYMBOLS, FI.THEORY]] = variable_features
                """

#                variable_features = self._extract_variable_feature(fnode, child_feature_list)
#                node_features[[FI.UNIQUE_SYMBOLS, FI.THEORY]] = variable_features
                tree_features = self._aggregate_parent_child_features(
                    node_features, child_feature_list
                )
                tree_cache_table[tree_cache_key] = tree_features
                
        return tree_cache_table[root_tree_cache_key]

    def _extract_by_recursion(self, node, depth, use_cache=None):

        use_cache = True
        if depth > self.max_extract_depth:
            return self.init_features()

        # For Speed up #
        fcm = self.feature_cache_manager
        tree_cache_table = fcm._tree_cache
        node_cache_table = fcm._node_cache
        if use_cache:
            tree_cache_key = fcm.get_cache_key(node, 'tree')
            tree_features = tree_cache_table.get(tree_cache_key)
            if isinstance(tree_features, np.ndarray):
                return tree_features

        node_features = None
        if use_cache:
            node_cache_key = fcm.get_cache_key(node, 'node')
            node_features = node_cache_table.get(node_cache_key)

        if not isinstance(node_features, np.ndarray):
            node_features = self._extract_node_feature(node)
            if use_cache:
                node_cache_table[node_cache_key] = node_features

        child_feature_list= []
        for arg in node.args():
            child_feature_list.append(self._extract_by_recursion(arg, depth+1, use_cache=use_cache))
        
        tree_features = self._aggregate_parent_child_features(node_features, child_feature_list)
        if use_cache:
            tree_cache_table[tree_cache_key] = tree_features
        return tree_features

    def extract_aggregable_feature(self, smt_script, extract_method='iteration'):

        if extract_method == 'iteration':
            do_extract = self._extract_by_iteration
        else:
            do_extract = self._extract_by_recursion

        feature_list = []
        for cmd in smt_script.commands:
            if cmd.name != 'assert':
                continue
            for arg in cmd.args:
                feature_list.append(do_extract(arg))
            
        features = self._aggregate_parent_child_features(
            node_features=self.init_features(),
            child_feature_list=feature_list,
            flag=1
        )
        return features

    def _extract_syntaxtree_feature(self, root):

        start = time.time()
        
        features = [0 for i in self.syntax_tree_feature_name]
        features[self.f_id['leaves']] = int(root.size(measure=SizeOracle.MEASURE_LEAVES))
        features[self.f_id['tree']] = int(root.size(measure=SizeOracle.MEASURE_TREE_NODES))
        features[self.f_id['symbols']] = root.size(measure=SizeOracle.MEASURE_SYMBOLS)
        features[self.f_id['depth']] = root.size(measure=SizeOracle.MEASURE_DEPTH)
        features[self.f_id['dag_nodes']] = root.size(measure=SizeOracle.MEASURE_DAG_NODES)
        features[self.f_id['bool_dag_nodes']] = root.size(measure=SizeOracle.MEASURE_BOOL_DAG)

#        theory = self.atom_oracle.walk(node)

        VG = VariableGraph()
        VG.build_graph(root)
        features[self.f_id['theory_num']] = len(VG.theory)

        time_list[4] += features[self.f_id['theory_num']]

        clause_list = VG.get_clause()
        clause_num = len(clause_list)
        clause_size = [len(x) for x in clause_list]

        features[self.f_id['clause_num']] = len(clause_list)
        if clause_num != 0:
            features[self.f_id['max_clause_size']] = max(clause_size)
            features[self.f_id['min_clause_size']] = min(clause_size)
            features[self.f_id['avg_clause_size']] = float(sum(clause_size)) / clause_num 

        end = time.time()

        return features
    
    def extract_syntaxtree_feature(self, smt_script):

        time_list[4] = 0
        features = np.zeros(self.syntax_tree_feature_num)
        for cmd in smt_script.commands:
            if cmd.name != 'assert':
                continue
            for arg in cmd.args:
                f1 = self._extract_syntaxtree_feature(arg)
                features = np.add(features, f1)

        return features

    def extract_logic_feature(self, logic):

        features = np.zeros(self.logic_num)
        features[self.logic_enum[logic]] = 1

        return features

    def to_numerical(self, features):

        features[FI.UNIQUE_SYMBOLS] = len(features[FI.UNIQUE_SYMBOLS])
        features[FI.THEORY] = len(features[FI.THEORY])
        return features

    def _extract(self, smt_file):

        if smt_file == None:
            raise ExtractError("smt_file object is None.")

        smt_script = self.do_parse(smt_file)
        if smt_script == None:
            raise ExtractError("smt_script object is None.")

 #       syntax_tree_features = self.extract_syntaxtree_feature(smt_script)
 #       raw_input()

        try:
            logic = str(smt_script.filter_by_command_name('set-logic').next().args[0])
        except:
            logic = ''
        meta_features = np.array([logic])

        s2 = time.time()
        aggregate_features = self.extract_aggregable_feature(smt_script)
        s3 = time.time()
        aggregate_features = self.to_numerical(aggregate_features)
        s4 = time.time()

        index = self.feature_process_index
        aggregate_features[index] = self.feature_processor.process_one(aggregate_features[index])
        s5 = time.time()

        logic_feature = self.extract_logic_feature(logic)
        features = np.append(aggregate_features, logic_feature)
        s6 = time.time()

        time_list[0] += s6 - s2
        time_list[1] += s3 - s2
        time_list[2] += s4 - s3
        time_list[3] += s5 - s4
        time_list[4] += s6 - s5
        consecutive_time.append(s6 - s2)
        print time_list

        return meta_features, features

    def extract(self, smt_files, dataset_name, output_dir=None):

        index_list = np.array([], dtype=int)
        all_meta_features = np.array([[]])
        all_features = np.array([[]])
        output_name = "{}.features".format(dataset_name)

        if (output_dir):
            output_name = os.path.join(output_dir, output_name)

        next_index = -1
        if os.path.exists(output_name):
            f = np.load(open(output_name, 'rb'))
            all_features = f['features'].copy()
            all_meta_features = f['meta_features'].copy()
            index_list = f['index_list'].copy()
            if len(index_list) != 0:
                next_index = max(index_list)
            else:
                next_index = -1

        total_num = len(smt_files)
        for index, smt_file in enumerate(smt_files):

            if index <= next_index:
                sys.stdout.write('[%s] Now process: [%d \ %d] (skip)\n' % (
                    datetime.now().strftime("%H:%M:%S"), index, total_num
                ))
                sys.stdout.flush()
                continue

            if index:
                log = '[%s] Now process: [%d \ %d] (cache_size: %d, formula_size: %d)\n'
                sys.stdout.write(log % (
                    datetime.now().strftime("%H:%M:%S"), 
                    index, 
                    total_num, 
                    self.feature_cache_manager.size(), 
                    len(get_env().formula_manager.formulae)
                ))
                sys.stdout.flush()

            try:
                meta_features, features = self._extract(smt_file=smt_file)
#                if self.feature_cache.size() > self.feature_cache.clean_threshold:
#                    embed()

                axis = 0 if all_features.size else 1
                all_features = np.append(all_features, [features], axis=axis)
                all_meta_features = np.append(all_meta_features, [meta_features], axis=axis)
                index_list = np.append(index_list, [index], axis=0)

            except KeyboardInterrupt:
                print 'break'
                break
            except:
                """
                err = exc.__class__.__name__
                msg = exc.message
                log = "When extracting \"{}\":\n{}: {}".format(smt_file, err, msg)
                self.logger.info("exctract(): {}".format(log))
                """
#                print 'error'
#                continue
                raise
 
        print 'saving features...'
        np.savez(
            open(output_name, 'wb'), 
            features=all_features, 
            meta_features=all_meta_features,
            index_list=index_list
        )
        f = open('consecutive_time.log', 'w')
        f.write('\n'.join([str(x) for x in consecutive_time]))
        f.close()
        return output_name



if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO
    )

    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--file", dest="filename", default=None,
        help="The name of target file"
    )
    parser.add_argument(
        "-extract", "--extract-features", 
        dest="extract", action="store_true", default=False,
        help="Extract features to the files store in a given \".filelist\" file."
    )
    parser.add_argument(
        "-p", "--partition",
        dest="partition_id", default=None,
        help="Extract features to specific partition of files in a given \".filelist\" file"
    )
    parser.add_argument(
        "-process", "--process-features",
        dest="process", action="store_true", default=False,
        help="Perform feature processing for a given \".features\" file."
    )
    parser.add_argument(
        "-output-dir", "--output-dir", 
        dest="output_dir", default=None
    )
    args = parser.parse_args()

    if (args.extract):
        dataset_name = os.path.basename(args.filename.replace('.filelist', ''))
        smt_files = open(args.filename, 'r').read().strip().split('\n')
       
        if (args.partition_id):
            partition = int(args.partition_id)
            dataset_name += str(partition)
            smt_files = smt_files[100000*partition:100000*(partition+1)]

        FE = FeatureExtracter()
        FE.extract(smt_files, dataset_name, args.output_dir)

    elif (args.process):
        FP = FeatureProcessor()
        f = np.load(open(args.filename, 'rb'))
        np.savez(
            open('processed_' + args.filename, 'wb'), 
            features=FP.process(f['features']), 
            meta_features=f['meta_features'],
            index_list=f['index_list']
        )

    else:
        parser.print_help()
        
    exit()

