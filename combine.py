
import os
import sys
import json
import logging
import numpy as np

import config

from argparse import ArgumentParser


class AnswerNotFound(Exception):

    def __init__(self, file_id, file_path):
        self.message = "id: %(id)s, path: %(path)s" % {'id':file_id, 'path':file_path}

    def __str__(self):
        return self.message


class AnswerNotComplete(Exception):

    def __init__(self, file_id, file_path):
        self.message = "id: %(id)s, path: %(path)s" % {'id':file_id, 'path':file_path}

    def __str__(self):
        return self.message


class DataCombiner(object):

    @property
    def logger(self):
        name = self.__class__.__name__
        return logging.getLogger(name)
        
    def __init__(self):

        self.solver_list = config.solver_list
        self.solver_num = len(self.solver_list) - 1
        self.tag_list = [
            'klee', 'angr', 'dd', 'expr', 'echo', 'printf', 'od', 'mknod', 'dircolors', 'pathchk'
        ]
        return

    def load_result(self, s):

        ret = s.replace('True', 'true').replace('False', 'false').replace("'", "\"")
        
        try:
            ret = json.loads(ret)
        except:
            print ret
            raise
        return ret

    def valid(self, a_result):

        # Error occurs when solving constraints. Ex. Non-supported logic.
        if len(a_result) == 3:
#            print a_result
            return not a_result['error']

        # Fail to get answer or solve constraints.
        if len(a_result[-2]) == 0:
            return False

        return True

    def _valid(self, a_result):

        # Error occurs when solving constraints. Ex. Non-supported logic.
        if len(a_result) == 3:
            if a_result[0] == True:
                return False

        # Fail to get answer or solve constraints.
        if len(a_result[-2]) == 0:
            return False

        return True

    def get_answer(self, result):

        time_list = [(i, v['answer'], v['time']) for i, v in enumerate(result) if self.valid(v)]
#        print time_list

        if len(time_list) == 0:
#            fastest_solver = len(self.solver_list)
#            fastest_solver = self.solver_list[-1]
            fastest_solver = self.solver_list.index('dummy')

        else:
            fastest_solver = min(time_list, key=lambda x: x[2])[0]

        return fastest_solver

    def _get_answer(self, result):

        time_list = [(i, v[-2], v[-1]) for i, v in enumerate(result) if self.valid(v)]
        print time_list

        if len(time_list) == 0:
#            fastest_solver = len(self.solver_list)
#            fastest_solver = self.solver_list[-1]
            fastest_solver = self.solver_list.index('dummy')

        else:
            fastest_solver = min(time_list, key=lambda x: x[2])[0]

        return fastest_solver

    def combine_features(self, 
        feature_filelist, dataset_name, 
        partition_size, output_dir=None):

        aggregate = {
            'features' : None, 
            'index_list' : None,
            'meta_features' : None
        }
        for i, feature_file in enumerate(feature_filelist):

            f = np.load(open(feature_file, 'rb'))

#            index_list = np.zeros(len(f['index_list'])) + [x + partition_size*i for x in f['index_list']]
            index_list = np.array([x + partition_size*i for x in f['index_list']])
            
            if i == 0:
                aggregate['features'] = f['features'].copy()
                aggregate['index_list'] = f['index_list'].copy()
                aggregate['meta_features'] = f['meta_features'].copy()
                continue

            axis = 0 if aggregate['features'].size else 1
            aggregate['features'] = np.append(aggregate['features'], f['features'], axis=0)
            aggregate['index_list'] = np.append(aggregate['index_list'], index_list)
            aggregate['meta_features'] = np.append(aggregate['meta_features'], f['meta_features'], axis=0)

        output_filename = dataset_name + '.features'
        if (output_dir):
            output_filename = os.path.join(output_dir, output_filename)

        np.savez(
            open(output_filename, 'wb'), 
            features = aggregate['features'],
            index_list = aggregate['index_list'],
            meta_features = aggregate['meta_features']
        )
        return

    def get_tag(self, file_path):

        tags = []
        
        for key in self.tag_list:
            if '/' + key in file_path:
                tags.append(key)

        assert(len(tags) <= 2)
        for _ in range(5 - len(tags)):
            tags.append('')
        
        print tags
        return tags

    def combine(self, feature_filename, answer_filename, dateset_name, output_dir=None):

        output = "{}.ml".format(dataset_name)
        if (output_dir):
            output = os.path.join(output_dir, output)

        f = np.load(open(feature_filename, 'rb'))
        feature_data = f['features']
        feature_index = f['index_list']
        meta_features = f['meta_features']

        answer_data = json.load(open(answer_filename, 'r')).get('result')
        answer_dict = {}
        for r in answer_data:
            answer_dict[r['id']] = {
                'file_path' : r['file_path'],
#                'result': [self.load_result(s) for s in r['result']]
                'result': r['result'] if len(r['result']) == 7 else r['result'][:-1]

            }
        
        training_data = {
            'x': feature_data,
            'y': np.array([]),
            'z': np.array([[]]),
            'm': meta_features,
            't': np.array([[]])
#            'm': np.array([[]]),
        }
        for i, index in enumerate(feature_index):
 
            print 'now process: %d / %d' % (i, len(feature_index))

            data = answer_dict.get(index)
            if data == None:
                raise AnswerNotFound(index, data['file_path'])

            if len(data['result']) < self.solver_num:
                raise AnswerNotComplete(index, data['file_path'])

            ans = self.get_answer(data['result'])
#            time_list = [r[-1] for r in data['result']]
#            time_list = [str(x) for x in data['result']]
            time_list = data['result']

            axis = 0 if training_data['z'].size else 1
#            print time_list
            training_data['y'] = np.append(training_data['y'], ans)
            training_data['z'] = np.append(training_data['z'], [time_list], axis=axis)
            training_data['t'] = np.append(
                training_data['t'], [self.get_tag(data['file_path'])], axis=axis
            )
#            training_data['m'] = np.append(training_data['m'], [meta_features[index][0], data['file_path']], axis=axis)
#            print training_data['m'][index]

        np.savez(
            open(output, 'wb'), 
            x=training_data['x'],
            y=training_data['y'],
            z=training_data['z'],
            m=training_data['m'],
            t=training_data['t']
        )

        return output


if __name__ == '__main__':

    DC = DataCombiner()

    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--file", dest="filename", default=None, nargs="+",
        help="The name of target file"
    )
    parser.add_argument(
        "-partition-size", "--partition-size",
        dest="partition_size", default=None,
        help="The size of each feature partition."
    )
    parser.add_argument(
        "-combine-features", "--combine-features", 
        dest="combine_features", action="store_true", default=False,
        help="Combine features for a given .featurelist file."
    )
    parser.add_argument(
        "-o", "--output-dir", dest="output_dir", default=None,
        help="Output path of file."
    )
    args = parser.parse_args()
    
    if (args.combine_features):
        dataset_name = os.path.basename(args.filename[0]).replace('.featurelist', '')
        feature_list = open(args.filename[0]).read().strip().split('\n')
        partition_size = int(args.partition_size)
        DC.combine_features(feature_list, dataset_name, partition_size, args.output_dir)

    else:
        dataset_name = os.path.basename(args.filename[0]).replace('.features', '')
        DC.combine(args.filename[0], args.filename[1], dataset_name, args.output_dir)
    
