#!/usr/bin/env python

import os
import sys
import json
import random
import logging

import numpy as np

from IPython import embed

class _DataPartitioner(object):

    @property
    def logger(self):
        name = self.__class__.__name__
        return logging.getLogger(name)

    def __init__(self, ratio=0.4):

        self.default_ratio = ratio 
        self.ratio = {}
        return

    def load(self, config):

        self.ration = {}
        self.tags = {}
        self.data = {}
        for info in config:
            
            self.data[info['dataset_name']] = np.load(open(info['dataset_path']))
            self.tags[info['dataset_name']] = info['tags']
            if 'ratio' in info:
                self.ration[info['dataset_name']] = info['ratio']

        return

    def shuffle(self, data):

        np.random.seed(1126)
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        
        ret = {}
        for l in ['x', 'y', 'z', 'm']:
            ret[l] = data[l].copy()
            ret[l][indices]

        return ret

    def cut(self, data, num):
        
        ret = {}
        for l in ['x', 'y', 'z', 'm']:
            ret[l] = data[l].copy()[:num]

        return ret

    def partition(self, dataset, ratio=None, output=None):

        train = {}
        test = {}
        default_ratio = self.ratio if ratio == None else ratio
        report = {}

        label = ['x', 'y', 'z', 'm']
        for name, data in self.data.iteritems():
           
            if name in self.ratio:
                ratio = self.ratio[name]
            else:
                ratio = self.default_ratio

            index_list = dict([(k, []) for k in data['tags'] + ['res']])

            data_num = len(data['x'])
            train_indices = sorted(random.sample(range(data_num), int(data_num * ratio)))
            test_indices = [x for x in range(data_num) if x not in train_indices]

            report[name] = {}
            report[name]['train'] = train_indices
            report[name]['test'] = test_indices

            print name, data_num
            print len(train_indices)
            print len(test_indices)

            for l in label:
                if l not in train:
                    train[l] = data[l][train_indices]
                    test[l] = data[l][test_indices]
                else:
                    train[l] = np.append(train[l], data[l][train_indices], axis=0)
                    test[l] = np.append(test[l], data[l][test_indices], axis=0)


        if output != None:
            json.dump(report, open(os.path.join(output, 'partition_info.json'), 'w'))

        return train, test



class DataPartitioner(object):

    @property
    def logger(self):
        name = self.__class__.__name__
        return logging.getLogger(name)

    def __init__(self, ratio=0.8):

        self.default_ratio = ratio 
        self.ratio = {}
        return

    def load(self, config):

        data = {}
        for data_name, file_path in config['data'].iteritems():
            data[data_name] = np.load(open(file_path))
        
        if 'ratio' in config:
            for data_name, ratio in config['ratio'].iteritems():
                self.ratio[data_name] = ratio

        return data

    def shuffle(self, data):

        np.random.seed(1126)
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        
        ret = {}
        for l in ['x', 'y', 'z', 'm']:
            ret[l] = data[l].copy()
            ret[l][indices]

        return ret

    def cut(self, data, num):
        
        ret = {}
        for l in ['x', 'y', 'z', 'm']:
            ret[l] = data[l].copy()[:num]

        return ret


    def partition(self, dataset, ratio=None, output=None):

        train = {}
        test = {}
        default_ratio = self.ratio if ratio == None else ratio
        report = {}

        label = ['x', 'y', 'z', 'm']
        for name, data in dataset.iteritems():
           
            if name in self.ratio:
                ratio = self.ratio[name]
            else:
                ratio = self.default_ratio

            data_num = len(data['x'])
            train_indices = sorted(random.sample(range(data_num), int(data_num * ratio)))
            test_indices = [x for x in range(data_num) if x not in train_indices]

            report[name] = {}
            report[name]['train'] = train_indices
            report[name]['test'] = test_indices

            print name, data_num
            print len(train_indices)
            print len(test_indices)

            for l in label:
                if l not in train:
                    train[l] = data[l][train_indices]
                    test[l] = data[l][test_indices]
                else:
                    try:
                        train[l] = np.append(train[l], data[l][train_indices], axis=0)
                        test[l] = np.append(test[l], data[l][test_indices], axis=0)
                    except:
                        embed()


        if output != None:
            json.dump(report, open(os.path.join(output, 'partition_info.json'), 'w'))

        return train, test

class DataGrouper(object):

    def __init__(self, config):

        self.load(config)

        return

    def load(self, config):

        ratio = {}
        dataset = {}
        for data_name, data_info in config['data'].iteritems():
            dataset[data_name] = np.load(data_info['path'])
            ratio[data_name] = data_info['ratio']
        self.dataset = dataset
        self.ratio = ratio
        return

    def group(self):

        data_group = {}
        for data_name, data in self.dataset.iteritems():

            data_group[data_name] = {}
            for index, tags in enumerate(data['t']):
                
                tag_key = "-".join([data_name] + [x for x in tags if len(x) != 0])
                if tag_key not in data_group[data_name]:
                    data_group[data_name][tag_key] = []
                data_group[data_name][tag_key].append(index)

        return data_group

    def partition_by_key(self, partition_key):
        
        dataset = self.dataset
        random_ratio = 0.4

        train = {}
        test = {}
        report = {}
        label = ['x', 'y', 'z', 'm', 't']
        partition_key = ['-'.join(x) for x in partition_key]
        print partition_key
        for name, data in dataset.iteritems():
            
            data_num = len(data['x'])
            train_indices = []
            random_indices = []
            for i, tags in enumerate(data['t']):
                

                tag_key = '-'.join([x for x in tags if x])

#                print tag_key
                if tag_key in partition_key:
                    train_indices.append(i)
                elif len(tag_key) == 0:
                    random_indices.append(i)
                    
            train_indices += sorted(random.sample(random_indices, int(len(random_indices) * random_ratio)))


            test_indices = [x for x in range(data_num) if x not in train_indices]

            report[name] = {}
            report[name]['train'] = train_indices
            report[name]['test'] = test_indices

            print name, data_num
            print 'random:', len(random_indices)
            print 'train:', len(train_indices)
            print 'test:', len(test_indices)

            for l in label:
                if l not in train:
                    train[l] = data[l][train_indices]
                    test[l] = data[l][test_indices]
                else:
                    try:
                        train[l] = np.append(train[l], data[l][train_indices], axis=0)
                        test[l] = np.append(test[l], data[l][test_indices], axis=0)
                    except:
                        embed()

        json.dump(report, open('report.json', 'w'))

        return train, test

    def partition(self):

        group = self.group()

        train = {}
        test = {}

        label = ['x', 'y', 'z', 'm', 't']
        report = {}
        for name, data in self.dataset.iteritems():

            train_index = []
            test_index = []
            ratio = self.ratio[name]
            for tag_key, index in group[name].iteritems():
                
                data_num = len(index)
                temp_train_index = sorted(random.sample(index, int(data_num * ratio)))
                temp_test_index = [x for x in index if x not in temp_train_index]
                
                train_index += temp_train_index
                test_index += temp_test_index

                print tag_key
                print len(temp_train_index), len(temp_test_index)
                print len(train_index), len(test_index)

            for l in label:
                if l not in train:
                    train[l] = data[l][train_index]
                    test[l] = data[l][test_index]
                else:
                    train[l] = np.append(train[l], data[l][train_index], axis=0)
                    test[l] = np.append(test[l], data[l][test_index], axis=0)

            print 'data_len:', len(train['x']), len(test['x'])

            report[name] = {}
            report[name]['train'] = train_index
            report[name]['test'] = test_index
            

        json.dump(report, open('report.json', 'w'))

        return train, test

    def output(self, data, name):
        
        np.savez(
            open(name + '.train', 'w'), 
            x=data['x'], 
            y=data['y'], 
            z=data['z'], 
            m=data['m'], 
            t=data['t']
        )

        return


def main():

    config = json.loads(open(sys.argv[1]).read())

    DP = DataPartitioner()
    DP.load(config)
    train, test = DP.partition()

    embed()    

    return


if __name__ == '__main__':

    main()

