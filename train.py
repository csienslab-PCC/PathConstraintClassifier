#!/usr/bin/env python

import os
import sys
import json
import logging

from model import *
from evaluate import SolverEvaluator
from combine import DataCombiner
from feature import FeatureExtracter
from argparse import ArgumentParser

"""
from ModelTrainer import DeepNeuralNetworkModelTrainer
from ModelTrainer import CostSensitiveModelTrainer
from ModelTrainer import OtherModelTrainer
"""

class NonSupportedModel(Exception):
    pass


class TrainingManager(object):
 
    _supported_model = [
        ("DNN", DNNModel),
        ("RandomForest", RandomForestModel),
        ("CostSensitive", CostSensitiveModel),
        ("Enhanced", EnhancedModel)
    ]

    @property
    def logger(self):
        name = self.__class__.__name__
        return logging.getLogger(name)

    def __init__(self):

        self.Evaluator = SolverEvaluator()
        self.DataCombiner = DataCombiner()
        self.FeatureExtracter = FeatureExtracter()
        self.supported_model = dict(self._supported_model)

        return
    
    def get_model(self, model_name):

        if model_name not in self.supported_model:
            raise NonSupportedModel(model_name)

        return self.supported_model[model_name]()

    def train(self, smt_files=None, training_data=None, feature_data=None, 
              answer_data=None, model_name=None):

        model_name = 'DNN' if model_name == None else model_name
 
        if smt_files != None:
            dataset_name = smt_files.replace('.filelist', '')
            smt_files = open(smt_file, 'r').read().strip().split('\n')

        if training_data == None:

            if feature_data == None:
                feature_data = self.FeatureExtracter.extract(smt_files, dataset_name)

            if answer_data == None:
                answer_data = self.Evaluater.evaluate(smt_files, dataset_name)
        
            training_data = self.DataCombiner.combine(
                feature_data=feature_data,
                answer_data=answer_data
            )
        
        model = self.get_model(model_name)
        model.train(training_data)

        return model


def main(args):

    TM = TrainingManager()
    model = TM.train(
        smt_files=args.smt_files,
        training_data=args.training_data,
        feature_data=args.feature_data,
        answer_data=args.answer_data,
        model_name=args.model_name
    )

    return model


if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO
    )    
    
    parser = ArgumentParser()
    parser.add_argument("-s", "--smt-files", dest="smt_files", action="store",
                        help="filelist of training data. (.smt2 file)")
    parser.add_argument("-a", "--answer-data", dest="answer_data", action="store",
                        help="answer data name.")
    parser.add_argument("-f", "--feature-data", dest="feature_data", action="store",
                        help="feature data name.")
    parser.add_argument("-t", "--training-data", dest="training_data", action="store",
                        help="training data name.")
    parser.add_argument("-m", "--model-name", dest="model_name", action="store",
                        help="target model name.")
    args = parser.parse_args()
    main(args)

