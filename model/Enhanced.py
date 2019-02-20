
import os 
import sys
import json
import time
import logging

import numpy as np

from DNN import DNNModel
from RandomForest import RandomForestModel

from ModelException import *

sys.path.append('../')
import config

class EnhancedModel(object):

    @property
    def logger(self):
        name = self.__class__.__name__
        return logging.getLogger(name)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):

        self._model = value

        """
        if isinstance(value, str):
            if value not in self._supported_model:
                raise NonSupportedModel(value)
            else:
                self._model = self._supported_model[value]
        else:
            self._model = value
        """

        return
            

    def __init__(self, model=None, model_name="DNN"):
        
        self.model = model
        self.solver_list = config.solver_list
        self.solver_enum = config.solver_enum
        self.supported_logic = config.supported_logic
        self.backup_solver = self.solver_enum['z3']
        self.backup_active_time = 1

        self.logic_selection_time = 0.0
        self.avg_logic_selection_time = 0.0
        self.prediction_time = 0
        self.avg_prediction_time = 0

        return

    def load(self, model_file_path):
        self._model.load(model_file_path)
        return

    def save(self, model_file_path):
        self._model.save(model_file_path)
        return

    def logic_selection(self, proba, logic):

        start = time.time()

        candidate = []
        for i, v in enumerate(proba):

            solver = self.solver_list[i]
            if solver == 'dummy':
                candidate.append([i, v])
            elif logic in self.supported_logic[solver]:
                candidate.append([i, v])

        if len(candidate) == 0:
            
            print logic
            print type(logic)
            return 'QQ'
            raise RuntimeError("Unexpected size of candidate: 0")
        
        max_proba = candidate[0]
        for c in candidate:
            if c[1] > max_proba[1]:
                max_proba = c
        
        end = time.time()
        self.logic_selection_time += end - start

        return max_proba[0]

    def train(self, training_data):

        self._model.train(training_data)

        return

    def predict(self, data, enable_logic_selection=True):

        if enable_logic_selection:
            proba = self.predict_proba(data)
        
            self.prediction_time = self._model.prediction_time
            self.avg_prediction_time = self._model.avg_prediction_time

            assert(len(proba) == len(data['m']))

            ret = []
            count = 0
            for i, (p, meta) in enumerate(zip(proba, data['m'])):
 
                ans = self.logic_selection(p, meta[0])
                ret.append(ans)

                if ret[-1] == 'QQ':
                    print data['z'][i]
                    print data['y'][i]
                    raise RuntimeError()

            self.avg_logic_selection_time = float(self.logic_selection_time) / len(data['m'])

        else:
            ret = self._model.predict(data)

#        print 'backup_count:', count
        return ret

    def predict_proba(self, data):

        return self._model.predict_proba(data)



if __name__ == '__main__':


    EM = EnhancedModel()
    
    EM.model = DNNModel()
    print EM.model

    print EM.solver_list

