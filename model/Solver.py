
import os
import sys
import logging


class SolverModel(object):

    @property
    def logger(self):
        name = self.__class__.__name__
        return logging.getLogger(name)

    def __init__(self, solver_id=-1, solver_num=7):

        self.solver_num = solver_num
        self.solver_id = solver_id
        self.prediction_time = 0.0
        self.avg_prediction_time = 0.0

        return

    def load(self, model_name):

        self.solver_id = open(model_name).read().strip()
        return 

    def save(self, model_name):

        with open(model_name, 'w') as f:
            f.write(str(self.solver_id))

        return 

    def predict(self, data):
    
        ans = self.solver_id
        return [ans for _ in range(len(data['x']))]

    def predict_proba(self, data):
        
        ans = [0.0 if i != self.solver_id else 1.0 for i in range(self.solver_num)]
        return [ans for _ in range(len(data['x']))]
        

if __name__ == '__main__':

    z3_model = SolverModel(solver_id=6)

