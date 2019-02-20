
import sys
import pickle
import logging
import numpy as np

from ModelException import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification

sys.path.append('../')
import config

class GradientBoostingModel(object):

    @property
    def logger(self):
        name = self.__class__.__name__
        return logging.getLogger(name)

    def __init__(self, model=None):

        self.solver_list = config.solver_list
        self.model = model
        return

    def save(self, model_file_path):
        pickle.dump(self.model, open(model_file_path, 'w'))       
        return

    def load(self, model_file_path):
        self.model = pickle.load(open(model_file_path, 'r'))
        return

    def process_data(self, training_data):

        return training_data

    def train(self, training_data):

        training_data = self.process_data(training_data)
        
        model = GradientBoostingClassifier()
        model.fit(training_data['x'], training_data['y'])
        self.model = model
        print 'train done.'
        return

    def predict(self, data):
        
        if self.model == None:
            raise EmptyModelError(self.__class__.__name__)

        ans = self.model.predict(data['x'])
        return ans

    def predict_proba(self, data):

        ans = self.model.predict_proba(data['x'])

        ret = []
        classes = self.model.classes_
        for a in ans:
            
            temp = [0.0 for _ in range(len(self.solver_list))]
            for i, c in enumerate(classes):
                temp[int(c)] = a[i]
            ret.append(temp)

        return ret


if __name__ == '__main__':

    data = np.load(sys.argv[1])

    RFM = GradientBoostingModel()
    RFM.train(data)
    ans = RFM.predict(data['x'])

    count = 0
    for i, a in enumerate(ans):
        if a == data['y']:
            count += 1

    print float(count) / len(ans)
