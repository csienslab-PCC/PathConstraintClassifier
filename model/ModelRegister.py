
import os
import logging

from DNN import *
from RandomForest import RandomForestModel
from GradientBoosting import GradientBoostingModel
from CostSensitive import CostSensitiveModel
from Enhanced import EnhancedModel
from Solver import SolverModel

class ModelNotFound(Exception):
    pass

class ModelFileNotFound(Exception):
    pass

class NonSupportedModelType(Exception):
    pass

class NotImplementError(Exception):
    pass

class ModelRegister(object):

    _supported_model = [
        ("DNN", DNNModel),
        ("DNN-alpha", DNNModelAlpha),
        ("DNN-beta", DNNModelBeta),
        ("DNN-gamma", DNNModelGamma),
        ("RandomForest", RandomForestModel),
        ("GradientBoosting", GradientBoostingModel),
        ("CostSensitive", CostSensitiveModel),
        ("Enhanced", EnhancedModel),
        ("Solver", SolverModel),
        ("Unknown", None)
    ]
    
    @property
    def logger(self):
        name = self.__class__.__name__
        return logging.getLogger(name)

    def __init__(self):

        self.supported_model = dict(self._supported_model)

        return

    def get_model(self, model_name):

        if model_name not in self.supported_model:
            raise ModelNotFound(model_name)
        else:
            return self.supported_model[model_name]()

