
import os
import logging

from DNN import DNNModel
from RandomForest import RandomForestModel
from CostSensitive import CostSensitiveModel

class ModelNotFound(Exception):
    pass

class ModelFileNotFound(Exception):
    pass

class NonSupportedModelType(Exception):
    pass

class NotImplementError(Exception):
    pass

class ModelManager(object):

    supported_model = dict([
        ("DNN", DNNModel),
        ("RandomForest", RandomForestModel),
        ("CostSensitive", CostSensitiveModel),
        ("Unknown", None)
    ])
    
    @property
    def logger(self):
        name = self.__class__.__name__
        return logging.getLogger(name)

    def __init__(self):

        self.models = {}
        self.meta = {}
        return

    def load_model(self, model_type, model_name, model_file_path):

        if model_type not in self.supported_model:
            raise NonSupportedModeltype(model_type)

        if not os.path.exists(model_file_path):
            raise ModelFileNotFound(model_file_path)

        model = self.supported_model[model_type]
        model.load(model_file_path)
        self.models[model_name] = model
        self.meta[model_name] = {'type': model_type, 'path': model_file_path}
        return

    def load_models(self, model_list):

        for model_data in model_list:
            self.load_model(model_data['type'], model_data['name'], model_data['path'])
        return

    def save_model(self, model_path_list):

        for model_name, model_path in model_path_list:
            
            if model_name not in self.models:
                continue
            model = self.models[model_name]
            model.save(model_path)

        return

    def set_model(self, model_name, model):
        self.models[model_name] = model
        self.meta[model_name] = {'type': model.__class__.__name__.replace('Model', ''), "path": ""}
        return

    def get_model(self, model_name):

        if model_name not in self.models:
            raise ModelNotFound(model_name)
        return self.models[mode_name]

    def get_meta(self, model_name):

        if model_name not in self.models:
            raise ModelNotFound(model_name)
        return self.meta[model_name]

