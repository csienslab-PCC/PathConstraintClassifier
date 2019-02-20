

class EmptyModelError(Exception):

    def __init__(self, model_name):
        self.message = 'Model object is None, use %sModel.train() to train new model.' % (model_name)

    def __str__(self):
        return self.message

class NonSupportedModel(Exception):

    def __init__(self, model_name):
        self.message = '%s is not supported.' % (model_name)

    def __str__(self):
        return self.message

