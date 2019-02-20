
import os
import sys
import logging

from shutil import copyfile

class CostSensitiveModel(object):

    @property
    def logger(self):
        name = self.__class__.__name__
        return logging.getLogger(name)

    def __init__(self):

        self.svm_train_path = os.path.join(__file__, 'libsvm-cost', 'svm-train')
        self.svm_predict_path = os.path.join(__file__, 'libsvm-cost', 'svm-predict')
        self.svm_work_path = os.path.join(__file__, 'libsvm-cost', 'work')
        self.svm_model_name = 'libsvm-cose.model'
        self.svm_train_name = 'libsvm-train'
        self.svm_cost_name = 'libsvm-cost'
        self.options_default_value = {
            "svm_type": "0",
            "kernel_type" : "2",
            "loss": "1",
            "degree": "3",
            "gamma": "1/k",
            "coef0": "0",
            "cost": "1",
            "nu": "0.5",
            "epsilon": "0.1",
            "shrinking": "1",
            "probability_estimates": "0",
            "weight": "1",
            "n": "5", # Not sure
            "dense": "0"
        }

        self.options_abbreviation = {
            "svm_type": "-s",
            "kernel_type" : "-k",
            "loss": "l",
            "degree": "-d",
            "gamma": "-g",
            "coef0": "-r",
            "cost": "-c",
            "nu": "-n",
            "epsilon": "-e",
            "shrinking": "-h",
            "probability_estimates": "-b",
            "weight": "-wi",
            "n": "-v",
            "dense": "-D"
        }

        return

    def save(self, model_file_path):
        copyfile(os.path.join(self.svm_work_path, self.svm_model_name), model_file_path)
        return

    def load(self, model_file_path):
        copyfile(model_file_path, os.path.join(self.svm_work_path, self.svm_model_name))
        return 

    def build_options_string(self, options):
    
        if 'loss' in options:
            if options['loss'] in ['3', '4']:
                options['loss'] = "{}.{}".format(options['loss'], self.svm_cost_name)

        options_list = []
        for keyword in self.options_abbreviation.keys():
            if keyword in options:

                option_list.append("{} {}".format(
                    self.options_abbreviation[keyword],
                    options[keyword]
                ))

        return " ".join(option_list)

    def _train(self, options_string, training_file):

        os.system("{svm_train_path} {options} {training_file} {model_file}".format(
            svm_train_path=self.svm_train_path,
            options=options_string,
            training_file=training_file,
            model_file=os.path.join(self.svm_work_path, self.svm_model_name)
        ))

        return

    def gen_libsvm_data(self, training_data, output_dir):
        
        output_name = os.path.join(output_dir, self.svm_train_name)




        return ouput_name

    def gen_libsvm_cost(self, training_data, output_dir):

        output_name = os.path.join(output_dir, self.svm_cost_name)





        return output_name

    def train(self, training_data, options, output_dir):

        training_file = self.gen_libsvm_data(training_data, output_dir)
        cost_file = self.gen_libsvm_cost(training_data, output_dir)
        options_string = self.build_options_string(options)

        self._train(options_string, training_file)

        return

    def predict(self, x):



        return



