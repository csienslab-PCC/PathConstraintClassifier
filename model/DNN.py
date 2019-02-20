#!/usr/bin/env python2


import sys
import logging

import math
import time
import numpy as np
import pandas as pd
import keras.backend as K


from IPython import embed
from keras.utils import np_utils
from keras.regularizers import *
from keras.optimizers import Adam
from keras.models import Sequential, load_model, save_model
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

from ModelException import *

def f1_score(y_true,y_pred):
    thresh = 0.2
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float64')
    tp = K.sum(y_true * y_pred,axis=-1)

    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

def time_loss(y_true, y_pred):

    print 'y_true:', y_true
    print 'y_pred:', y_pred


    loss = K.sum(y_true * y_pred, axis=-1)
#    return loss * (-1)
    return loss

def scale(x, max_abs_time, min_t):

    if min_t == max_abs_time:
        return 0

    return (math.log(x) - math.log(min_t)) / (math.log(max_abs_time) - math.log(min_t)) 

def normalize(x, max_t, min_t):

    if max_t == min_t:
        return 0

    return float(x - min_t) / (max_t - min_t)


class DNNModel(object):

    @property
    def logger(self):
        name = self.__class__.__name__
        return logging.getLogger(name)

    def __init__(self, output_model_name="DNN.model", class_num=7):

        self.class_num = class_num
        self.output_model_name = output_model_name
        self._model = None
        self.dim = 0
        self.feature_num = -1
        self.prediction_time = 0
        self.avg_prediction_time = 0
        return

    def get_model(self):
        return self._model

    def load(self, model_file_path):
        self._model = load_model(model_file_path, custom_objects={'f1_score': f1_score})
        return

    def save(self, model_file_path):
        save_model(self._model, model_file_path)
        return

    def _delete(self, training):

        indices = np.array([])
        for i, f in enumerate(training['z']):
            if all(f.astype(float) == 1000):
                indices = np.append(indices, i)
                #print f

        #embed()
        for symbol in ['x', 'm']:
            training[symbol] = np.delete(training[symbol], indices, axis=0)
       
        for symbol in ['y', 'z']:
            training[symbol] = np.delete(training[symbol], indices, axis=0)
        
        return 

    def _shuffle(self, training):

        np.random.seed(1126)
        indices = np.arange(len(training['y']))
        np.random.shuffle(indices)
        for symbol in ['x', 'y', 'z', 'm', 't']:
            training[symbol] = training[symbol][indices]
        return

    def process_data(self, training):
        
        training = dict([(key, training[key]) for key in ['x', 'y', 'z', 'm', 't']])
#        self._delete(training)
        self._shuffle(training)

        x = training['x']
        y = training['y']
        z = training['z']
        m = training['m']
        t = training['t']

        try:
            y = np_utils.to_categorical(y, self.class_num)
        except:
            print 'QQ'
            embed()

        self.feature_num = len(x[0])
        self.dim = x.shape[1]
        sp = x.shape[0]*7//8

        x_train = x[:sp]
        y_train = y[:sp]
        x_test = x[sp:]
        y_test = y[sp:]

#        return x_train, y_train, x_test, y_test
        return x, y, x, y

    def build_model(self):

        dim = self.dim
        rate = 0.1
        model = Sequential()
        model.add(Dense(units=512, activation='relu', input_shape=(dim,)))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        #model.add(Dropout(rate))
        model.add(Dense(units=512, activation='relu'))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        model.add(Dropout(rate))
        model.add(Dense(units=512, activation='relu'))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        #model.add(Dropout(rate))
        model.add(Dense(units=512, activation='relu'))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        model.add(Dropout(rate))
        model.add(Dense(units=512, activation='relu'))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        #model.add(Dropout(rate))
        model.add(Dense(units=512, activation='relu'))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        model.add(Dropout(rate))
        model.add(Dense(units=256, activation='relu'))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        #model.add(Dropout(rate))
        model.add(Dense(units=self.feature_num, activation='sigmoid'))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        model.add(Dropout(rate))
        model.add(Dense(units=self.class_num, activation='softmax'))
#        model.add(Dense(units=self.class_num))
        model.summary()

        opt = Adam(lr=1e-5)
        # use f1
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', f1_score])
    #    model.compile(loss='mse', optimizer=opt, metrics=['accuracy', f1_score])
        return model

    def _train(self, model, x_train, y_train, batch_size):

#        print x_train
#        raw_input()
#        print y_train
#        raw_input()

        checkpoint = ModelCheckpoint(self.output_model_name, monitor='val_f1_score')
        earlystopping = EarlyStopping(monitor='val_f1_score', patience = 30, verbose=1, mode='max')
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1000, validation_split=0.1, callbacks=[checkpoint, earlystopping])
        #model.fit(x_train, y_train, batch_size=batch_size, epochs=1000, validation_split=0.1, callbacks=[checkpoint])
        return 

    def _test(self, x_test, y_test):

        print self.output_model_name
        model = load_model(self.output_model_name, custom_objects={'f1_score': f1_score})
        pre = model.predict(x_test)

        f = open('submit.csv', 'w')
        f.write('id,label\n')
        cnt = 0
        for i, a in enumerate(pre):
            ans = np.where(a == a.max())[0][0]
            if (y_test[i][ans] == 1): cnt += 1
            f.write('{},{}\n'.format(i, ans))
        print 'acc:', cnt/float(pre.shape[0])
        f.close()
        return

    def train(self, training_data):

        x_train, y_train, x_test, y_test = self.process_data(training_data)
        print x_train, y_train
        
        model = self.build_model()
        self._train(model, x_train, y_train, batch_size=128)
        self._model = model

        print 'train done'

        self._test(x_test, y_test)
        return 

    def predict_proba(self, data):

        if self._model == None:
            raise EmptyModelError("DNN")

        to_predict_data = np.array([x for x in data['x']])

        start = time.time()
#        ret = self._model.predict(data['x'])
        ret = self._model.predict(to_predict_data)
        end = time.time()

        self.prediction_time = end - start
        self.avg_prediction_time = float(self.prediction_time) / len(data['x'])
        return ret

    def predict(self, data):

        ans = self.predict_proba(data)
        ret = [np.argmax(x) for x in ans]
        return ret

class DNNModelAlpha(DNNModel):


    def __init__(self, output_model_name='DNNAlpha.model', class_num=7):

        super(DNNModelAlpha, self).__init__(output_model_name, class_num)
        
        self.max_loss = 100

        return

    def evaluate_loss(self, T_solver, T_max, T_min, solve):

        max_loss = self.max_loss
        if not solve:
           return max_loss

        return (max_loss / 2) * (T_solver - T_min) / (T_max - T_min)

    def normalize(self, x, max_t, min_t):

        if max_t == min_t:
            return 0

        return float(x - min_t) / (max_t - min_t)

    def gen_score(self, solver_result):

        max_loss = self.max_loss
        time_out = 100

        print solver_result

        loss_vec = [0 for x in solver_result]
        for i, sr in enumerate(solver_result):
            if sr['error']:
                loss_vec[i] += max_loss * 3 
            elif sr['time'] >= time_out:
                loss_vec[i] += max_loss * 2
            elif sr['time'] < time_out and not sr['answer']:
                loss_vec[i] += max_loss * 2
            
        solve_time_list = sorted([
            (i, sr['time']) for i, sr in enumerate(solver_result) if sr['answer']
        ], key=lambda x : x[1])
        print solve_time_list
        if len(solve_time_list) >= 2:
            
            max_time = solve_time_list[-1][1]
            min_time = solve_time_list[0][1]
            dist = max_time - min_time

            if dist != 0:
                for s in solve_time_list:
#                    loss_vec[s[0]] += (float(max_loss) / 2) * (s[1] - min_time) / dist
                    loss_vec[s[0]] += (float(max_loss) / 2) * self.normalize(s[1], max_time, min_time)


        
        print loss_vec
        return loss_vec

    def process_data(self, training):
        
        training = dict([(key, training[key]) for key in ['x', 'y', 'z', 'm', 't']])
#        self._delete(training)
        self._shuffle(training)

        x = training['x']
        y = training['y']
        z = training['z']
        m = training['m']
        t = training['t']

        y = np_utils.to_categorical(y, self.class_num)
        new_y = []
        for i, yy in enumerate(y):
            
            print i, yy
            new_y.append(self.gen_score(z[i]))
            print new_y[-1]
        
#        raw_input('press')
        y = np.array(new_y)
        print x

        self.feature_num = len(x[0])
        self.dim = x.shape[1]
        sp = x.shape[0]*7//8

        x_train = x[:sp]
        y_train = y[:sp]
        x_test = x[sp:]
        y_test = y[sp:]

#        return x_train, y_train, x_test, y_test
        return x, y, x, y

    def load(self, model_file_path):
        self._model = load_model(model_file_path, custom_objects={'f1_score': f1_score, 'time_loss': time_loss})
        return

    def build_model(self):

        dim = self.dim
        rate = 0.1
        model = Sequential()
        model.add(Dense(units=512, activation='relu', input_shape=(dim,)))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        #model.add(Dropout(rate))
        model.add(Dense(units=512, activation='relu'))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        model.add(Dropout(rate))
        model.add(Dense(units=512, activation='relu'))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        #model.add(Dropout(rate))
        model.add(Dense(units=512, activation='relu'))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        model.add(Dropout(rate))
        model.add(Dense(units=512, activation='relu'))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        #model.add(Dropout(rate))
        model.add(Dense(units=512, activation='relu'))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        model.add(Dropout(rate))
        model.add(Dense(units=256, activation='relu'))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        #model.add(Dropout(rate))
        model.add(Dense(units=self.feature_num, activation='sigmoid'))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        model.add(Dropout(rate))
        model.add(Dense(units=self.class_num, activation='softmax'))
        model.summary()

        opt = Adam(lr=1e-5)
        # use f1
        model.compile(loss=time_loss, optimizer=opt, metrics=['accuracy', f1_score])
    #    model.compile(loss='mse', optimizer=opt, metrics=['accuracy', f1_score])
        return model

    def _test(self, x_test, y_test):

        print self.output_model_name
        model = load_model(self.output_model_name, custom_objects={'f1_score': f1_score, 'time_loss': time_loss})
        pre = model.predict(x_test)

        f = open('submit.csv', 'w')
        f.write('id,label\n')
        cnt = 0
        for i, a in enumerate(pre):
            ans = np.where(a == a.max())[0][0]
            if (y_test[i][ans] == 1): cnt += 1
            f.write('{},{}\n'.format(i, ans))
        print 'acc:', cnt/float(pre.shape[0])
        f.close()
        return


class DNNModelBeta(DNNModelAlpha):

    def __init__(self, output_model_name='DNNBeta.model', class_num=7):

        super(DNNModelBeta, self).__init__(output_model_name, class_num)

        self.max_loss = 100
        return

    def evaluate_loss(self, T_solver, T_max, T_min, solve):

        max_loss = self.max_loss
        if not solve:
           return max_loss

        return (max_loss / 2) * (T_solver - T_min) / (T_max - T_min)

    def scaling(self, x):
        
        factor = max(math.log10(x), -6) + 7.0
        return factor

    def gen_score(self, solver_result):

        max_loss = float(self.max_loss)
        time_out = 100.0

        print solver_result

        loss_vec = [0 for x in solver_result]
        for i, sr in enumerate(solver_result):
            if sr['time'] > time_out:
                loss_vec[i] += max_loss

            elif sr['error']:
                loss_vec[i] += max_loss * 2
            
        solve_time_list = sorted([
            (i, sr['time']) for i, sr in enumerate(solver_result) if sr['answer']
        ], key=lambda x : x[1])
        print solve_time_list

        if len(solve_time_list) >= 2:
            
            max_t = solve_time_list[-1][1]
            min_t = solve_time_list[0][1]
            dist = max_t - min_t
                
#            dist = math.log10(float(max_time) / min_time)

            if dist != 0:
                for s in solve_time_list:
                    loss_vec[s[0]] += (max_loss / 2) * scale(s[1], time_out, min_t) * normalize(s[1], max_t, min_t)

        loss_vec = [x for x in loss_vec]
        print loss_vec

        return loss_vec

class DNNModelGamma(DNNModel):

    def __init__(self, output_model_name='DNNGamma.model', class_num=7):

        super(DNNModelGamma, self).__init__(output_model_name, class_num)
        
        self.max_loss = 100

        return

    def gen_score(self, solver_result):

        return [sr['time'] for sr in solver_result]

    def process_data(self, training):
     
#        embed()

        training = dict([(key, training[key]) for key in ['x', 'y', 'z', 'm', 't']])
#        self._delete(training)
        self._shuffle(training)

        x = training['x']
        y = training['y']
        z = training['z']
        m = training['m']
        t = training['t']

        y = np_utils.to_categorical(y, self.class_num)
        new_y = []
        for i, yy in enumerate(y):
            
            print i, yy
            new_y.append(self.gen_score(z[i]))
            print new_y[-1]
        
#        raw_input('press')
        y = np.array(new_y)
        print x

        self.feature_num = len(x[0])
        self.dim = x.shape[1]
        sp = x.shape[0]*7//8

        x_train = x[:sp]
        y_train = y[:sp]
        x_test = x[sp:]
        y_test = y[sp:]

#        return x_train, y_train, x_test, y_test
        return x, y, x, y

    def load(self, model_file_path):
        self._model = load_model(model_file_path, custom_objects={'f1_score': f1_score, 'time_loss': time_loss})
        return

    def build_model(self):

        dim = self.dim
        rate = 0.1
        model = Sequential()
        model.add(Dense(units=512, activation='relu', input_shape=(dim,)))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        #model.add(Dropout(rate))
        model.add(Dense(units=512, activation='relu'))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        #model.add(Dropout(rate))
        model.add(Dense(units=512, activation='relu'))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        #model.add(Dropout(rate))
        model.add(Dense(units=512, activation='relu'))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        #model.add(Dropout(rate))
        model.add(Dense(units=512, activation='relu'))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        #model.add(Dropout(rate))
        model.add(Dense(units=512, activation='relu'))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        #model.add(Dropout(rate))
        model.add(Dense(units=512, activation='relu'))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        #model.add(Dropout(rate))
#        model.add(Dense(units=self.feature_num, activation='sigmoid'))#, kernel_regularizer=l2(1e-7)))
        model.add(Dense(units=256, activation='tanh'))#, kernel_regularizer=l2(1e-7)))
        model.add(BatchNormalization())
        #model.add(Dropout(rate))
        model.add(Dense(units=self.class_num))
        model.summary()

        opt = Adam(lr=1e-4)
        # use f1
        model.compile(loss='mean_squared_error', optimizer=opt)
        return model

    def _test(self, x_test, y_test):

        print self.output_model_name
        model = load_model(self.output_model_name, custom_objects={'f1_score': f1_score, 'time_loss': time_loss})
        pre = model.predict(x_test)

        f = open('submit.csv', 'w')
        f.write('id,label\n')
        cnt = 0
        for i, a in enumerate(pre):
            ans = np.where(a == a.max())[0][0]
            if (y_test[i][ans] == 1): cnt += 1
            f.write('{},{}\n'.format(i, ans))
        print 'acc:', cnt/float(pre.shape[0])
        f.close()
        return

    def _train(self, model, x_train, y_train, batch_size):

        checkpoint = ModelCheckpoint(self.output_model_name, monitor='val_loss', period=1)
        earlystopping = EarlyStopping(monitor='val_loss', patience = 10, verbose=1)
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1000, validation_split=0.1, callbacks=[checkpoint, earlystopping])
        #model.fit(x_train, y_train, batch_size=batch_size, epochs=1000, validation_split=0.1, callbacks=[checkpoint])
        return 

    def predict(self, data):

        ans = self.predict_proba(data)
        
        ans = [[t + 10000 if t < 0 else t for t in x]  for x in ans]

        for a in ans:
            print "time_predict:", a
            print "ans:", np.argmin(a)

        ret = [np.argmin(x) for x in ans]
        return ret


class DNNModelError(DNNModelAlpha):
    
    def __init__(self, output_model_name='DNNError.model', class_num=7):

        super(DNNModelError, self).__init__(output_model_name, class_num)
        return
   
    def gen_score(self, solver_result):
        
        print 'XDD'

        ret = [1 if not sr['error'] else 0 for sr in solver_result]
        print ret
        return ret

def test_DNNModelError():
    
    data = np.load(sys.argv[1])
    DNNE = DNNModelError()
    DNNE.train(data)
    DNNE.save('./DNNError.model')
    ans = DNNE.predict(data)

    embed()

    return

if __name__ == '__main__':

    test_DNNModelError()

    data = np.load(sys.argv[1])
    DNN = DNNModelAlpha()
    DNN.train(data)
    ans = DNN.predict(data['x']) 

    print ans

