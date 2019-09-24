import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
import math
import jieba as jieba
import numpy as np
import pickle
from time import time
import keras.models
import keras 
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, CuDNNLSTM, CuDNNGRU,  GlobalMaxPooling1D, GlobalAveragePooling1D, GRU
from keras.preprocessing.text import text_to_word_sequence
import copy
import matplotlib.pyplot as plt 
import random

from src.hyper_parameters import * 

def model_fun(hyper_pars,gpu=0):
    pars = hyper_pars.varirate
    input_title = Input(shape=(pars['lstm1_max_len'],))
    
    with open('data/output/embeddings/embedding_matrix.pkl', 'rb') as f:
        embedding_matrix = pickle.load(f)
        
    embedding_matrix= embedding_matrix[:,:(hyper_pars.varirate['n_embedding']+1)]
    
    net_title = Embedding(embedding_matrix.shape[0] ,
            embedding_matrix.shape[1],
            weights=[embedding_matrix],
            input_length=pars['lstm1_max_len'],
            trainable=False)(input_title)

    for i in range(1, pars['lstm1_layer']):
        if gpu != "-1":
            net_title = CuDNNGRU(pars['lstm1_neurons'], return_sequences=True)(net_title)
        else:
            net_title = GRU(pars['lstm1_neurons'], return_sequences=True)(net_title)

    if gpu != "-1":
        net_title = CuDNNGRU(pars['lstm1_neurons'])(net_title)
    else:
        net_title = GRU(pars['lstm1_neurons'])(net_title)

    net_title = Dropout(pars['lstm1_dropout'])(net_title)

    for i in range(1,pars['fc_layer']+1):
        net_title = Dense(pars['fc_neurons'], activation='relu')(net_title)
        net_title = Dropout(pars['fc_dropout'])(net_title)
    net_title = Dense(1)(net_title)

    out = keras.models.Model(inputs=[input_title], outputs=[net_title] )

    return out


def get_fixed(root = "./"):
    fixed = {
                'batch_size': 256,
                'patience' : 2,
                'epochs' : 50,
                'train' : 'data/output/training_data.pkl', 
                'test' : 'data/output/testing_data.pkl', 
                'predict' : 'data/output/prediction_data_HK2019.pkl', 
                'model_folder' : 'model/',
                'mod_id' : str(round((time())))
            }    
    return fixed


def gen_hyper_pars(root = "./"):
    x = hyper_parameters(
        varirate ={
            'lstm1_max_len' : 30,
            'lstm1_neurons' : 30 ,
            'lstm1_dropout' : 0.1 ,
            'lstm1_layer' : 5,
            'fc_neurons' : 30,
            'fc_dropout' : 0.3,
            'fc_layer' : 2,
            'max_words' : 10000,
            'lr' : 0.002,
            'n_embedding' : 150,
            'decay': 0.0001,
            'w': 0.3
        },
        fixed = get_fixed(root)
    )
    return x


class pci_crackdown():
    def __init__(self, pars=gen_hyper_pars(), mm = None, loss = None):
        self.set_pars(pars)
        self.loss = loss
        if mm is not None:
            self.model = mm

    def set_pars(self, pars):
        self.pars = pars
        self.model = model_fun(self.pars)

    def save(self,file=None):
        mm = self.model
        mm.save(file +'.model')
        self.pars.save(file + ".pars")
        with open(file +'.loss', 'wb') as f:
            pickle.dump(self.loss, f)

    def set_loss(self, loss):
        self.loss = loss
    
    @staticmethod
    def load(file):
        mm = keras.models.load_model(file + '.model')
        pars = hyper_parameters_load(file + '.pars')
        with open(file +'.loss', 'rb') as f:
            loss = pickle.load(f)
        x = pci_crackdown(pars = pars, mm = mm, loss = loss )
        return x

    def run(self, df_train, df_test):
        self.model.compile(
            loss='mean_absolute_error', 
            optimizer=keras.optimizers.adam(
                self.pars.varirate['lr'], 
                decay= self.pars.varirate['decay'])
            )

        self.model.fit(
            self.get_x_seq(df_train['x']),
            df_train['y'], 
            batch_size=self.pars.fixed['batch_size'], 
            epochs=self.pars.fixed['epochs'],
            validation_data=(self.get_x_seq(df_test['x']),df_test['y']) , 
            shuffle=True, 
            verbose = 0)

        # print(self.model.evaluate(self.get_x_seq(df_train['x']) ,df_train['y']) )
        # print(self.model.evaluate(self.get_x_seq(df_test['x']) ,df_test['y']))
        self.set_loss(self.model.evaluate(self.get_x_seq(df_test['x']) ,df_test['y']))

    def get_x_seq(self,x):
        return pad_sequences(x, maxlen=self.pars.varirate['lstm1_max_len'], padding='post', truncating='post')


    def sa(self, df_train, df_test, T=0.05, discount=0.05, bandwidth = 0.05, period = 1):
        base = pci_crackdown(self.pars)
        base.run(df_train, df_test)

        best = base.loss

        prev_pars = base.pars
        prev = base.loss

        for i in range(period):
            new = pci_crackdown(update_hyper_pars(prev_pars,bandwidth))
            new.run(df_train, df_test)
            if new.loss - T * (1/(1+ i * discount)) < prev:
                prev_pars = new.pars
                prev = new.loss
                print("updated prev")
            if new.loss  < best:
                best_model = new
                best = new.loss
                print("updated best")

        print(best)

        if best < self.loss:
            g=open("model/best.txt", "a+")
            g.write(str(best)+"\n")
            g.close()
            if best == base.loss:
                best_model = base
            return best_model
        else:
            return None


def gen_candidate(x, bandwidth=0.05, type='int', min_value = None, max_value = None):
    r = random.uniform(-bandwidth, bandwidth)
    new_x = x * (1+r)

    if type == 'int':
        if x * bandwidth < 1 :
            new_x = x + random.choice([-1,0,1])
        else:
            new_x = round(new_x) 

    if min_value != None:
        new_x = max(new_x, min_value)       

    if max_value != None:
        new_x = min(new_x, max_value)    

    return(new_x)   


def update_hyper_pars(hyper_pars, bandwidth= 0.05):
    v = copy.deepcopy(hyper_pars.varirate)
    v['lstm1_max_len']    =  gen_candidate( v['lstm1_max_len']   , bandwidth = bandwidth , type = 'int' , min_value = 1)
    v['lstm1_neurons']    =  gen_candidate( v['lstm1_neurons']   , bandwidth = bandwidth , type = 'int' , min_value = 1)
    v['lstm1_dropout']    =  gen_candidate( v['lstm1_dropout']   , bandwidth = bandwidth , type = '' , min_value = 0, max_value=0.99)
    v['lstm1_layer']      =  gen_candidate( v['lstm1_layer']     , bandwidth = bandwidth , type = 'int' , min_value = 1)
    v['fc_neurons']       =  gen_candidate( v['fc_neurons']      , bandwidth = bandwidth , type = 'int' , min_value = 1)
    v['fc_dropout']       =  gen_candidate( v['fc_dropout']      , bandwidth = bandwidth , type = '' , min_value = 0, max_value=0.99)
    v['fc_layer']         =  gen_candidate( v['fc_layer']        , bandwidth = bandwidth , type = 'int' , min_value = 1)
    v['max_words']        =  gen_candidate( v['max_words']       , bandwidth = bandwidth , type = 'int' , min_value = 1)
    v['lr']               =  gen_candidate( v['lr']              , bandwidth = bandwidth , type = '' , min_value = 0.000001)
    v['n_embedding']      =  gen_candidate( v['n_embedding']     , bandwidth = bandwidth , type = 'int' , min_value = 1, max_value = 300)
    v['decay']            =  gen_candidate( v['decay']           , bandwidth = bandwidth , type = '' , min_value = 0)
    # print(v)

    f = copy.deepcopy(hyper_pars.fixed)
    f['mod_id'] = str(round((time())))
    return hyper_parameters(v, f) 



