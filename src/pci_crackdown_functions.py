import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
import math
import jieba
import numpy as np
import pickle
import sklearn 
import sklearn.preprocessing
from sklearn import linear_model

from keras.preprocessing.sequence import pad_sequences
import os, sys, re, argparse, math, pickle
import sqlite3
import numpy as np
import datetime as dt

from time import time 

import keras 
from keras import backend as K

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, CuDNNLSTM, CuDNNGRU,  GlobalMaxPooling1D, GlobalAveragePooling1D, GRU

import copy, random

from src.pci_crackdown import * 
from src.hyper_parameters import * 


np.random.seed(100)
def proc_embedding(input_file, output_path):
    print('Reading embedding file')
    embedding_raw = pd.read_csv(
        input_file,
        delim_whitespace = True,
        header= None,
        skiprows = 1,
        quoting =3
    )

    dim_embedding = embedding_raw.shape[1] - 1 

    embedding = {}
    for index,i in embedding_raw.iterrows():
        word = i[0]
        coefs = i[1:]
        embedding[word] = coefs

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open( output_path + '/embedding.pkl' , 'wb') as f:
        pickle.dump(embedding, f)

    print('Preparing tokenizer')
    ## Prepare tokenizer
    all_text = [*embedding, 'unk']
    all_text = [i for i in all_text if type(i) is not float]
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(all_text)
    with open(output_path +  '/tokenizer.pkl' , 'wb') as f:
        pickle.dump(tokenizer, f)

    ## Prepare embedding_matrix
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, dim_embedding))
    for word, i in word_index.items():
        embedding_vector = embedding.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    new_vec = np.zeros((embedding_matrix.shape[0],1))
    new_vec[word_index.get('unk')] = 1 

    embedding_matrix = np.concatenate((new_vec, embedding_matrix), axis=1)

    with open(output_path + '/embedding_matrix.pkl' , 'wb') as f:
        pickle.dump(embedding_matrix, f)


def stratify_sample(n,k):
    num_seq = list(range(1,k+1)) * (math.ceil(n/k))
    num_seq = np.random.choice(num_seq, len(num_seq))
    return( np.random.choice(num_seq[0:n], n, replace=False))

def cut(x):
    return " ".join(jieba.cut(x))

def proc_data(data_path, embedding_path, tokenizer_path, create_training_sample, output_path, filename):
    with open(embedding_path , 'rb') as f:
        embedding = pickle.load(f)
    with open(tokenizer_path , 'rb') as f:
        tokenizer = pickle.load(f)

    df = pd.read_pickle(data_path)

    df['strata'] = df.groupby(['id'] )['id'].transform(lambda x: stratify_sample(x.shape[0], 10))

    print("jieba")
    df['sentence'] = df.apply(lambda row: cut(row['sentence']), axis=1)

    df['sentence_seg'] = df.sentence.apply(lambda x : [ word if word in embedding.keys() else 'unk'  for word in text_to_word_sequence(x) ] )
    df['sentence_seg'] = df.sentence_seg.apply(lambda x : " ".join(x) )

    df['sentence_seg'] = df.sentence_seg.apply(text_to_word_sequence)
    df['sentence_seg'] = tokenizer.texts_to_sequences(df.sentence_seg)

    if create_training_sample: 
        tmp = df[df['strata']<=8]
        training_data = {'y': tmp.days_since.values.reshape(-1,1), 'x':tmp['sentence_seg'], 'df':tmp}

        tmp = df[df['strata']>8]
        testing_data = {'y': tmp.days_since.values.reshape(-1,1), 'x':tmp['sentence_seg'], 'df':tmp}

        with open(output_path + '/' + 'training_data_' + filename , 'wb') as f:
            pickle.dump(training_data, f)
        with open(output_path + '/' + 'testing_data_' + filename , 'wb') as f:
            pickle.dump(testing_data, f)
    else: 
        out = {'y': df.days_since.values.reshape(-1,1), 'x':df['sentence_seg'], 'df':df} 
        with open(output_path + '/' + filename , 'wb') as f:
            pickle.dump(out, f)


def compile_results(data_path, model, include_text, output):
    with open(data_path , 'rb') as f:
        data = pickle.load(f)

    tamhk = pci_crackdown.load(model)

    df = data['df']
    df['predict'] = tamhk.model.predict(pad_sequences(data['x'], maxlen=tamhk.pars.varirate['lstm1_max_len'], padding='post', truncating='post'))
    if ~include_text:
        df = df.drop(columns =['sentence', 'sentence_seg'])
    df.to_excel(output)



