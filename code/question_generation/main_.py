#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

import pickle
import gensim

import pandas as pd
from random import shuffle

import numpy as np
np.random.seed(19)

import tensorflow as tf
tf.set_random_seed(3)

import math
import os
import time
import sys
from nltk.tokenize import word_tokenize

from parameters import *

from preprocessing import *

from generator_copynet import Generator

# Load and preprocess data

print('Loading and preprocessing...')

i2w = np.load('../../data/voc.npy')
(contexts, ans, questions, w2i) = read_data("../../data/tsv/squad_tl.tsv", i2w)


if config['PRETRAINED_EMBEDDINGS']:
    print("Loading Google's pre-trained Word2Vec model")
    word2vec = gensim.models.KeyedVectors.load_word2vec_format('https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz', binary=True)

    print('Computing embedding matrix')
    embedding_m = embedding_matrix(i2w, word2vec)
    G = Generator(SESSION, pretrainedEmbeddings=embeddings_m)
else:
    G = Generator(SESSION)

"""
count = map(lambda x: len(x)<200, contexts)
print(np.sum(count), len(contexts))
sys.exit("")
"""
"""
n=10
contexts=contexts[:n]
ans_locs=ans_locs[:n]
questions=questions[:n]
"""


if os.path.exists('sessions/' + str(SESSION)) == False:
    os.system('mkdir sessions/' + str(SESSION))

pickle.dump(i2w, open('sessions/' + str(SESSION) + '/i2w.p', 'wb'))
pickle.dump(w2i, open('sessions/' + str(SESSION) + '/w2i.p', 'wb'))

print(len(contexts), 'examples in train set')

# Pretrained word embeddings
if config['PRETRAINED_EMBEDDINGS']:

    # Load Google's pre-trained Word2Vec model.
    print("Loading Google's pre-trained Word2Vec model")
    word2vec = \
        gensim.models.KeyedVectors.load_word2vec_format('https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
            , binary=True)

    print('Computing embedding matrix')
    embeddings = embedding_matrix(i2w, word2vec)

    G = Generator(SESSION, pretrainedEmbeddings=embeddings)
else:
    G = Generator(SESSION)

# Train

batch_size = 40  # Number of examples by batch
batch_size_test = 5  # Number of examples by batch
n_epoch = 10
training_iters = int(math.ceil(len(contexts) / batch_size))
t0 = time.time()

def train_batch(
    e,
    it,
    start,
    end,
    lr_,
    tf_,
    ):
    batch = batch_hred(contexts[start:end], ans_locs[start:end], questions[start:end])
    if batch[2] == []:
        print("Empty batch")
        return 0
    try:
        """
        batch = [[],[],[],[],[]]
        for i in range(batch_size):
            batch[0].append([[k for k in range(200)]])
            batch[1].append(200)
            batch[2].append([k for k in range(200)])
            batch[3].append(1)
            batch[4].append([[0 for k in range(200)]])
        """
        loss=(G.train_step_supervised(batch,lr_,tf_))
        print("T: "+str(int(time.time()-t0))+" | Epoch "+str(e)+" | Batch "+str(it+1)+" / "+str(training_iters)+" | loss: "+str(loss)+" | perplexity: "+str(math.exp(loss))+" | lr: "+str(lr_)+" | teacher forcing: "+str(tf_))
        return loss
    except e:
        print("Error on batch: ",e)
        return 0


def train_supervised():

    # Load last trained model

    if RESUME_TRAINING and os.path.exists('ckpt/' + str(SESSION)
            + '/last_ckpt.npy'):
        if RESUME_CKPT == 0:
            ckpt = np.load('ckpt/' + str(SESSION) + '/last_ckpt.npy')
            lr = np.load('ckpt/' + str(SESSION) + '/lr.npy')
        else:
            ckpt = RESUME_CKPT
            lr = RESUME_LR
            if lr == 0:
                lr = np.load('ckpt/' + str(SESSION) + '/lr.npy')

        G.load_model(ckpt)
        teacher_forcing = np.load('ckpt/' + str(SESSION) + '/tf.npy')
        L_ = np.load('sessions/' + str(SESSION) + '/loss_test.npy'
                     ).tolist()[:ckpt]
        L_train = np.load('sessions/' + str(SESSION) + '/loss_train.npy'
                          ).tolist()[:ckpt]
    else:
        ckpt = 0
        lr = config['INITIAL_LR']
        teacher_forcing = config['INITIAL_TF']
        L_ = []
        L_train = []

    print('Supervised training with ' + str(batch_size)
          + ' examples / batch...')
    for e in range(1 + ckpt, n_epoch + 1 + ckpt):
        L_train_ = 0
        print('Epoch ' + str(e))
        for it in range(training_iters):
            l = train_batch(
                e,
                it,
                batch_size * it,
                batch_size * (it + 1),
                lr,
                teacher_forcing,
                )
            L_train_ += l / training_iters

        L_train.append(L_train_)
        np.save('sessions/' + str(SESSION) + '/loss_train.npy', L_train)

        lr = config['INITIAL_LR'] / (1. + float(e) / float(config['N_EPOCH_HALF_LR']))

        teacher_forcing = config['INITIAL_TF'] / (1. + float(e)
                / float(config['N_EPOCH_HALF_TF']))
        G.save_model(e)
        np.save('ckpt/' + str(SESSION) + '/lr.npy', lr)
        np.save('ckpt/' + str(SESSION) + '/tf.npy', teacher_forcing)


"""train_batch(
    0,
    0,
    0,
    batch_size,
    0,
    0,
    )
"""
train_supervised()




"""batch = [
    [[[1,2,4,4,4,3,1],[1,2,4,4,4,3,1],[0,0,0,0,0,0,0]], [[1,6,6,6,6,1,0],[1,6,6,6,6,1,0],[1,6,6,6,6,1,0]]],
    [7,7,0,6,6,6],
    [[1,4,4,4,4,5,1], [1,4,4,4,4,5,1]],
    [2, 3]
    ]"""
