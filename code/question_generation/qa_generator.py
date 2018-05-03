import tensorflow as tf
from copynet import *
import gensim
import numpy as np

from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper
from preproc import read_data, embedding_matrix

from generator_copynet import Generator
from parameters import *

# Load and preprocess data


print('Loading and preprocessing...')

i2w = np.load('../../data/voc.npy')

"""
vocab_size = len(i2w)
gen_vocab_size = vocab_size
(contexts, ans, questions, w2i) = read_data("../../data/tsv/squad_tl.tsv", i2w)

print("Loading Google's pre-trained Word2Vec model")
word2vec = gensim.models.KeyedVectors.load_word2vec_format('https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz', binary=True)
"""

word2vec=[]

print('Computing embedding matrix')
embedding_m = embedding_matrix(i2w, word2vec, config['EMBED_SIZE'], config['N_SPECIAL_TOKENS'])
print(embedding_m)

########
G = Generator(SESSION)
contexts=[[6,8,3,4,5,0],[6,8,3,4,5,0]]
ans=[[0,0,1,1,0,0],[0,0,1,1,0,0]]
questions=[[1,5,3,0],[1,5,3,0]]

batch= []
batch.append(contexts)
batch.append(ans)
batch.append([6,6])
batch.append(questions)
batch.append([4,4])

print(G.train_step_supervised(batch,0.01,embeddings_m))