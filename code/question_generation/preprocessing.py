# -*- coding: utf-8 -*-

import numpy as np
from parameters import *
from nltk.tokenize import word_tokenize
import nltk
import itertools
import csv

def read_lines(filename):
    """
    Reads the corpus
    """
    return open(filename).read().decode('utf8').split('\n')[:-1]


def embedding_matrix(i2w, word2vec):
	E = np.zeros((config['VOCAB_SIZE'],config['INPUT_EMBEDDING_SIZE']))
	for i in range(config['VOCAB_SIZE']):
		if i >= config['N_SPECIAL_TOKENS'] and i2w[i] in word2vec.vocab:
			E[i:i+1,:] = word2vec[i2w[i]][:config['INPUT_EMBEDDING_SIZE']].T
	return E




def batch_hred(contexts,ans_locs,questions):
    batch=[]
    batch.append([])
    batch.append([])
    batch.append([])
    batch.append([])
    batch.append([])#ans locs
    max_len = 0
    max_conv = 0
    utt_len = []
    #max_c = max(map(len, contexts))
    max_len = config['DECODER_LENGTH']#min(config['DECODER_LENGTH'], max_c)
    for c,a,q in zip(contexts,ans_locs,questions):

        c_ = list(c)
        a_ = list(a)
        q_ = list(q)
        if max_len - len(c) < 0:
            #print("Context too long !")
            continue
        for p in range(max_len - len(c)):
            c_.append(config['PAD'])
        for p in range(max_len - len(a)):
            a_.append(config['PAD'])
        for p in range(max_len - len(q)):
            q_.append(config['PAD'])
        batch[0].append([c_])
        batch[1].append(len(c))
        batch[2].append(q_)
        batch[3].append(1)
        batch[4].append([a_])
    return batch

def intseq(seq, lookup, conf):
    """
    Replaces words with indices in a sequence
    Replaces with unknown if word not in lookup
    Returns [list of indices]
    """
    indices = [conf['BOS']]
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(conf['UNK'])
    indices.append(conf['EOS'])
    return indices

def decode(seq, lookup):
    utt = ""
    for word in seq:
	utt = utt+" "+str(lookup[word].encode("utf-8"))
    return utt

def vectorize_text(lines, vocab_size, conf):
    """
    Transforms utterances into indice sequences with a word2index dictionary OR pretrained word2vec embeddings
    Input

        vocab_size: number of known words (dictionary size)

    Output
        sequences: list of sequences
        index2word: dictionary index-word
        word2index: dictionary word-index
    """
    #Tokenize
    tokenized_sentences = [word_tokenize(s.lower()) for s in lines]

    #Index
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*(tokenized_sentences)))
    print len(freq_dist.keys()), "different words"
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + ['<S>'] + ['<UNK>'] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )

    vecs = [intseq(s, word2index, conf) for s in tokenized_sentences]


    return vecs, index2word, word2index


def read_data(path, index2word):
    # word2index
    word2index = dict([(w, i) for i, w in enumerate(index2word)])
    contexts = []
    ans_locs = []
    questions = []
    with open(path, 'r') as input:
        #TSV (id, tokenized context id seq, answer start, answer end, tokenized question id seq)
        reader = csv.reader(input, delimiter='\t')
        for row in reader:
            c = map(int,row[1].split(","))
            contexts.append(c)
            questions.append(map(int,row[4].split(",")))
            ans_loc = np.zeros(len(c))
            ans_loc[int(row[2]):int(row[3])] = 1
            ans_locs.append(ans_loc)

    return contexts, ans_locs, questions, word2index



######


def process_conversations_cornell(conv_path, lines_path):
    output = open("cornell_movie_corpus/formatted.txt","w")

    convs_raw = read_lines(conv_path)
    convs=[]
    for c in convs_raw:
        line_numbers = c.split(" +++$+++ ")[3].replace("L","").replace("['","").replace("']","").split("', '")
        convs.append(line_numbers)

    lines_raw = read_lines(lines_path)
    lines={}
    for l in lines_raw:
        lines[int(l.split(";")[0].replace("L",""))] = l.split(";")[4]

    for c in convs:
        for l in c:
            output.write((lines[int(l)]+"\n").encode("utf8"))
        output.write("\n")








