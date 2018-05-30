# -*- coding: utf-8 -*-
import io
import os
import json
import unicodecsv as csv
from nltk.tokenize import word_tokenize
from uuid import uuid1
import nltk
import numpy as np
import itertools
import gensim



"""
PAD : 0
EOS : 1
UNK : 2

Output TSV: (id, tokenized context id seq, answer start, answer end, question) + voc.npy (+ embedding_matrix.npy)
"""

#Load SQUAD Json
squad = json.load(io.open("../../data/squad/train-v1.1.json", encoding="utf-8"))

def intseq(seq, lookup):
    """
    Replaces words with indices in a sequence
    Replaces with unknown if word not in lookup
    Returns [list of indices]
    """
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(2)
    indices.append(0)
    return indices

LIMIT = len(squad['data'])
LIMIT = 20

#Création du dictionnaire
print("Creating vocabulary from SQuAD")
tokenized_sentences = []
for E in squad['data'][:LIMIT]:
    for P in E['paragraphs']:
        for QA in P['qas']:
            for A in QA['answers']:
                context = word_tokenize(P['context'])
                question = word_tokenize(QA['question'])
                ans = word_tokenize(A['text'])
                tokenized_sentences.append(context)
                tokenized_sentences.append(ans)

# get frequency distribution
freq_dist = nltk.FreqDist(itertools.chain(*(tokenized_sentences)))
print len(freq_dist.keys()), "different words"
# get vocabulary of 'vocab_size' most used words
vocab = freq_dist.most_common(len(freq_dist.keys()))
# index2word
index2word = ['_'] + ['<BOS>'] + ['<UNK>'] + [x[0] for x in vocab]


np.save("../../data/voc2.npy",index2word)

# word2index
word2index = dict([(w, i) for i, w in enumerate(index2word)])


c = 0
with open('../../data/tsv/squad_tl2.tsv', 'w') as output:
    writer = csv.writer(output, delimiter='\t')
    for E in squad['data'][:LIMIT]:
        for P in E['paragraphs']:
            for QA in P['qas']:
                for A in QA['answers']:
                    context = word_tokenize(P['context'])
                    question = word_tokenize(QA['question'])
                    ans = word_tokenize(A['text'])
                    print(context,question,ans)
                    print("---------------------------")
                    i = 0
                    while context[i:i+len(ans)] != ans and i < len(context):
                        i += 1
                    if i < len(context):
                        writer.writerow([uuid1(), ','.join(map(str,intseq(context,word2index))), i, i+len(ans), ','.join(map(str,intseq(question,word2index)))])
                    else:
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Nope")
        c += 1
        print(str(c)+" / "+str(len(squad['data'])))


print('Done!')