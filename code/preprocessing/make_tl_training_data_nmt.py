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

#Load SQUAD Json
squad_train = json.load(io.open("../../data/squad/train-v1.1.json", encoding="utf-8"))
squad_dev = json.load(io.open("../../data/squad/dev-v1.1.json", encoding="utf-8"))

def sentence(seq, start_ans=-1, end_ans=-1):
    s=""
    i=0
    for word in seq:
        if i >= start_ans and i < end_ans:
            suffix = "_1"
        else:
            suffix = "_0"
        i += 1
        s = s+word+suffix+" "
    return s


vocab_size = 50000

#Création du dictionnaire
print("Creating vocabulary from SQuAD")
tokenized_sentences, ts_ans = [], []
for E in squad_train['data']:
    for P in E['paragraphs']:
        for QA in P['qas']:
            for A in QA['answers']:
                ans = word_tokenize(A['text'].lower())
                ts_ans.append(ans)
            context = word_tokenize(P['context'].lower())
            question = word_tokenize(QA['question'].lower())
            tokenized_sentences.append(context)
            tokenized_sentences.append(question)

# get frequency distribution
freq_dist = nltk.FreqDist(itertools.chain(*(tokenized_sentences)))
print len(freq_dist.keys()), "different words"
# get vocabulary of 'vocab_size' most used words
vocab = freq_dist.most_common(vocab_size)

freq_dist = nltk.FreqDist(itertools.chain(*(ts_ans)))
print len(freq_dist.keys()), "different answer words"
# get vocabulary of 'vocab_size' most used words
vocab_ans = freq_dist.most_common(len(freq_dist.keys()))
# index2word
index2word = ['<unk>'] + ['<s>'] + ['</s>'] + ['unk_1'] + [x[0]+"_0" for x in vocab] + [x[0]+"_1" for x in vocab_ans]

with open('../../data/squad/voc.doc', 'w') as output, open('../../data/squad/voc.q', 'w') as output2:
    writer = csv.writer(output, delimiter='\t')
    writer2 = csv.writer(output2, delimiter='\t')
    for k in range(len(index2word)):
        writer.writerow([index2word[k]])
        writer2.writerow([index2word[k]])

np.save("../../data/squad/voc.npy",index2word)


def write_data(data,name):
    c = 0
    with open('../../data/squad/'+name+'.doc', 'w') as output_doc, open('../../data/squad/'+name+'.q', 'w') as output_q:
        writer_doc = csv.writer(output_doc, delimiter='\t')
        writer_q = csv.writer(output_q, delimiter='\t')
        for E in data:
            for P in E['paragraphs']:
                for QA in P['qas']:
                    for A in QA['answers']:
                        context = word_tokenize(P['context'])
                        question = word_tokenize(QA['question'])
                        ans = word_tokenize(A['text'])
                        i = 0
                        while context[i:i + len(ans)] != ans and i < len(context):
                            i += 1
                        if i < len(context):
                            writer_doc.writerow([sentence(context,i, i+len(ans))])
                            writer_q.writerow([sentence(question)])
            c += 1
            print(str(c)+" / "+str(len(data)))

write_data(squad_train['data'],"train")
write_data(squad_dev['data'][:int(len(squad_dev['data'])/1.1)],"test")
write_data(squad_dev['data'][int(len(squad_dev['data'])/1.1):],"dev")
print('Done!')