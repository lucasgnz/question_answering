# -*- coding: utf-8 -*-

import json
import spacy
import io
import unicodecsv as csv
nlp = spacy.load('en')

squad = json.load(io.open("../../data/squad/train-v1.1.json", encoding="utf-8"))


def NER(t):
    T = nlp(t)
    for ent in T.ents:
        t = t.lower().replace(ent.text.lower(), "ENT_"+ent.label_)
    return t


total = 0
for E in squad['data']:
    for P in E['paragraphs']:
        for QA in P['qas']:
            for A in QA['answers']:
                total += 1

c=0
with open('../../data/tsv/squad_tl.tsv', 'w') as output:
    writer = csv.writer(output, delimiter='\t')
    for E in squad['data']:
        for P in E['paragraphs']:
            for QA in P['qas']:
                for A in QA['answers']:
                    context = NER(P['context'])
                    question = NER(QA['question'])
                    ans = NER(A['text'])
                    writer.writerow([context,question,ans])
                    c += 1
                    print(c,total)

print('Done!')