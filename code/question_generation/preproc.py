import unicodecsv as csv
import numpy as np

def read_data(path, index2word):
  # word2index
  word2index = dict([(w, i) for i, w in enumerate(index2word)])
  contexts = []
  ans_locs = []
  questions = []
  with open(path, 'r') as input:
    # TSV (id, tokenized context id seq, answer start, answer end, tokenized question id seq)
    reader = csv.reader(input, delimiter='\t')
    for row in reader:
      c = map(int, row[1].split(","))
      contexts.append(c)
      questions.append(map(int, row[4].split(",")))
      ans_loc = np.zeros(len(c))
      ans_loc[int(row[2]):int(row[3])] = 1
      ans_locs.append(ans_loc)

  return contexts, ans_locs, questions, word2index


def embedding_matrix(i2w, word2vec, embed_size, n_special_tokens):
  vocab_size = len(i2w)
  E = np.zeros((vocab_size,embed_size))
  if word2vec == []:
    return E
  for i in range(vocab_size):
    if i >= n_special_tokens and i2w[i] in word2vec.vocab:
      E[i:i+1,:] = word2vec[i2w[i]][:embed_size].T
  return E
