import tensorflow as tf
from copynet import *
import gensim
import numpy as np
import time, math
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper
from preproc import read_data, embedding_matrix
import random
from generator_copynet import Generator
from parameters import *

SMALL = '2'
SHORTEN = False
WORD_EMBEDDINGS = False



# Load and preprocess data
print('Loading and preprocessing...')
i2w = np.load('../../data/voc'+SMALL+'.npy')
config['VOCAB_SIZE'] = min(len(i2w),config['LIM_VOCAB_SIZE'])
print("Total vocab size: ",config['VOCAB_SIZE'])

(contexts_, ans_, questions_) = read_data("../../data/tsv/squad_tl"+SMALL+".tsv")

print(len(contexts_), " examples in dataset before filtering")



contexts, ans, questions = [], [], []

for (c,a,q) in zip(contexts_, ans_, questions_):
  if len(c) <= config['MAX_LEN']:
    contexts.append(c)
    ans.append(a)
    questions.append(q)
  else:
    if SHORTEN:
      contexts.append(c[:config['MAX_LEN']])
      ans.append(a[:config['MAX_LEN']])
      questions.append(q)



c = list(zip(contexts, ans, questions))
random.shuffle(c)
contexts, ans, questions = zip(*c)

LIMIT = 2000
LIMIT = len(contexts)

contexts=contexts[:LIMIT]
questions=questions[:LIMIT]
ans=ans[:LIMIT]

print(len(contexts), " examples in dataset after filtering")
N = int(len(contexts) / 1.3)

contexts_test = contexts[N:]
ans_test = ans[N:]
questions_test = questions[N:]

contexts = contexts[:N]
ans = ans[:N]
questions = questions[:N]


batch_size = config['BATCH_SIZE']
n_epoch = 100

training_iters = int(math.ceil(len(contexts) / batch_size))
testing_iters = int(math.ceil(len(contexts_test) / batch_size))


if WORD_EMBEDDINGS:
  print("Loading Google's pre-trained Word2Vec model...")
  word2vec = gensim.models.KeyedVectors.load_word2vec_format('https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz', binary=True)
  print('Computing embedding matrix...')
  embedding_m = embedding_matrix(i2w, word2vec, config['EMBED_SIZE'], config['N_SPECIAL_TOKENS'])
  G = Generator(SESSION, pretrainedEmbeddings=embedding_m)
else:
  G = Generator(SESSION)

def replace_by_unk(l):
  return [[i if i < config['LIM_VOCAB_SIZE'] else config['UNK'] for i in e] for e in l]

def pad(x, l):
  return [list(e) + [0 for i in range(l - len(e))] for e in x]

def max_pad(x):
  y = map(len,x)
  l = max(y)
  return pad(x, l), y

def one_batch(c,a,q,i,j):
  batch = []
  contexts_b, contexts_len = max_pad(c[i:j])
  questions_b, questions_len = max_pad(q[i:j])
  batch.append(replace_by_unk(contexts_b))
  batch.append(pad( a[i:j], len(contexts_b[0])))
  batch.append(contexts_len)
  batch.append(replace_by_unk(questions_b))
  batch.append(np.array(questions_len))
  return batch


q= [np.random.randint(0, high=config['VOCAB_SIZE']) for i in range(25)]
def max_batch(n):
  batch = []
  batch.append([[i for i in range(config['MAX_LEN'])] for k in range(n)])
  batch.append([[0 for i in range(config['MAX_LEN'])] for k in range(n)])
  batch.append([config['MAX_LEN'] for k in range(n)])
  batch.append([q for k in range(n)])
  batch.append([25 for k in range(n)])
  return batch

t0 = time.time()

def train_supervised(contexts, ans, questions):
    # Load last trained model
    if RESUME_TRAINING and os.path.exists('ckpt/' + str(SESSION) + '/last_ckpt.npy'):
      if RESUME_CKPT == 0:
        ckpt = np.load('ckpt/' + str(SESSION) + '/last_ckpt.npy')
        lr = np.load('ckpt/' + str(SESSION) + '/lr.npy')
      else:
        ckpt = RESUME_CKPT
        lr = RESUME_LR
        if lr == 0:
          lr = np.load('ckpt/' + str(SESSION) + '/lr.npy')
      G.load_model(ckpt)
      L_ = np.load('sessions/' + str(SESSION) + '/loss_test.npy').tolist()[:ckpt]
      L_train = np.load('sessions/' + str(SESSION) + '/loss_train.npy').tolist()[:ckpt]
    else:
      ckpt = 0
      lr = config['INITIAL_LR']
      L_ = []
      L_train = []

    print('Supervised training with ' + str(batch_size) + ' examples / batch...')
    for e in range(1 + ckpt, n_epoch + 1 + ckpt):
      # TRAINING EPOCH
      c = list(zip(contexts, ans, questions))
      random.shuffle(c)
      contexts, ans, questions = zip(*c)
      L_train_ = 0
      print('Epoch ' + str(e))
      for it in range(training_iters):
        batch = one_batch(
          contexts,
          ans,
          questions,
          batch_size * it,
          batch_size * (it + 1)
        )
        #batch = max_batch(batch_size)
        try:
          l = G.train_step_supervised(batch, lr)
          print("T: "+str(int(time.time()-t0))+" | Epoch "+str(e)+" | Batch "+str(it+1)+" / "+str(training_iters)+" | loss: "+str(l)+" | perplexity: "+str(math.exp(l))+" | lr: "+str(lr))
          L_train_ += l / training_iters
        except:
          print("Error on batch, shape of batch: ",np.shape(batch))

      L_train.append(L_train_)
      np.save('sessions/' + str(SESSION) + '/loss_train.npy', L_train)


      #LOSS ON TEST SET
      test_loss = 0
      for it in range(testing_iters):
        batch = one_batch(
          contexts_test,
          ans_test,
          questions_test,
          batch_size * it,
          batch_size * (it + 1)
        )
        try:
          l = G.compute_loss(batch)
          test_loss += l / testing_iters
        except:
          print("Error on batch, shape of batch: ",np.shape(batch))

      print("Loss on test set:", test_loss)
      L_.append(test_loss)
      np.save('sessions/' + str(SESSION) + '/loss_test.npy', L_)

      lr = config['INITIAL_LR'] / (1. + float(e) / float(config['N_EPOCH_HALF_LR']))
      
      G.save_model(e)
      np.save('ckpt/' + str(SESSION) + '/lr.npy', lr)



train_supervised(contexts, ans, questions)
"""while True:
  nn=int(raw_input("n examples?"))
  b = max_batch(nn)
  print(G.train_step_supervised(b, 0.01))"""