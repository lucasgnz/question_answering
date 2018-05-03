import tensorflow as tf
from copynet import *
import gensim
import numpy as np

from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper
from preproc import read_data, embedding_matrix


#Parameters
embed_size = 300
hidden_units = 50
attention_units = 30
vocab_size=9
gen_vocab_size = vocab_size


# Load and preprocess data
print('Loading and preprocessing...')
"""
i2w = np.load('../../data/voc.npy')
vocab_size = len(i2w)
gen_vocab_size = vocab_size
(contexts, ans, questions, w2i) = read_data("../../data/tsv/squad_tl.tsv", i2w)
"""
"""
print("Loading Google's pre-trained Word2Vec model")
word2vec = gensim.models.KeyedVectors.load_word2vec_format('https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz', binary=True)

print('Computing embedding matrix')
embedding_m = embedding_matrix(i2w, word2vec)
"""

paragraphs = tf.placeholder(shape=(None, None), dtype=tf.float32, name='paragraphs')
input_ids = tf.cast(paragraphs,tf.int32)
ans_locs = tf.placeholder(shape=(None, None), dtype=tf.float32, name='ans_locs')
encoder_inputs_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_lengths')
targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='targets')
targets_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='targets_lengths')

batch_size, max_time = tf.unstack(tf.shape(paragraphs))

embeddings = tf.Variable(tf.random_uniform([vocab_size, embed_size], -0.01, 0.01), dtype=tf.float32)

paragraphs_embedded = tf.nn.embedding_lookup(embeddings, tf.transpose(tf.cast(paragraphs, tf.int32), [1,0]))
targets_embedded = tf.nn.embedding_lookup(embeddings, tf.transpose(targets, [1,0]))

encoder_inputs = tf.concat([paragraphs_embedded, tf.expand_dims(tf.cast(tf.transpose(ans_locs, [1,0]), tf.float32), axis=2)],axis=2)

encoder_cell_fw = LSTMCell(hidden_units)
encoder_cell_bw = LSTMCell(hidden_units)


((encoder_fw_outputs,encoder_bw_outputs),(encoder_fw_final_state,encoder_bw_final_state)) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw,
	                                    cell_bw=encoder_cell_bw,
	                                    inputs=encoder_inputs,
	                                    sequence_length=encoder_inputs_lengths,
	                                    dtype=tf.float32, time_major=True)
            )

encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

encoder_final_state = LSTMStateTuple(
    c=encoder_final_state_c,
    h=encoder_final_state_h
)


#Shape: (batch_size, time_step, hidden_units)
encoder_outputs = tf.transpose(tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2), [1,0,2])

decoder_cell = LSTMCell(hidden_units*2)

attention_mechanism = BahdanauAttention(attention_units, encoder_outputs)
attention_cell = AttentionWrapper(decoder_cell, attention_mechanism)

copynet_cell = CopyNetWrapper(attention_cell, encoder_outputs, input_ids, vocab_size, gen_vocab_size)

decoder_initial_state = copynet_cell.zero_state(batch_size, tf.float32).clone(cell_state=attention_cell.zero_state(batch_size=batch_size, dtype=tf.float32))

helper = tf.contrib.seq2seq.TrainingHelper(targets_embedded, targets_lengths, time_major=True)
#helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, tf.ones([batch_size], dtype=tf.int32), 0)

decoder = tf.contrib.seq2seq.BasicDecoder(copynet_cell, helper, decoder_initial_state, output_layer=None)
decoder_outputs, final_state, coder_seq_length = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)
decoder_logits, decoder_ids = decoder_outputs

#labels = tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32)

########

contexts=[[6,8,3,4,5,0],[6,8,3,4,5,0]]
ans=[[0,0,1,1,0,0],[0,0,1,1,0,0]]
questions=[[1,5,3,0],[1,5,3,0]]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run([decoder_logits], feed_dict={
  paragraphs:contexts,
  ans_locs:ans,
  encoder_inputs_lengths:[6,6],
  targets: questions,
  targets_lengths:[4,4]
}))
