import numpy as np
import tensorflow as tf
np.set_printoptions(threshold=np.nan)
from tensorflow.python import debug as tf_debug
import os
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper
from copynet import *
import time
from lazy_adam_optimizer import LazyAdamOptimizer

from parameters import *

class Generator:
	def __init__(self,n_session, pretrainedEmbeddings=[]):
		tf.reset_default_graph()
		self.n_sess = n_session
		self.sess = tf.Session()#config=CONFIG_TF)
		self.learning_rate = tf.placeholder(tf.float32)

		hidden_units = config['HIDDEN_UNITS']
		attention_units = config['ATTENTION_UNITS']
		vocab_size = config['VOCAB_SIZE']
		gen_vocab_size = config['GEN_VOCAB_SIZE']
		embed_size = config['EMBED_SIZE']

		self.paragraphs = tf.placeholder(shape=(None, None), dtype=tf.float32, name='paragraphs')
		self.ans_locs = tf.placeholder(shape=(None, None), dtype=tf.float32, name='ans_locs')
		self.encoder_inputs_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_lengths')
		self.targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='targets')
		self.targets_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='targets_lengths')

		paragraphs = self.paragraphs
		ans_locs = self.ans_locs
		encoder_inputs_lengths = self.encoder_inputs_lengths
		targets = self.targets
		targets_lengths = self.targets_lengths

		input_ids = tf.cast(paragraphs, tf.int32)


		batch_size, max_time = tf.unstack(tf.shape(paragraphs))

		# Load pretrained embeddings if any
		if pretrainedEmbeddings != []:
			embeddings = tf.Variable(pretrainedEmbeddings, dtype=tf.float32)
		else:
			embeddings = tf.Variable(tf.random_uniform([vocab_size, embed_size], -0.01, 0.01), dtype=tf.float32)

		paragraphs_embedded = tf.nn.embedding_lookup(embeddings, tf.transpose(tf.cast(paragraphs, tf.int32), [1,0]))

		start_tokens = tf.ones([batch_size], dtype=tf.int32)
		decoder_inputs = tf.concat([tf.expand_dims(start_tokens, 1), targets], 1)
		decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, tf.transpose(decoder_inputs, [1,0]))

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

		helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs_embedded, targets_lengths, time_major=True)
		#helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, tf.ones([batch_size], dtype=tf.int32), 0)

		decoder = tf.contrib.seq2seq.BasicDecoder(copynet_cell, helper, decoder_initial_state, output_layer=None)
		decoder_outputs, final_state, coder_seq_length = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)
		decoder_logits, decoder_ids = decoder_outputs

		#LOSS
		decoder_targets = tf.transpose(targets, [1,0])
		labels = tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32)

		decoder_logits_ = tf.transpose(decoder_logits,[1,0,2])

		stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
			labels=labels,
			logits=decoder_logits_
		)

		"""eos = tf.constant(config['EOS'], dtype=tf.int32)
		where_eos_targ = tf.cast(tf.equal(tf.cast(decoder_targets, dtype=tf.int32), eos), tf.float32)
		n_tokens = tf.cast(tf.argmax(where_eos_targ, axis=0), tf.float32)"""

		targets_max_len, _ = tf.unstack(tf.shape(decoder_targets))

		self.loss = tf.reduce_sum(stepwise_cross_entropy, axis=0) / tf.cast(targets_max_len, tf.float32)
		self.loss = tf.reduce_sum(self.loss) / tf.cast(batch_size, tf.float32)
		#self.loss	= tf.Print(self.loss,[tf.nn.softmax(decoder_logits),labels], summarize=100)

		optimizer = tf.train.AdagradOptimizer(self.learning_rate)#tf.train.GradientDescent
		gradients, variables = zip(*optimizer.compute_gradients(self.loss))
		gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
		self.train_op = optimizer.apply_gradients(zip(gradients, variables))#.minimize(self.loss)


		self.saver = tf.train.Saver(max_to_keep=None)
		if os.path.exists("ckpt/"+str(self.n_sess)) == False:
			os.system("mkdir ckpt/"+str(self.n_sess))


		self.sess.run(tf.global_variables_initializer())


	def train_step_supervised(self, batch, lr):
			_, l = self.sess.run([self.train_op, self.loss], feed_dict={
  			self.paragraphs:batch[0],
				self.ans_locs:batch[1],
				self.encoder_inputs_lengths:batch[2],
				self.targets: batch[3],
				self.targets_lengths:batch[4],
				self.learning_rate:lr
			})
			return l

	def compute_loss(self, batch):
		l = self.sess.run([self.loss], feed_dict={
			self.paragraphs: batch[0],
			self.ans_locs: batch[1],
			self.encoder_inputs_lengths: batch[2],
			self.targets: batch[3],
			self.targets_lengths: batch[4]
		})
		return l[0]


	def save_model(self, e):
		save_path = self.saver.save(self.sess, "ckpt/"+str(self.n_sess)+"/generator_"+str(e)+".ckpt")
		np.save("ckpt/"+str(self.n_sess)+"/last_ckpt.npy",e)
		print("Model saved in file: %s" % save_path)

	def load_model(self,e):
		restore_path = "ckpt/"+str(self.n_sess)+"/generator_"+str(e)+".ckpt"
		reader = tf.train.NewCheckpointReader(restore_path)
		saved_shapes = reader.get_variable_to_shape_map()
		var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables() if var.name.split(':')[0] in saved_shapes])
		restore_vars = []
		name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
		with tf.variable_scope('', reuse=True):
			for var_name, saved_var_name in var_names:
				curr_var = name2var[saved_var_name]
				var_shape = curr_var.get_shape().as_list()
				if var_shape == saved_shapes[saved_var_name]:
					restore_vars.append(curr_var)
		saver = tf.train.Saver(restore_vars)
		saver.restore(self.sess, restore_path)
		print("Model restored from file")