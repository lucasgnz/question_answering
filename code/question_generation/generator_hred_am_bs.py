import numpy as np
import tensorflow as tf
np.set_printoptions(threshold=np.nan)
from tensorflow.python import debug as tf_debug
import os

from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple


import time
import helpers


from parameters import *

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

class Generator:
    def __init__(self,n_session, pretrainedEmbeddings=[]):
	tf.reset_default_graph()
        self.n_sess = n_session
	self.sess = tf.Session()#config=CONFIG_TF)
	self.attention=tf.Variable(0)
	self.to_print = tf.Variable(0)
	self.learning_rate = tf.placeholder(tf.float32)

        vocab_size = config['VOCAB_SIZE']
        input_embedding_size = config['INPUT_EMBEDDING_SIZE']

        encoder_hidden_units = config['ENCODER_HIDDEN_UNITS']
        decoder_hidden_units = config['DECODER_HIDDEN_UNITS']
        context_hidden_units = config['CONTEXT_HIDDEN_UNITS']

	self.teacher_forcing = tf.placeholder(tf.bool)
	teacher_forcing = self.teacher_forcing		
	
	#Shape: (*, 1)
        self.conversations_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='conversations_lengths')
        conversations_lengths = self.conversations_lengths

	#Shape: (batch_size, n_conv, encoder_max_time)
        self.encoder_inputs = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='encoder_inputs')
        encoder_inputs = self.encoder_inputs
	
	#Shape: (*, 1)
        self.encoder_inputs_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_lengths')
        encoder_inputs_lengths = self.encoder_inputs_lengths
	
	#Shape: (decoder_length, batch_size)
        self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
        decoder_targets = self.decoder_targets

	self.ans_locs = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='ans_locs')
	ans_locs = self.ans_locs

        #Load pretrained embeddings if any
        if pretrainedEmbeddings != []:
	        embeddings = tf.Variable(pretrainedEmbeddings, dtype=tf.float32)
        else:
		embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -0.01, 0.01), dtype=tf.float32)

        
        batch_size, n_conv, encoder_max_time = tf.unstack(tf.shape(encoder_inputs))

        encoder_inputs_ = tf.reshape(encoder_inputs, (batch_size * n_conv, encoder_max_time))
	ans_locs_ = tf.transpose(tf.reshape(ans_locs, (batch_size * n_conv, encoder_max_time)), [1,0])
        encoder_inputs_ = tf.transpose(encoder_inputs_, [1,0])
        encoder_inputs_embedded_ = tf.nn.embedding_lookup(embeddings, encoder_inputs_)
	encoder_inputs_embedded = tf.concat([encoder_inputs_embedded_, tf.expand_dims(tf.cast(ans_locs_, tf.float32),axis=2)], axis=2)

        self.encoder_inputs_embedded = encoder_inputs_embedded


        
        encoder_cell_fw = LSTMCell(encoder_hidden_units)
        encoder_cell_bw = LSTMCell(encoder_hidden_units)

	encoder_cell_fw = tf.nn.rnn_cell.DropoutWrapper( encoder_cell_fw, output_keep_prob=config['DROPOUT_KEEP_PROB'])
	encoder_cell_bw = tf.nn.rnn_cell.DropoutWrapper( encoder_cell_bw, output_keep_prob=config['DROPOUT_KEEP_PROB'])

        ((encoder_fw_outputs,
          encoder_bw_outputs),
         (encoder_fw_final_state,
          encoder_bw_final_state)) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw,
	                                    cell_bw=encoder_cell_bw,
	                                    inputs=encoder_inputs_embedded,
	                                    sequence_length=encoder_inputs_lengths,
	                                    dtype=tf.float32, time_major=True)
            )

        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
	
	#encoder_outputs=tf.Print(encoder_outputs,[encoder_outputs,tf.shape(encoder_outputs)], summarize=30, message="Encoder outputs:")

        encoder_final_state_c = tf.concat(
            (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

        encoder_final_state_h = tf.concat(
            (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

        encoder_final_state = LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h
        )


        def extract_axis_1(data, ind):

            """Get specified elements along the first axis of tensor.
            :param data: Tensorflow tensor that will be subsetted.
            :param ind: Indices to take (one for each element along axis 0 of data).
            :return: Subsetted tensor."""


            batch_range = tf.range(tf.shape(data)[0])
            indices = tf.stack([batch_range, ind], axis=1)
            res = tf.gather_nd(data, indices)

            return res
	

	encoder_outputs = tf.transpose(encoder_outputs, [1,0,2])
	self.encoder_outputs = encoder_outputs
        #encoder_outputs has shape (batch_size * n_conv, encoder_max_time, encoder_hidden_units * 2)
        context_inputs_ = extract_axis_1(encoder_outputs, tf.maximum(0,encoder_inputs_lengths-1))
	


        #context_inputs_ has shape (batch_size * n_conv, encoder_hidden_units * 2)
        context_inputs = tf.reshape(context_inputs_, (n_conv, batch_size, encoder_hidden_units*2))	
	
	
        with tf.variable_scope("context"):
	        context_cell = LSTMCell(context_hidden_units)
		context_cell = tf.nn.rnn_cell.DropoutWrapper(context_cell, output_keep_prob=config['DROPOUT_KEEP_PROB'])
	        context_outputs, context_final_state = tf.nn.dynamic_rnn(context_cell, context_inputs, conversations_lengths, dtype=tf.float32, time_major=True)

        self.decoder_cell = LSTMCell(decoder_hidden_units)
	self.decoder_cell = tf.nn.rnn_cell.DropoutWrapper(self.decoder_cell, output_keep_prob=config['DROPOUT_KEEP_PROB'])
        decoder_lengths = config['DECODER_LENGTH']

        W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -0.01, 0.01), dtype=tf.float32)
        b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)
        self.W, self.b = W, b

        eos_time_slice = tf.fill([batch_size], config['EOS'], name='EOS')
        bos_time_slice = tf.fill([batch_size], config['BOS'], name='BOS')
        pad_time_slice = tf.fill([batch_size], config['PAD'], name='PAD')

        self.embeddings = embeddings
        eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
        bos_step_embedded = tf.nn.embedding_lookup(embeddings, bos_time_slice)
        pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

	W_attention = tf.Variable(tf.random_uniform([decoder_hidden_units,decoder_hidden_units], -0.01, 0.01), dtype=tf.float32)
	U_attention = tf.Variable(tf.random_uniform([decoder_hidden_units,encoder_hidden_units * 2], -0.01, 0.01), dtype=tf.float32)
	b_attention = tf.Variable(tf.random_uniform([decoder_hidden_units], -0.01, 0.01), dtype=tf.float32)
	self.W_attention, self.U_attention, self.b_attention = W_attention, U_attention, b_attention


        self.context_final_state = context_final_state
	if config['ATTENTION_MECHANISM']:
		decoder_initial_input = tf.concat([bos_step_embedded, tf.zeros([batch_size, encoder_hidden_units*2])], 1)
	else:
		decoder_initial_input = bos_step_embedded
        def loop_fn_initial():
		initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
		initial_input = decoder_initial_input
		initial_cell_state = context_final_state
		initial_cell_output = None
		initial_loop_state = None  # we don't need to pass any additional information
		return (initial_elements_finished,
		    initial_input,
		    initial_cell_state,
		    initial_cell_output,
		    initial_loop_state)
	
        def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
		def get_next_input():
			output_logits = tf.add(tf.matmul(previous_output, W), b)
			prediction = tf.argmax(output_logits, axis=1)

			next_input_ = tf.cond(teacher_forcing, lambda: tf.gather(decoder_targets, time-1), lambda: tf.cast(prediction, tf.int32))
			next_input_embedded = tf.nn.embedding_lookup(embeddings, next_input_)

			#next_input_embedded = tf.Print(next_input_embedded,[prediction,tf.gather(decoder_targets, time-1),next_input_,next_input_embedded], summarize=10)

			if config['ATTENTION_MECHANISM']:
				#encoder_outputs has shape (batch_size * n_conv, encoder_max_time, encoder_hidden_units * 2)
				#encoder_outputs_am has shape (encoder_hidden_units * 2, encoder_max_time * n_conv * batch_size)
				#encoder_inputs shape: (batch_size, n_conv, encoder_max_time)
				encoder_outputs_am = tf.reshape(tf.transpose(encoder_outputs, [2,1,0]),[encoder_hidden_units * 2,encoder_max_time, batch_size, n_conv])
				encoder_outputs_am = tf.transpose(encoder_outputs_am, [0,3,1,2])
				encoder_outputs_am = tf.reshape(encoder_outputs_am, [encoder_hidden_units * 2, n_conv * encoder_max_time * batch_size])
				C = tf.expand_dims(tf.matmul(W_attention,tf.transpose(previous_output, [1,0])),1)
				D = tf.reshape(tf.matmul(U_attention, encoder_outputs_am),[decoder_hidden_units, n_conv * encoder_max_time, batch_size])
				E = tf.tanh(
								tf.add(
									C,
									D,
								)
							)
				beta_attention = tf.reshape(tf.matmul(tf.expand_dims(b_attention,0),
							tf.reshape(E,[decoder_hidden_units, -1]))
						,[n_conv*encoder_max_time, batch_size])
				

				alpha_attention = tf.nn.softmax(beta_attention, 0)

				attention_input = tf.matmul(
						tf.expand_dims(tf.transpose(alpha_attention,[1,0]),1),
						tf.transpose(tf.reshape(encoder_outputs_am,[encoder_hidden_units * 2, n_conv * encoder_max_time, batch_size]),[2,1,0])
						)
				
				next_input = tf.concat([next_input_embedded, tf.reshape(attention_input,[batch_size, encoder_hidden_units * 2])], 1)

			else:
				next_input = next_input_embedded

			return next_input

		elements_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size]
					                  # defining if corresponding sequence has ended

		finished = tf.reduce_all(elements_finished) # -> boolean scalar
		if config['ATTENTION_MECHANISM']:
			input = tf.cond(finished, lambda: tf.concat([pad_step_embedded, tf.zeros([batch_size, encoder_hidden_units * 2])], 1), get_next_input)
		else:
			input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
		state = previous_state
		output = previous_output
		loop_state = None

		return (elements_finished, 
		    input,
		    state,
		    output,
		    loop_state)

        def loop_fn(time, previous_output, previous_state, previous_loop_state):
            if previous_state is None:    # time == 0
	        assert previous_output is None and previous_state is None
	        return loop_fn_initial()
            else:
	        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

	decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(self.decoder_cell, loop_fn)
        decoder_outputs = decoder_outputs_ta.stack()

        decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
        decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
        decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
        decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))

        labels = tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32)
  
        decoder_prediction = tf.argmax(decoder_logits, 2)
        self.decoder_prediction = decoder_prediction

	
        

	#TO VISUALIZE ATTENTION, comme avant mais en rajoutant une dimension DECODER_LENGTH
	"""
	if config['ATTENTION_MECHANISM']:
		#encoder_outputs has shape (batch_size * n_conv, encoder_max_time, encoder_hidden_units * 2)
		#encoder_outputs_am has shape (encoder_hidden_units * 2, encoder_max_time * n_conv * batch_size)
		#encoder_inputs shape: (batch_size, n_conv, encoder_max_time)
		encoder_outputs_am_ = tf.reshape(tf.transpose(encoder_outputs, [2,1,0]),[encoder_hidden_units * 2,encoder_max_time, batch_size, n_conv])
		encoder_outputs_am_ = tf.transpose(encoder_outputs_am_, [0,3,1,2])
		encoder_outputs_am_ = tf.reshape(encoder_outputs_am_, [encoder_hidden_units * 2, n_conv * encoder_max_time * batch_size])

		C_ = tf.expand_dims(tf.reshape(tf.matmul(W_attention,tf.reshape(tf.transpose(decoder_outputs, [2,1,0]),[decoder_hidden_units,-1])),[decoder_hidden_units, batch_size, config['DECODER_LENGTH']]),1)

		D_ = tf.reshape(tf.matmul(U_attention, encoder_outputs_am_),[decoder_hidden_units, n_conv * encoder_max_time, batch_size])
		D_ = tf.expand_dims(D_, 3)
		E_ = tf.tanh(
						tf.add(
							C_,
							D_,
						)
					)
		beta_attention_ = tf.reshape(tf.matmul(tf.expand_dims(b_attention,0),
					tf.reshape(E_,[decoder_hidden_units, -1]))
				,[n_conv*encoder_max_time, batch_size, config['DECODER_LENGTH']])


		#encoder_inputs shape: (batch_size, n_conv, encoder_max_time)
		#encoder_inputs_am_ = tf.reshape(tf.transpose(encoder_inputs, [2,1,0]), [encoder_max_time*n_conv, batch_size])
		#mask_special_tokens_ = tf.cast(tf.greater(encoder_inputs_am_, 2), tf.float32)
		#beta_attention2_ = tf.multiply(tf.expand_dims(mask_special_tokens_,-1), beta_attention_)
		alpha_attention_ = tf.nn.softmax(beta_attention_, 0)
		
		self.attention = tf.transpose(alpha_attention_,[1,2,0])
	"""
	
	if config['LOSS_TYPE'] == 1:
		
		decoder_pred = tf.reshape(decoder_prediction,[-1])
		decoder_targ = tf.reshape(decoder_targets,[-1])
		
		decoder_logits_f = tf.reshape(decoder_logits, (-1, vocab_size))
		labels_f = tf.reshape(labels, (-1, vocab_size))
		
		eos = tf.constant(config['EOS'], dtype=tf.int32)
		bos = tf.constant(config['BOS'], dtype=tf.int32)

		where_eos_pred = tf.cast(tf.equal(tf.cast(decoder_prediction, dtype=tf.int32), eos), tf.float32)
		where_eos_targ = tf.cast(tf.equal(tf.cast(decoder_targets, dtype=tf.int32), eos), tf.float32)

		index_eos_pred = tf.argmax(where_eos_pred, axis=0)
		index_eos_targ = tf.argmax(where_eos_targ, axis=0)

		index_eos = tf.maximum(index_eos_pred, index_eos_targ)

		mask =  tf.cast(tf.reshape(tf.less_equal(tf.expand_dims(tf.range(decoder_max_steps, dtype=tf.int32),1) - tf.cast(index_eos,dtype=tf.int32), 0),[-1]), tf.float32)

		decoder_logits_f_m = tf.multiply(tf.expand_dims(mask,-1), decoder_logits_f)
		labels_f_m = tf.multiply(tf.expand_dims(mask,-1), labels_f)
	
		decoder_logits_ = tf.reshape(decoder_logits_f_m, (decoder_max_steps, decoder_batch_size, vocab_size))

		labels_ = tf.reshape(labels_f_m, (decoder_max_steps, decoder_batch_size, vocab_size))
		
		stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
		    labels=labels_,
		    logits=decoder_logits_
		)

		mask = tf.reshape(mask, (decoder_max_steps, decoder_batch_size))
		n_tokens = tf.reduce_sum(mask, axis=0)
		
		self.loss = tf.reduce_sum(stepwise_cross_entropy,axis=0) 
		self.loss = tf.reduce_sum(self.loss / n_tokens) / tf.cast(batch_size, tf.float32)

	elif config['LOSS_TYPE'] == 2:
		
		decoder_pred = tf.reshape(decoder_prediction,[-1])
		decoder_targ = tf.reshape(decoder_targets,[-1])
		
		decoder_logits_f = tf.reshape(decoder_logits, (-1, vocab_size))
		labels_f = tf.reshape(labels, (-1, vocab_size))
		
		pad = tf.constant(config['PAD'], dtype=tf.int32)
		where_pred = tf.cast(tf.not_equal(tf.cast(decoder_pred, dtype=tf.int32), pad), tf.float32)
		where_targ = tf.cast(tf.not_equal(tf.cast(decoder_targ, dtype=tf.int32), pad), tf.float32)
		mask = tf.cast(tf.not_equal(where_pred + where_targ, 0.0), tf.float32)

		decoder_logits_f_m = tf.multiply(tf.expand_dims(mask,-1), decoder_logits_f)
		labels_f_m = tf.multiply(tf.expand_dims(mask,-1), labels_f)
	
		decoder_logits_ = tf.reshape(decoder_logits_f_m, (decoder_max_steps, decoder_batch_size, vocab_size))
		
		labels_ = tf.reshape(labels_f_m, (decoder_max_steps, decoder_batch_size, vocab_size))
		
		stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
		    labels=labels_,
		    logits=decoder_logits_
		)

		mask = tf.reshape(mask, (decoder_max_steps, decoder_batch_size))
		n_tokens = tf.reduce_sum(mask, axis=0)
		
		self.loss = tf.reduce_sum(stepwise_cross_entropy,axis=0) 
		self.loss = tf.reduce_sum(self.loss / n_tokens) / tf.cast(batch_size, tf.float32)

	elif config['LOSS_TYPE'] == 3:
		stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
		    labels=labels,
		    logits=decoder_logits
		)
		self.loss = tf.reduce_sum(stepwise_cross_entropy) / tf.cast(batch_size * config['DECODER_LENGTH'], tf.float32)
	elif config['LOSS_TYPE'] == 4:
		
		decoder_pred = tf.reshape(decoder_prediction,[-1])
		decoder_targ = tf.reshape(decoder_targets,[-1])
		
		decoder_logits_f = tf.reshape(decoder_logits, (-1, vocab_size))
		labels_f = tf.reshape(labels, (-1, vocab_size))
		
		eos = tf.constant(config['EOS'], dtype=tf.int32)
		bos = tf.constant(config['BOS'], dtype=tf.int32)

		where_eos_pred = tf.cast(tf.equal(tf.cast(decoder_prediction, dtype=tf.int32), eos), tf.float32)
		where_eos_targ = tf.cast(tf.equal(tf.cast(decoder_targets, dtype=tf.int32), eos), tf.float32)

		index_eos_pred = tf.argmax(where_eos_pred, axis=0)
		index_eos_targ = tf.argmax(where_eos_targ, axis=0)
		index_eos = tf.maximum(index_eos_pred, index_eos_targ)

		mask =  tf.cast(tf.reshape(tf.less_equal(tf.expand_dims(tf.range(decoder_max_steps, dtype=tf.int32),1) - tf.cast(index_eos,dtype=tf.int32), 0),[-1]), tf.float32)



		mask_pred =  tf.cast(tf.reshape(tf.less_equal(tf.expand_dims(tf.range(decoder_max_steps, dtype=tf.int32),1) - tf.cast(index_eos_pred,dtype=tf.int32), 0),[-1]), tf.float32)



		pad_pred = tf.cast(tf.equal(tf.range(vocab_size, dtype=tf.int32), 0), tf.float32)
		pad_pred = tf.tile(pad_pred, [batch_size* config['DECODER_LENGTH']])
		pad_pred = tf.reshape(pad_pred,(-1, vocab_size))

		pad_pred = tf.multiply(tf.expand_dims(1.0-mask_pred,-1), pad_pred, name="410")

		
		decoder_logits_f_m = tf.multiply(tf.expand_dims(mask_pred,-1), decoder_logits_f, name="413")
		decoder_logits_f_m = tf.add(decoder_logits_f_m,pad_pred)
		decoder_logits_f_m = tf.multiply(tf.expand_dims(mask,-1), decoder_logits_f_m, name="415")

		labels_f_m = tf.multiply(tf.expand_dims(mask,-1), labels_f, name="416")

		decoder_logits_ = tf.reshape(decoder_logits_f_m, (decoder_max_steps, decoder_batch_size, vocab_size), name="418")

		labels_ = tf.reshape(labels_f_m, (decoder_max_steps, decoder_batch_size, vocab_size), name="420")
		

		

		stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
		    labels=labels_,
		    logits=decoder_logits_
		)

		mask = tf.reshape(mask, (decoder_max_steps, decoder_batch_size), name="427")
		n_tokens = tf.reduce_sum(mask, axis=0)
		
		self.loss = tf.reduce_sum(stepwise_cross_entropy,axis=0) 
		self.loss = tf.reduce_sum(self.loss / n_tokens) / tf.cast(batch_size, tf.float32)

	self.stepwise_cross_entropy=stepwise_cross_entropy

	regularizer = config['REG_OUT']*tf.nn.l2_loss(W)
	
	if config['ATTENTION_MECHANISM']:
		regularizer += config['REG_AM']*(tf.nn.l2_loss(W_attention)+tf.nn.l2_loss(U_attention))

	self.loss_ = self.loss + regularizer

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_))
        gradients, _ = tf.clip_by_global_norm(gradients, 8.0)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))#.minimize(self.loss) #
        
  
        self.saver = tf.train.Saver(max_to_keep=None)
        if os.path.exists("ckpt/"+str(self.n_sess)) == False:
            os.system("mkdir ckpt/"+str(self.n_sess))

        
	self.sess.run(tf.global_variables_initializer())


                

    def train_step_supervised(self, batch, lr, teacher_forcing):
        _, l, cross_entropy, p,= self.sess.run([self.train_op, self.loss, self.stepwise_cross_entropy, self.to_print], {
			self.encoder_inputs:np.array(batch[0]),
			self.encoder_inputs_lengths:np.transpose(batch[1]),
			self.conversations_lengths:np.transpose(batch[3]),
			self.decoder_targets:np.transpose(batch[2]),
			self.ans_locs:np.array(batch[4]),
			self.teacher_forcing: bool(np.random.random()<teacher_forcing),
			self.learning_rate: lr
	            })
	return l

    def compute_loss(self, batch):
	try:
        	a, p = self.sess.run([self.loss, self.to_print], {
				self.encoder_inputs:np.array(batch[0]),
				self.encoder_inputs_lengths:np.transpose(batch[1]),
				self.conversations_lengths:np.transpose(batch[3]),
				self.decoder_targets:np.transpose(batch[2]),
				self.ans_locs: np.array(batch[4]),
				self.teacher_forcing: False,
				self.learning_rate: 0
			    })
		return a
	except Exception as e :
		print(e)
		return 0

    def save_model(self, e):
        save_path = self.saver.save(self.sess, "ckpt/"+str(self.n_sess)+"/generator_"+str(e)+".ckpt")
	np.save("ckpt/"+str(self.n_sess)+"/last_ckpt.npy",e)
        print("Model saved in file: %s" % save_path)
    
    def load_model(self,e):
        restore_path = "ckpt/"+str(self.n_sess)+"/generator_"+str(e)+".ckpt"
	reader = tf.train.NewCheckpointReader(restore_path)
	saved_shapes = reader.get_variable_to_shape_map()
	var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
			if var.name.split(':')[0] in saved_shapes])
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
    
    def beam_search(self, conv, enc_input_len):
        beam_size = config['BEAM_SIZE']
        decoder_initial_state, encoder_outputs_ = self.sess.run([self.context_final_state, self.encoder_outputs], {
			self.encoder_inputs:[conv],
			self.encoder_inputs_lengths:np.transpose(enc_input_len),
			self.conversations_lengths:[len(conv)],
			self.decoder_targets: np.transpose([ [] ]),
			self.teacher_forcing: False,
			self.learning_rate: 0
		})
	initial = tf.placeholder(tf.bool)
	previous_output = tf.placeholder(shape=(config['DECODER_HIDDEN_UNITS']), dtype=tf.float32)
	inp = tf.placeholder(shape=(1), dtype=tf.int32)
        inp_embedded = tf.nn.embedding_lookup(self.embeddings, inp)
	n_conv = encoder_outputs_.shape[0]
	encoder_max_time = encoder_outputs_.shape[1]
        encoder_outputs = tf.placeholder(shape=(n_conv, encoder_max_time, config['ENCODER_HIDDEN_UNITS'] * 2), dtype=tf.float32)
	if config['ATTENTION_MECHANISM']:
		encoder_outputs_am = tf.reshape(tf.transpose(encoder_outputs, [2,1,0]),[config['ENCODER_HIDDEN_UNITS'] * 2, encoder_max_time, n_conv])
		encoder_outputs_am = tf.transpose(encoder_outputs_am, [0,2,1])
		encoder_outputs_am = tf.reshape(encoder_outputs_am, [config['ENCODER_HIDDEN_UNITS'] * 2, n_conv * encoder_max_time])
		C = tf.matmul(self.W_attention,tf.expand_dims(previous_output,1))
		D = tf.reshape(tf.matmul(self.U_attention, encoder_outputs_am),[config['DECODER_HIDDEN_UNITS'], n_conv * encoder_max_time])
		E = tf.tanh(
						tf.add(
							C,
							D,
						)
					)
		beta_attention = tf.reshape(tf.matmul(tf.expand_dims(self.b_attention,0),
					E)
				,[n_conv*encoder_max_time])


		alpha_attention = tf.nn.softmax(beta_attention, 0)

		attention_input = tf.cond(initial, lambda:tf.zeros([1,config['ENCODER_HIDDEN_UNITS'] * 2]), lambda: tf.matmul(
				tf.expand_dims(alpha_attention,0),
				tf.transpose(encoder_outputs_am,[1,0])
				))
	else:
		alpha_attention = tf.Variable(0)


        state_c = tf.placeholder(shape=(1, config['DECODER_HIDDEN_UNITS']), dtype=tf.float32)
        state_h = tf.placeholder(shape=(1, config['DECODER_HIDDEN_UNITS']), dtype=tf.float32)
	if config['ATTENTION_MECHANISM']:
		total_inp = tf.concat([inp_embedded,attention_input],1)
	else:
		total_inp = inp_embedded
        output, state = self.decoder_cell.__call__(total_inp, (state_c,state_h))
        logits = tf.nn.softmax(tf.add(tf.matmul(output, self.W), self.b), 1)
        pred = tf.argmax(logits, 1)
        i = config['BOS']
        s = decoder_initial_state
        seq = [[i]]
        states = [s]
        logit = [1]
	prev_out = [np.zeros(config['DECODER_HIDDEN_UNITS'])]
	attention = [[]]

        for t in range(config['DECODER_LENGTH']):
                logit_, seq_, states_, prev_out_, attention_ = [], [], [], [], []
                for k in range(len(seq)):
                        if seq[k][-1] == config['EOS']:
                                logit_.append(logit[k])
                                seq_.append(seq[k])
                                states_.append(states[k])
				prev_out_.append(prev_out[k])
				attention_.append(attention[k])
                        else:
				
                                l, st, out, att = self.sess.run([logits,state,output, alpha_attention], {inp:[seq[k][-1]], encoder_outputs:encoder_outputs_, initial: (t==0), previous_output:prev_out[k], state_c: states[k].c, state_h: states[k].h})
                                l = np.array(l[0])
                                preds = l.argsort()[-beam_size:]
                                for p in range(beam_size):
                                        logit_.append(logit[k] * l[preds[p]])
                                        seq_.append(seq[k] + [preds[p]])
					attention_.append(attention[k] + [att])
                                        states_.append(st)
					prev_out_.append(np.reshape(out, config['DECODER_HIDDEN_UNITS']))
                keep =  np.array(logit_).argsort()[-beam_size:][::-1]
                logit = [logit_[idx] for idx in keep.tolist()]
                seq = [seq_[idx] for idx in keep.tolist()]
                states = [states_[idx] for idx in keep.tolist()]
		prev_out = [prev_out_[idx] for idx in keep.tolist()]
		attention = [attention_[idx] for idx in keep.tolist()]
        return(seq, logit, attention)


    def reply(self, conv, enc_input_len):
        bs_result = self.beam_search(conv, enc_input_len)
	#bs_result=([],[])
        
        output, att = self.sess.run([self.decoder_prediction, self.attention], {
			self.encoder_inputs:[conv],
			self.encoder_inputs_lengths:np.transpose(enc_input_len),
			self.conversations_lengths:[len(conv)],
			self.decoder_targets: np.transpose([ [] ]),
			self.teacher_forcing: False,
			self.learning_rate: 0
		})

        return output, bs_result, att


