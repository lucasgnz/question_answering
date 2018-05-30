import tensorflow as tf

vocab_size = 9

encoder_input_ids = tf.placeholder(tf.int32, shape=(None,None))
copy_score = tf.placeholder(tf.float32, shape=(None,None))

encoder_input_mask = tf.one_hot(encoder_input_ids, vocab_size)
#expanded_copy_score = tf.einsum("ijn,ij->ij", encoder_input_mask, copy_score)
#prob_c = expanded_copy_score
prob_c_one_hot2 = tf.einsum("ijn,ij->in", encoder_input_mask, copy_score)

batch_size, time_steps = tf.unstack(tf.shape(encoder_input_ids))
inputs_flat = tf.reshape(encoder_input_ids, [-1])
copy_score_flat = tf.reshape(copy_score, [-1])
rr = tf.range(tf.cast(batch_size*time_steps, tf.int64), dtype=tf.int64)
indices = tf.stack([rr,tf.cast(inputs_flat, tf.int64)], axis=1)
shape = tf.cast([batch_size*time_steps, vocab_size], tf.int64)
expanded_copy_score_sparse_flat = tf.SparseTensor(indices,copy_score_flat,shape)
expanded_copy_score_sparse = tf.sparse_reshape(expanded_copy_score_sparse_flat, [batch_size,time_steps,vocab_size])
copy_score_sparse = tf.sparse_reduce_sum_sparse(expanded_copy_score_sparse, axis=1)
prob_c_one_hot = tf.sparse_to_dense(copy_score_sparse.indices,copy_score_sparse.dense_shape,copy_score_sparse.values)

with tf.Session() as sess:
  print(sess.run([prob_c_one_hot,prob_c_one_hot2], feed_dict={
    encoder_input_ids:[[5,4,3,2,1],[3,4,1,5,2]],
    copy_score:[[0,0.5,0,0.5,0],[0.3,0.7,0,0,0]]
  }))