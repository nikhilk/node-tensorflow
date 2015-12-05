import tensorflow as tf

with tf.Graph().as_default() as graph:
  c1 = tf.constant(1)
  c2 = tf.constant(41)

  result = tf.add(c1, c2, name = 'result')

