import tensorflow as tf

with tf.Graph().as_default() as graph:
  size = [2,2]
  p1 = tf.placeholder(tf.float32, shape=size, name='p1')
  p2 = tf.placeholder(tf.float32, shape=size, name='p2')

  result = tf.add(p1, p2, name = 'result')

