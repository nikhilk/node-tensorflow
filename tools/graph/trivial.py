# trivial.py
# Builds a trivial graph for most basic example of loading/running TensorFlow.
#

import tensorflow as tf
from . import util as util

def build_graph():
  with tf.Graph().as_default() as graph:
    c1 = tf.constant(1, name='c1')
    c2 = tf.constant(41, name='c2')
    result = tf.add(c1, c2, name='result')

    return graph

util.save_graph(build_graph(), 'trivial')
