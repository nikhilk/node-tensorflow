# graph.py
# Builds a trivial graph for most basic example of loading/running TensorFlow.
#
# Run with the following command:
# python graph.py
#
# This should produce graph.proto (which is used from node.js) along with graph.proto.txt and
# graph.proto.json for readable versions.

import google.protobuf.json_format as json
import tensorflow as tf

def save_graph(graph, name='graph'):
  tf.train.write_graph(graph, '.', name + '.proto', as_text=False)
  tf.train.write_graph(graph, '.', name + '.proto.txt', as_text=True)

  data = json.MessageToJson(graph.as_graph_def())
  with open(name + '.proto.json', 'w') as f:
    f.write(data)


def build_graph():
  with tf.Graph().as_default() as graph:
    var1 = tf.placeholder(dtype=tf.int32, shape=[2,2], name='var1')
    var2 = tf.placeholder(dtype=tf.int32, shape=[2,1], name='var2')
    var3 = tf.Variable(initial_value=[[1],[1]], dtype=tf.int32, name='var3')

    tf.variables_initializer(tf.global_variables(), name='init')

    with tf.name_scope('computation'):
      tf.add(tf.matmul(var1, var2), var3, name='result')

    return graph

save_graph(build_graph())
