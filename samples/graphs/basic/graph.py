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
    c1 = tf.constant(1, name='c1')
    c2 = tf.constant(41, name='c2')
    result = tf.add(c1, c2, name='result')

    return graph

save_graph(build_graph())
