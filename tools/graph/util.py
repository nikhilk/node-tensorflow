# util.py
# Shared utility and helper methods

import google.protobuf.json_format as json
import tensorflow as tf

def save_graph(graph, name):
  tf.train.write_graph(graph, '../../test', name + '.proto', as_text=False)
  tf.train.write_graph(graph, '../../test', name + '.proto.txt', as_text=True)

  data = json.MessageToJson(graph.as_graph_def())
  with open('../../' + name + '.proto.json', 'w') as f:
    f.write(data)
