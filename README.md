# TensorFlow + Node.js

[TensorFlow](https://tensorflow.org) is Google's machine learning runtime. It
is implemented as C++ runtime, along with Python framework to support building
a variety of models, especially neural networks for deep learning.

It is interesting to be able to use TensorFlow in a node.js application
using just JavaScript (or TypeScript if that's your preference). However,
the Python functionality is vast (several ops, estimator implementations etc.)
and continually expanding. Instead, it would be more practical to consider
building Graphs and training models in Python, and then consuming those
for runtime use-cases (like prediction or inference) in a pure node.js and
Python-free deployment. This is what this node module enables.

This module takes care of the building blocks and mechanics for working
with the TensorFlow C API, and instead provides an API around Tensors, Graphs,
Sessions and Models.

This is still in the works, and recently revamped to support TensorFlow 1.4+.

## High Level Interface - Models

This is in plan. The idea here is to point to a saved model and be able to
use it for predictions. Instances-in, inferences-out.

Stay tuned for a future update.

## Low Level Interface - Tensors, Graphs and Sessions

### Trivial Example - Loading and Running Graphs
Lets assume we have a simple TensorFlow graph. For illustration purposes, a
trivial graph produced from this Python code, and saved as a GraphDef
protocol buffer file.

```python
import tensorflow as tf

with tf.Graph().as_default() as graph:
  c1 = tf.constant(1, name='c1')
  c2 = tf.constant(41, name='c2')
  result = tf.add(c1, c2, name='result')

  tf.train.write_graph(graph, '.', 'trivial.graph.proto', as_text=False)
```

Now, in node.js, you can load this serialized graph definition, load a
TensorFlow session, and then run specific operations to retrive tensors.

```javascript
const tf = require('tensorflow');

// Load the Graph and create a Session to be able to run the operations
// defined in the graph.
let graph = tf.graph('trivial.graph.proto');
let session = graph.createSession();

// Run to evaluate and retrieve the value of the 'result' op.
let result = session.run(/* inputs */ null,
                         /* outputs */ 'result',
                         /* targets */ null);

// The result is a Tensor, which contains value, type and shape fields.
// This Should print out '42'
console.log(result.value);
    
// Cleanup
graph.delete();
```

### Feeding and Fetching Tensors with a Session

This example goes a bit further - in particular, the Graph contains
variables, and placeholders, requiring initialization as well as feeding values,
when executing the graph. Additionally the Tensors are integer matrices.

```python
import tensorflow as tf

with tf.Graph().as_default() as graph:
  var1 = tf.placeholder(dtype=tf.int32, shape=[2,2], name='var1')
  var2 = tf.placeholder(dtype=tf.int32, shape=[2,1], name='var2')
  var3 = tf.Variable(initial_value=[[1],[1]], dtype=tf.int32)

  tf.variables_initializer(tf.global_variables(), name='init')

  with tf.name_scope('computation'):
    tf.add(tf.matmul(var1, var2), var3, name='result')

  tf.train.write_graph(graph, '.', 'graph.proto', as_text=False)
```

Here is the corresponding node.js snippet to work with the Graph defined above:

```javascript
const tf = require('tensorflow');

let graph = tf.graph('graph.proto');
let session = graph.createSession();

// Run the 'init' op to initialize variables defined in the graph.
session.run(null, null, 'init');

// Generally you can use arrays directly. This samples demonstrates creating
// Tensors to explicitly specify types to match the int32 types that the graph
// expects.
let a = tf.tensor([[2,2],[4,4]], tf.Types.int32);
let b = tf.tensor([[3],[5]], tf.Types.int32);

// You can fetch multiple outputs as well.
let outputs = session.run({ var1: a, var2: b }, ['var3', 'computation/result']);
console.log(outputs.var3.value)
console.log(outputs['computation/result'].value);
    
graph.delete();
```

## Installation

Installation is pretty straight-forward. Installing this module automatically brings installs
the TensorFlow binary dependencies (by default, TensorFlow CPU v1.4.1).

    npm install tensorflow

Optionally, you can specify the build of TensorFlow binaries to install using environment
variables.

    export TENSORFLOW_LIB_TYPE=gpu
    export TENSORFLOW_LIB_VERSION=1.5.0
    npm install tensorflow

The TensorFlow binaries automatically installed within the directory containing the node
module. If you have a custom build of TensorFlow you would like to use instead, you can
suppress downloadinging the binaries at installation time.

    export TENSORFLOW_LIB_PATH=path-to-custom-binaries
    npm install tensorflow

Note that the path you specify must be a directory that contains both `libtensorflow.so` and
`libtensorflow_framework.so`.


## TensorFlow Setup and Docs
Note that to use the Python interface to build TensorFlow graphs and train models, you will
also need to install TensorFlow directly within your Python environment.

    pip install tensorflow==1.4.1

For more information, check out the TensorFlow [install](https://www.tensorflow.org/install)
and [API](https://www.tensorflow.org/api_docs/) documentation.


## In the works, and more to come ...
Some things on the plan to be tackled.

1. Support for string tensors.
2. Support for high-level API (and saved models representing results of training)
3. Support for Windows

Please file issues for feature suggestions, bugs or questions.
