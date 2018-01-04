# TensorFlow + Node.js

[TensorFlow](https://tensorflow.org) is Google's machine learning runtime.
It is implemented as C++ runtime, along with Python bindings and framework
out-of-the-box to support building a variety of models, especially neural
networks for deep learning.

It is interesting to be able to use TensorFlow in a node.js application
using just JavaScript (or TypeScript if that's your preference). However,
the Python functionality is vast (several ops, estimator implementations etc.)
and continually expanding. Instead, it would be interesting to consider
building Graphs and training models in Python, and then consuming those
for runtime use-cases (like prediction or inference) in a pure node.js and
Python-free deployment. This is what this node module offers.

This module takes care of the building blocks and mechanics for working
with the TensorFlow C API, and instead provides an API around Tensors, Graphs,
Sessions and Models.

This is still in the works, and recently revamped to support TensorFlow 1.4+.

## High Level Interface - Models

This is in plan. The idea here is to point to a saved model and be able to
use it for predictions. Instances-in, inferences-out.


## Low Level Interface - Tensors, Graphs and Sessions

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

// Load the session with the specified graph definition
let session = tf.Session.fromGraphDef('trivial.graph.proto');
    
// Load references to operations that will be referenced later,
// mapping friendly names to graph names (useful for handling
// fully-qualified tensor or op names)
session.graph.loadReferences({ result: 'result' })
    
// Run to evaluate and retrieve the value of the 'result' op.
let outputs = session.run(/* inputs */ null,
                          /* outputs */ [ 'result' ],
                          /* targets */ null);

// Should print out '42'
console.log(outputs.result.toValue());
    
// Low-level interface requires explicit cleanup, so be sure to
// delete native objects.
outputs.result.delete();
session.delete();
```

Above is obviously an overly-simplistic sample. A real-sample (forthcoming)
would demonstrate both inputs and outputs, and multi-dimensional tensors
rather than scalars.

Stay tuned!

As more of this functionality is in place, it will be available on
[npmjs](https://www.npmjs.org/package/tensorflow) as any other module.

## In the works, and more to come ...

Some things on the plan to be tackled.

1. Support for string tensors.
2. Support for high-level API (using saved models representing results
   of training)
3. Nicely packaged and published npm package along with better docs.
4. What else? Please file issues...

If any of this is interesting to you, please let me know!
