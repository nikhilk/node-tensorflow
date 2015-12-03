# TensorFlow + Node.js

[TensorFlow](https://tensorflow.org) is Google's machine learning and
data flow-based computation system. It is implemented as C++ runtime, and comes with Python bindings out-of-the-box, including a comprehensive toolbox of
primitives for use in defining a variety of models, including various kinds of
neural networks.

This module makes it possible to tap into those capabilities from node.js, with
a 100% JavaScript (or TypeScript if you prefer) developer experience. It
provides a natural node.js development experience, while taking care of the
interop.

Essentially it allows you to define graphs, work with tensors and operations,
and then execute that graph to run through your data within a session. These
key concepts define the essence of the programming model.

## A glimpse at what is in the works

This is project is super early, so the current implementation is far from
complete, and has been cobbled together somewhat quickly to share early ideas,
and gather community input and participation.

But, anyway, here is what is in the works:

    var tf = require('tensorflow'),
        fs = require('fs');

    // Define the graph
    var g = new tf.Graph();
    var shape = [2, 2]
    var p1 = g.placeholder(tf.types.float, shape).named('p1');
    var p2 = g.placeholder(tf.types.float, shape).named('p2');
    var value = g.matmul(p1, p2).named('value');

    // Optionally save it out (with corresponding APIs to load, instead
    // of re-building the graph, for example when using the resulting model).
    fs.writeFileSync('/tmp/hello.graph', g.save());

    // Execute the graph
    var session = new tf.Session(g);

    var data = {};
    data[p1] = new tf.Tensor([[1.0, 0.0],[0.0, 1.0]]);
    data[p2] = new tf.Tensor([[3.0, 3.0],[3.0, 3.0]]);

    var results = session.run([ value ], data);
    console.log(results[value]);


As it starts to come together it will be available on
[npmjs](https://www.npmjs.org/package/tensorflow) as any other module.

## More for later, and helping ...

Stay tuned for more. If you're interested in helping out, by all means, 
please connect here. I'll share a bit more detail about the roadmap, as well
as issues for big questions, design issues and areas of exploration.

