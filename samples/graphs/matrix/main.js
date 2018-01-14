const tf = require('tensorflow');

let session = tf.Session.fromGraphDef('./graph.proto',
  {
    init: 'init',
    var1: 'var1',
    var2: 'var2',
    result: 'computation/result'
  });

let inputs = {
  var1: tf.Tensor.create([[2,2],[4,4]], tf.Types.int32),
  var2: tf.Tensor.create([[3],[5]], tf.Types.int32)
};

session.run(null, null, ['init']);
let outputs = session.run(inputs, ['result']);
console.log(outputs.result.toValue());

inputs.var1.delete();
inputs.var2.delete();
outputs.result.delete();
session.delete();
