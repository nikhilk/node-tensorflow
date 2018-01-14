const tf = require('tensorflow');

let session = tf.Session.fromGraphDef('./graph.proto',
  {
    init: 'init',
    var1: 'var1',
    var2: 'var2',
    result: 'computation/result'
  });

session.run(null, null, 'init');

let a = tf.Tensor.create([[2,2],[4,4]], tf.Types.int32);
let b = tf.Tensor.create([[3],[5]], tf.Types.int32);
let result = session.run({ var1: a, var2: b }, 'result');
console.log(result.toValue());

a.delete();
b.delete();
result.delete();

session.delete();
