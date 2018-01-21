const tf = require('tensorflow');

let graph = tf.graph('./graph.proto');
let session = graph.createSession();

session.run(null, null, 'init');

let a = tf.tensor([[2,2],[4,4]], tf.Types.int32);
let b = tf.tensor([[3],[5]], tf.Types.int32);

let outputs = session.run({ var1: a, var2: b }, ['var3', 'computation/result']);
console.log(outputs.var3.value)
console.log(outputs['computation/result'].value);

graph.delete();
