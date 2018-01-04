const path = require('path');
const tf = require('../src/index');

console.log('Tensor Test');

var tensor = tf.Tensor.create([0.5, 42.0]);
console.log(tensor.shape);
let data = tensor.toValue();
console.log(data);
tensor.delete();

let items = 
[
  [
    [1,2], [3,4]
  ],

  [
    [5,6], [7,8]
  ]
]
var tensor2 = tf.Tensor.create(items);
console.log(tensor2.shape);
let data2 = tensor2.toValue();
console.log(data2);


console.log('Graph Test');

let graph = tf.Graph.fromGraphDef('./trivial.proto')
graph.loadOperations({c1: 'c1', c2: 'c2', result: 'result'});
console.log(graph);
graph.delete();


console.log('Session Test');

let session = tf.Session.fromGraphDef('./trivial.proto', true);
session.graph.loadOperations({ result: 'result' });
let tensors = session.run(null, ['result'], null);
console.log(tensors.result.toValue());
tensors.result.delete();
session.delete();
