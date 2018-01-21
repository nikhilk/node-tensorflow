const tf = require('tensorflow');

console.log('Tensor Test');

var tensor = tf.tensor([0.5, 42.0]);
console.log(tensor.shape);
console.log(tensor.value);

let items = 
[
  [
    [1,2], [3,4]
  ],

  [
    [5,6], [7,8]
  ]
]
var tensor2 = tf.tensor(items);
console.log(tensor2.shape);
console.log(tensor2.value);


console.log('Graph Test');

let graph = tf.graph('./graph.proto');
let session = graph.createSession();
let result = session.run(null, 'result');
console.log(result);

graph.delete();
