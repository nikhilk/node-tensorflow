const tf = require('tensorflow');

let graph = tf.graph('./graph.proto');
let session = graph.createSession();

let result = session.run({ input: ['example', 'data'] }, 'output');
console.log(result);

graph.delete();
