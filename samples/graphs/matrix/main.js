const tf = require('tensorflow');

let session = tf.Session.fromGraphDef('./graph.proto', { result: 'computation/result' });

let outputs = session.run(null, ['result'], null);
console.log(outputs.result.toValue());

outputs.result.delete();
session.delete();
