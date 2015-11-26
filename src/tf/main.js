// main.js
// Runs through TensorFlow functionality.
//

'use strict';

var tensorflow = require('./libtensorflow.js');

var status = tensorflow.createStatus();
tensorflow.updateStatus(status, 0, 'OK');
console.log(tensorflow.statusCode(status));
console.log(tensorflow.statusMessage(status));
console.log(tensorflow.success(status));
tensorflow.deleteStatus(status);

var shape = tensorflow.types.LongLongArray([2]);
var data = tensorflow.types.FloatArray([0.5, 42.0]);

var tensor =
  tensorflow.createTensor(tensorflow.dataTypes.float, shape, shape.length,
                          data.buffer, data.buffer.length);
console.log('dimensions: ' + tensorflow.tensorDimensions(tensor));
console.log('dimension[0]: ' + tensorflow.tensorDimensionLength(tensor, 0));
console.log('data length: ' + tensorflow.tensorDataLength(tensor));
var buffer = tensorflow.tensorRead(tensor);
console.log(buffer.readFloatLE(0));
console.log(buffer.readFloatLE(4));
tensorflow.deleteTensor(tensor);


var sessionOptions = tensorflow.createSessionOptions();
var status = tensorflow.createStatus();
var session = tensorflow.createSession(sessionOptions, status);
if (tensorflow.success(status)) {
  console.log('session created');
  tensorflow.closeSession(session, status);
  tensorflow.deleteSession(session);
}
else {
  console.log('Failed to create session');
}

