// main.js
// Runs through TensorFlow functionality.
//

'use strict';

var fs = require('fs'),
    path = require('path'),
    tensorflow = require('./libtensorflow.js');

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

  var graph = fs.readFileSync(path.join(__dirname, 'hello.graph'));
  tensorflow.extendGraph(session, graph, graph.length, status);

  if (tensorflow.success(status)) {
    console.log('graph loaded');

    var outputNames = tensorflow.types.StringArray(['result']);
    var outputs = tensorflow.types.TensorArray(1);
    tensorflow.run(session,
                   // inputNames, inputs, inputNames.length,
                   null, null, 0,
                   outputNames, outputs, outputNames.length,
                   // targets, targets.length,
                   null, 0,
                   status);
    if (tensorflow.success(status)) {
      console.log('successfully executed the graph');

      // Expecting int32 scalar
      var result = outputs[0];
      var data = tensorflow.tensorRead(result).readInt32LE(0);

      console.log('result was ' + data + ' - ' + tensorflow.tensorType(result));
    }
    else {
      console.log('error executing the graph');
      console.log(tensorflow.statusMessage(status));
    }
  }

  tensorflow.closeSession(session, status);
  console.log('session closed');

  tensorflow.deleteSession(session, status);
  console.log('session deleted');
}
else {
  console.log('Failed to create session');
}

