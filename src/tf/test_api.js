// test_api.js
// Runs through TensorFlow APIs.
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

  var graph = fs.readFileSync(path.join(__dirname, 'x.graph'));
  tensorflow.extendGraph(session, graph, graph.length, status);

  if (tensorflow.success(status)) {
    console.log('graph loaded');

    var outputNames = tensorflow.types.StringArray(['result']);
    var outputs = tensorflow.types.TensorArray(1);
    tensorflow.run(session,
                   null, null, 0,
                   outputNames, outputs, outputNames.length,
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


var session = tensorflow.createSession(sessionOptions, status);
if (tensorflow.success(status)) {
  var graph = fs.readFileSync(path.join(__dirname, 'matrix.graph'));
  tensorflow.extendGraph(session, graph, graph.length, status);

  if (tensorflow.success(status)) {
    var m1 = tensorflow.types.FloatArray([1.0,0.0,0.0,1.0]);
    var m2 = tensorflow.types.FloatArray([3.0,3.0,3.0,3.0]);
    var p1 = tensorflow.createTensor(tensorflow.dataTypes.float,
                                     tensorflow.types.LongLongArray([2,2]), 2,
                                     m1.buffer, m1.buffer.length);
    var p2 = tensorflow.createTensor(tensorflow.dataTypes.float,
                                     tensorflow.types.LongLongArray([2,2]), 2,
                                     m2.buffer, m2.buffer.length);
    var inputNames = tensorflow.types.StringArray(['p1','p2']);
    var inputs = tensorflow.types.TensorArray([p1, p2]);

    var outputNames = tensorflow.types.StringArray(['result']);
    var outputs = tensorflow.types.TensorArray(1);
    tensorflow.run(session,
                   inputNames, inputs, inputNames.length,
                   outputNames, outputs, outputNames.length,
                   null, 0,
                   status);
    if (tensorflow.success(status)) {
      console.log('successfully executed the matrix graph with matrices');

      var result = outputs[0];
      console.log('result type: ' + tensorflow.tensorType(result));
      console.log('result shape: ' + tensorflow.tensorDimensions(result));
      console.log('result dims: ' + tensorflow.tensorDimensionLength(result, 0) +
                  ', ' + tensorflow.tensorDimensionLength(result, 1));
      console.log('data length: ' + tensorflow.tensorDataLength(result));

      var resultData = tensorflow.tensorRead(result);
      console.log(resultData.readFloatLE(0));
      console.log(resultData.readFloatLE(4));
      console.log(resultData.readFloatLE(8));
      console.log(resultData.readFloatLE(12));
    }
    else {
      console.log('error executing the matrix graph');
      console.log(tensorflow.statusMessage(status));
    }
  }

  tensorflow.closeSession(session, status);
  tensorflow.deleteSession(session, status);
}

