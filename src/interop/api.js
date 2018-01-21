// tf.js
// Interface for TensorFlow C API
//
// This defines the TensorFlow library matching a subset of the C API methods as defined in
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h
//

'use strict';

const ffi = require('ffi'),
      fs = require('fs'),
      path = require('path'),
      protobuf = require('protocol-buffers'),
      ref = require('ref'),
      refArray = require('ref-array'),
      refStruct = require('ref-struct');

// Interop types to work with the C API.
const types = {
  Void: 'void',
  Int: 'int32',
  IntArray: refArray('int32'),
  LongLong: 'longlong',
  LongLongArray: refArray('longlong'),
  Float: 'float',
  FloatArray: refArray('float'),
  String: 'string',
  StringArray: refArray('string'),
  Size: 'size_t',
  Any: ref.refType('void')
};
types.Tensor = types.Any;
types.TensorArray = refArray(types.Tensor);
types.Status = types.Any;
types.Buffer = types.Any;
types.Graph = types.Any;
types.ImportGraphDefOptions = types.Any;
types.Operation = types.Any;
types.OperationArray = refArray(types.Operation);
types.OperationValue = refStruct({ op: types.Operation, index: 'int32' });
types.OperationValueArray = refArray(types.OperationValue);
types.Session = types.Any;
types.SessionOptions = types.Any;

// Tensor data types supported by TensorFlow.
const tensorTypes = {
  float: 1,
  double: 2,
  int32: 3,
  uint8: 4,
  int16: 5,
  int8: 6,
  string: 7,
  complex64: 8,
  int64: 9,
  bool: 10,
  qint8: 11,
  quint8: 12,
  qint32: 13,
  bfloat16: 14,
  qint16: 15,
  quint16: 16,
  complex128: 18,
  half: 19,
  resource: 20,
  variant: 21,
  uint32: 22,
  uint64: 23
};

// Status codes used by the TensorFlow API.
const statusCodes = {
  ok: 0,
  cancelled: 1,
  unknown: 2,
  invalidArgument: 3,
  deadlineExceeded: 4,
  notFound: 5,
  alreadyExists: 6,
  permissionDenied: 7,
  resourceExhausted: 8,
  failedPrecondition: 9,
  aborted: 10,
  outOfRange: 11,
  unimplemented: 12,
  internal: 13,
  unavailable: 14,
  dataLoss: 15,
  unauthenticated: 16,
};

let libPath = process.env['TENSORFLOW_LIB_PATH'];
if (!libPath) {
  libPath = path.join(__dirname, '..', '..', 'lib');
}
if (!fs.existsSync(path.join(libPath, 'libtensorflow.so'))) {
  throw new Error(`libtensorflow.so was not found at "${libPath}"`);
}
if (!fs.existsSync(path.join(libPath, 'libtensorflow_framework.so'))) {
  throw new Error(`libtensorflow_framework.so was not found at "${libPath}"`);
}

// Change the TensorFlow logging level to WARNING (default is INFO[4], which gets pretty noisy).
process.env['TF_CPP_MIN_LOG_LEVEL'] = process.env['TENSORFLOW_LIB_LOG_LEVEL'] || '3';

// Defines the subset of relevant TensorFlow APIs.
// Each entry corresponds to an exported API signature in form of name -> [return type, arg types].
const libApi = {
  // Status TF_NewStatus()
  TF_NewStatus: [types.Status, []],

  // void TF_DeleteStatus(Status)
  TF_DeleteStatus: [types.Void, [types.Status]],

  // void TF_SetStatus(Status, int code, string message)
  TF_SetStatus: [types.Void, [types.Status, types.Int, types.String]],

  // int TF_GetCode(Status)
  TF_GetCode: [types.Int, [types.Status]],

  // string TF_Message(Status)
  TF_Message: [types.String, [types.Status]],

  // Tensor TF_NewTensor(int dataType, longlong* dimLengths, int dims, void* data, size_t length,
  //                     void* dealloc, void* deallocarg)
  TF_NewTensor: [types.Tensor, [types.Int, types.LongLongArray, types.Int, types.Any, types.Size,
                                types.Any, types.Any]],

  // void TF_DeleteTensor(Tensor)
  TF_DeleteTensor: [types.Void, [types.Tensor]],

  // int TF_TensorType(tensor)
  TF_TensorType: [types.Int, [types.Tensor]],

  // int TF_NumDims(tensor)
  TF_NumDims: [types.Int, [types.Tensor]],

  // longlong TF_Dim(tensor, int dimensionIndex)
  TF_Dim: [types.LongLong, [types.Tensor, types.Int]],

  // size_t TF_TensorByteSize(tensor)
  TF_TensorByteSize: [types.Size, [types.Tensor]],

  // void* TF_TensorData(tensor)
  TF_TensorData: [types.Any, [types.Tensor]],

  // Buffer TF_NewBufferFromString(void* data, size_t len)
  TF_NewBufferFromString: [types.Buffer, [types.Any, types.Size]],

  // void TF_DeleteBuffer(Buffer)
  TF_DeleteBuffer: [types.Void, [types.Buffer]],

  // ImportGraphDefOptions TF_NewImportGraphDefOptions()
  TF_NewImportGraphDefOptions: [types.ImportGraphDefOptions, []],

  // void TF_DeleteImportGraphDefOptions(Graph)
  TF_DeleteImportGraphDefOptions: [types.Void, [types.ImportGraphDefOptions]],

  // Graph TF_NewGraph()
  TF_NewGraph: [types.Graph, []],

  // void TF_DeleteGraph(Graph)
  TF_DeleteGraph: [types.Void, [types.Graph]],

  // void TF_GraphImportGraphDef(Graph, Buffer graph_def, ImportGraphDefOptions options, Status)
  TF_GraphImportGraphDef: [types.Void,
                           [types.Graph, types.Buffer, types.ImportGraphDefOptions, types.Status]],

  // Operation TF_GraphOperationByName(Graph graph, char* oper_name);
  TF_GraphOperationByName: [types.Operation, [types.Graph, types.String]],

  // SessionOptions TF_NewSessionOptions()
  TF_NewSessionOptions: [types.SessionOptions, []],

  // void TF_DeleteSessionOptions(Graph)
  TF_DeleteSessionOptions: [types.Void, [types.SessionOptions]],

  // Session TF_NewSession(Graph graph, SessionOptions options, Status status);
  TF_NewSession: [types.Session, [types.Graph, types.SessionOptions, types.Status]],

  // void TF_DeleteSessionOptions(Graph)
  TF_DeleteSession: [types.Void, [types.Session, types.Status]],

  // void TF_SessionRun(Session, Buffer options,
  //                    Input* input_ops, Tensor* input_values, int inputs,
  //                    Output* output_ops, Tensor* output_values, int outputs,
  //                    Operation* target_ops, int targets,
  //                    Buffer metadata, Status)
  TF_SessionRun: [types.Void, [types.Session, types.Buffer,
                               types.OperationValueArray, types.TensorArray, types.Int,
                               types.OperationValueArray, types.TensorArray, types.Int,
                               types.OperationArray, types.Int,
                               types.Buffer, types.Status]]
};

const library = ffi.Library(path.join(libPath, 'libtensorflow'), libApi);
library.Protos = require('./messages');
library.ApiTypes = types;
library.Status = library.TF_NewStatus();
library.StatusCodes = statusCodes;
library.Types = tensorTypes;

// A no-op deallocator, that can be passed in when creating tensors.
// The buffer allocated to hold tensors is automatically freed up within node.js.
library.TensorDeallocator = ffi.Callback(types.Void, [types.Any, types.Size, types.Any],
                                         function() {});

module.exports = library;
