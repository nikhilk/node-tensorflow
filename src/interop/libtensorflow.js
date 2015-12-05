// interface.js
// Wrapper for the TensorFlow C interface.
//

'use strict';

var fs = require('fs'),
    path = require('path'),
    protobuf = require('protocol-buffers'),
    ref = require('ref'),
    refArray = require('ref-array');
var ffi = require('./ffiplus');

// Proto definitions
var protos = protobuf(fs.readFileSync(path.join(__dirname,
                                                'tensorflow.proto')));


// Enum definitions

function _buildEnumLookup(map) {
  var lookup = [];
  for (var n in map) {
    lookup[map[n]] = n;
  }

  return lookup;
}

// TODO(nikhilk): Might be able to replace all this with T/F.
//                Check which ones does TensorFlow actually use.
var statusCodes = {
  'ok': 0,
  'cancelled': 1,
  'unknown': 2,
  'invalidArgument': 3,
  'deadlineExceeded': 4,
  'notFound': 5,
  'alreadyExists': 6,
  'permissionDenied': 7,
  'resourceExhausted': 8,
  'failedPrecondition': 9,
  'aborted': 10,
  'outOfRange': 11,
  'unimplemented': 12,
  'internal': 13,
  'unavailable': 14,
  'dataLoss': 15,
  'unauthenticated': 16
};
var _statusCodeNames = _buildEnumLookup(statusCodes);

var dataTypes = {
  'float': 1,
  'double': 2,
  'int32': 3,
  'uint8': 4,
  'int16': 5,
  'int8': 6,
  'string': 7,
  'complex': 8,
  'int64': 9,
  'bool': 10,
  'qint8': 11,
  'quint8': 12,
  'qint32': 13,
  'bfloat16': 14
};
var _dataTypeNames = _buildEnumLookup(dataTypes);


// Type definitions
var types = {
  Pointer: ref.refType('void'),
  Status: ref.refType('void'),
  Tensor: ref.refType('void'),
  Session: ref.refType('void'),
  SessionOptions: ref.refType('void'),
  LongLongArray: refArray('longlong'),
  FloatArray: refArray('float'),
  StringArray: refArray('string')
};
types.TensorArray = refArray(types.Tensor);


// TensorFlow library definition

var tensorflow =
  ffi.defineLibrary(path.join(__dirname, 'libtensorflow'))
     .export('createStatus', 'TF_NewStatus', types.Status)
     .export('deleteStatus', 'TF_DeleteStatus', 'void', types.Status)
     .export('updateStatus', 'TF_SetStatus',
             'void',
             types.Status,
             /* code */ 'int',
             /* message */ 'string')
     .export('_statusCode', 'TF_GetCode', 'int', types.Status)
     .export('statusMessage', 'TF_Message', 'string', types.Status)
     .export('_createTensor', 'TF_NewTensor',
             types.Tensor,
             /* datatype */ 'int',
             /* dim lengths */ types.LongLongArray,
             /* dims */ 'int',
             /* data */ types.Pointer,
             /* length */ 'size_t',
             /* dealloc */ types.Pointer,
             /* deallocarg */ types.Pointer)
     .export('deleteTensor', 'TF_DeleteTensor', 'void', types.Tensor)
     .export('_tensorType', 'TF_TensorType', 'int', types.Tensor)
     .export('tensorDimensions', 'TF_NumDims', 'int', types.Tensor)
     .export('tensorDimensionLength', 'TF_Dim',
             'longlong',
             types.Tensor,
             /* dimension index */ 'int')
     .export('tensorDataLength', 'TF_TensorByteSize', 'size_t', types.Tensor)
     .export('tensorData', 'TF_TensorData', types.Pointer, types.Tensor)
     .export('createSessionOptions', 'TF_NewSessionOptions',
             types.SessionOptions)
     .export('deleteSessionOptions', 'TF_DeleteSessionOptions',
             'void',
             types.SessionOptions)
     .export('setSessionTarget', 'TF_SetTarget', types.SessionOptions, 'string')
     .export('setSessionConfig', 'TF_SetConfig',
             types.SessionOptions,
             /* graph proto */ types.Pointer,
             /* graph proto length */ 'size_t',
             types.Status) 
     .export('createSession', 'TF_NewSession',
             types.Session,
             types.SessionOptions,
             types.Status)
     .export('deleteSession', 'TF_DeleteSession',
             'void',
             types.Session,
             types.Status)
     .export('closeSession', 'TF_CloseSession',
             'void',
             types.Session,
             types.Status)
     .export('extendGraph', 'TF_ExtendGraph',
             'void',
             types.Session,
             /* graph proto */ types.Pointer,
             /* graph proto length */ 'size_t',
             types.Status)
     .export('run', 'TF_Run',
             'void',
             types.Session,
             /* input names */ types.StringArray,
             /* input tensors */ types.TensorArray,
             /* inputs */ 'int',
             /* output names */ types.StringArray,
             /* outputs tensors */ types.TensorArray,
             /* outputs */ 'int',
             /* node names */ types.StringArray,
             /* nodes */ 'int',
             /* status */ types.Status)
     .create();


tensorflow.statusCodes = statusCodes;
tensorflow.dataTypes = dataTypes;
tensorflow.types = types;
tensorflow.protos = protos;


// Helper Methods

tensorflow.statusCode = function(status) {
  var statusCode = tensorflow._statusCode(status);
  return _statusCodeNames[statusCode];
}

tensorflow.success = function(status) {
  return tensorflow._statusCode(status) === statusCodes.ok;
}


// A no-op deallocator, that can be passed in when creating tensors.
// The buffer allocated to hold tensors is automatically freed up within the
// nodejs environment.
var _tensorDeallocator =
  ffi.Callback('void', [ types.Pointer, 'size_t', types.Pointer ],
               function() { });

tensorflow.createTensor = function(dataType, dimensionLengths, dimensions,
                                   data, length) {
  return tensorflow._createTensor(dataType, dimensionLengths, dimensions,
                                  data, length,
                                  _tensorDeallocator, null);
}

tensorflow.tensorType = function(tensor) {
  var dataType = tensorflow._tensorType(tensor);
  return _dataTypeNames[dataType];
}

tensorflow.tensorRead = function(tensor) {
  var length = tensorflow.tensorDataLength(tensor);
  var type = tensorflow.tensorType(tensor);

  var pointer = tensorflow.tensorData(tensor);
  pointer.type = type

  return pointer.reinterpret(length, 0);
}


module.exports = tensorflow;

