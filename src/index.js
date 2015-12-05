// index.js
// TensorFlow API for node.js applications
//

'use strict';

var tensorflow = require('./interop/libtensorflow');

module.exports = {
  Graph: require('./graph'),
  Tensor: require('./tensor'),
  Session: require('./session')
};

