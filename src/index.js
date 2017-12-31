// index.js
// Defines the TensorFlow module.
//

const api = require('./api');

module.exports = {
  Types: api.Types,
  Tensor: require('./tensor'),
  Graph: require('./graph'),
  Session: require('./session')
};
