// index.js
// Defines the TensorFlow module.
//

const api = require('./interop/api'),
      tensor = require('./tensor'),
      graph = require('./graph');

module.exports = {
  Types: api.Types,
  graph: graph.create,
  tensor: tensor.create,
};
