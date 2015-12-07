// graph.js
// Node.js wrapper for a TensorFlow graph and related APIs.
//

var tensorflow = require('./interop/libtensorflow');
var op = require('./op.js'),
    tensor = require('./tensor.js');

var ops = [
  values: require('./ops/values')
];

function initializeOperations(proto) {
  ops.forEach(function(opGroup) {
    for (var name in opGroup) {
      proto[name] = opGroup[name];
    }
  }
}


function Graph() {
}

initializeOperations(Graph.prototype);

module.exports = Graph;
