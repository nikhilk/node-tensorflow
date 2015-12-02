// test_graph.js
// Tests graph serialization

var fs = require('fs');
var tensorflow = require('./libtensorflow');

var protos = tensorflow.protos;
var node1 = {
  name: 'Const',
  op: 'Const',
  attr: {
    dtype: {
      type: protos.DataType.DT_INT32
    },
    value: {
      tensor: {
        dtype: protos.DataType.DT_INT32,
        tensor_shape: {
        },
        int_value: 1
      }
    }
  }
}

var node2 = {
  name: 'Const_1',
  op: 'Const',
  attr: {
    dtype: {
      type: protos.DataType.DT_INT32
    },
    value: {
      tensor: {
        dtype: protos.DataType.DT_INT32,
        tensor_shape: {
        },
        int_value: 41
      }
    }
  }
}


var node3 = {
  name: "result",
  op: "Add",
  input: [
    "Const",
    "Const_1"
  ],
  attr: {
    T: {
      type: protos.DataType.DT_INT32
    }
  }
}

var graph = {
  node: [
    node1,
    node2,
    node3
  ]
}

graphBuffer = protos.GraphDef.encode(graph);
fs.writeFileSync('x.graph', graphBuffer);

