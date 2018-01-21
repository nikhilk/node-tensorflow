const tf = require('tensorflow');

var const1 = {
  name: 'c1',
  op: 'Const',
  attr: {
    value: {
      value: 'tensor',
      tensor: {
        dtype: 3,
        tensor_shape: { dim: [] },
        int_val: [1]
      }
    },
    dtype: {
      value: 'type',
      type: 3
    }
  }
};

var const2 = {
  name: 'c2',
  op: 'Const',
  attr: {
    value: {
      value: 'tensor',
      tensor: {
        dtype: 3,
        tensor_shape: { dim: [] },
        int_val: [41]
      }
    },
    dtype: {
      value: 'type',
      type: 3
    }
  }
};

var add = {
  name: 'sum',
  op: 'Add',
  input: [
    'c1',
    'c2'
  ],
  attr: {
    T: {
      value: 'type',
      type: 3
    }
  }
};

var graphDef = {
  node: [ const1, const2, add ]
}

let graph = tf.graph(graphDef);
let session = graph.createSession();

let results = session.run(null, ['sum'], null);

console.log(results.sum.value);

graph.delete();
