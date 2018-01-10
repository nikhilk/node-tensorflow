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

let session = tf.Session.fromGraphDef(graphDef, { sum: 'sum' });
let results = session.run(null, ['sum'], null);

console.log(results.sum.toValue());
results.sum.delete();
session.delete();
