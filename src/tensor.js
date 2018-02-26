// tensor.js
// Implements the Tensor class to represent tensor data and encapsulates marshalling logic.
//

'use strict';

const api = require('./interop/api'),
      serializers = require('./interop/serializers');


class Tensor {

  constructor(value, type, shape) {
    this._value = value;
    this._type = type;
    this._shape = shape;
  }

  get shape() {
    return this._shape;
  }

  get type() {
    return this._type;
  }

  get value() {
    return this._value;
  }
}


function createTensor(value, type, shape) {
  if ((value === null) || (value === undefined)) {
    throw new Error('A value representing the Tensor data must be specified.');
  }

  if (value.constructor === Tensor) {
    return value;
  }

  if (Buffer.isBuffer(value)) {
    if ((type === null) || (type === undefined) || (shape === null) || (shape === undefined)) {
      throw new Error('The type and shape of a raw tensor data buffer must be specified.');
    }

    return new Tensor(value, type, shape);
  }

  if ((shape === null) || (shape === undefined)) {
    shape = calculateShape(value);

    // Convert to value to a flat array
    if (shape.length === 0) {
      // Ensure the value is represented as an array, even for scalars
      value = [value];
    }
    else if (shape.length > 1) {
      // Flatten the value, so it can be converted into a buffer containing all the values.
      value = flattenList(value);
    }

    if (type === undefined) {
      if (value[0].constructor == Number) {
        type = api.Types.float;
      }
      else if (value[0].constructor == String) {
        type = api.Types.string;
      }
      else {
        throw new Error('Unsupported data type for creating a Tensor');
      }
    }
  }

  return new Tensor(value, type, shape);
}

function createHandleFromTensor(value) {
  let tensor = createTensor(value);

  // Convert to a buffer containing raw byte representation of the Tensor
  let serializer = serializers.create(tensor.type);
  let data = serializer.toBuffer(tensor.value);

  return api.TF_NewTensor(tensor.type,
                          api.ApiTypes.LongLongArray(tensor.shape), tensor.shape.length,
                          data, data.length,
                          api.TensorDeallocator, null);
}

function createTensorFromHandle(tensorHandle) {
  let shape = [];

  let dimensions = api.TF_NumDims(tensorHandle);
  for (let i = 0; i < dimensions; i++) {
    shape.push(api.TF_Dim(tensorHandle, i));
  }

  // Read data into a buffer and reset the current position in the buffer to be at the start.
  let dataLength = api.TF_TensorByteSize(tensorHandle);
  let data = api.TF_TensorData(tensorHandle);
  data = data.reinterpret(dataLength, 0);

  let type = api.TF_TensorType(tensorHandle);
  let serializer = serializers.create(type);

  let value = serializer.fromBuffer(data, shape);
  if ((shape.length > 1) && !Buffer.isBuffer(value)) {
    value = reshapeList(value, shape);
  }

  return new Tensor(value, type, shape);
}

function calculateShape(value) {
  if ((value.shape !== undefined) && (value.shape !== null)) {
    return value.shape;
  }

  // Detect the shape by walking the arrays (to handle nested arrays). This assumes the arrays
  // are not jagged.
  let shape = [];

  let element = value;
  while (Array.isArray(element)) {
    shape.push(element.length);
    element = element[0];
  }

  return shape;
}

// Flatten the list. This is only relevant for multi-dimensional tensors.
function flattenList(list) {
  // Make a copy, so the original tensor is not modified.
  list = [].concat(list);

  // Note that i must be checked against the length of the list each time through the loop, as the
  // list is modified within the iterations.
  for (let i = 0; i < list.length; i++) {
    if (Array.isArray(list[i])) {
      // Replace the item with the flattened version of the item (using the ... operator).
      // Replace with the items and backtrack 1 position
      list.splice(i, 1, ...list[i]);

      // Decrement i to look at the element again; we'll keep looking at this i index, until
      // the most deeply nested item has been flattened.
      i--;
    }
  }

  return list;
}

function reshapeList(list, shape) {
  // Work from the inner-most dimension to outer-most, building up arrays of items matching
  // dimension length (this is essentially the inverse of the tensorToList implementation).
  for (let i = shape.length - 1; i > 0; i--) {
    let dimension = shape[i];
    const newList = [];

    for (let j = 0; j < list.length / dimension; j++) {
      newList.push(list.slice(j * dimension, (j + 1) * dimension));
    }

    list = newList;
  }

  return list;
}


module.exports = {
  create: createTensor,
  fromHandle: createTensorFromHandle,
  toHandle: createHandleFromTensor
};
