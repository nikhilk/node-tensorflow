// tensor.js
// Defines the Tensor class.
//

'use strict';

const api = require('./interop/api'),
      os = require('os');

// TODO: Improve Tensor support, esp. for JavaScript use-cases.
// - Go beyond float32, int32. Specifically add support for strings.
// - Using typed arrays?

// Flatten a Tensor to a flat array. This is only relevant for multi-dimensional tensors.
function tensorToList(tensor) {
  // Make a copy, so the original tensor is not modified.
  let list = [].concat(tensor);

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

function listToTensor(list, shape) {
  // This modifies the list in place, given this is run on a temporary list; hence avoiding
  // copying cost.

  // Work from the inner-most dimension to outer-most, building up arrays of items matching
  // dimension length (this is essentially the inverse of the tensorToList implementation).
  for (let i = shape.length - 1; i > 0; i--) {
    let dimension = shape[i];

    for (let j = 0; j < list.length; j++) {
      let items = list.splice(j, dimension);
      list.splice(j, 0, items);
    }
  }

  return list;
}


class Tensor extends api.Reference {

  constructor(handle, type, shape, array) {
    super(handle, api.TF_DeleteTensor);
    this._type = type || 0;
    this._shape = shape || null;
    this._array = array || null;
  }

  static create(data, dataType, shape) {
    let array = data;

    // Detect the shape by walking the arrays (to handle nested arrays). This assumes the arrays
    // are not jagged.
    if (shape === undefined) {
      shape = [];

      let element = data;
      while (element.constructor == Array) {
        shape.push(element.length);
        element = element[0]
      }
    }

    // Flatten the arrays, so it can be converted into a buffer containing all the values.
    if (shape.length > 1) {
      data = tensorToList(data);
    }

    // Detect data type based on the first element. JavaScript numbers are all treated as floats.
    // If integer types are required, the caller should explicitly specify a dataType.
    if (dataType === undefined) {
      if (data[0].constructor == Number) {
        dataType = api.Types.float;
      }
      // TODO: Add support for strings
    }

    // Convert to a native array
    if (dataType === api.Types.float) {
      data = api.ApiTypes.FloatArray(data);
    }
    else if (dataType === api.Types.int32) {
      data = api.ApiTypes.IntArray(data);
    }
    else if (dataType === api.Types.int64) {
      data = api.ApiTypes.LongLongArray(data);
    }
    else {
      throw new Error('Unsupported tensor element type.');
    }

    let handle = api.TF_NewTensor(dataType, api.ApiTypes.LongLongArray(shape), shape.length,
                                data.buffer, data.buffer.length,
                                api.TensorDeallocator, null);
    if (handle === null) {
      throw new Error('Unable to allocate Tensor.');
    }

    return new Tensor(handle, dataType, shape, array);
  }

  get shape() {
    this.ensureValid();
  
    if (!this._shape) {
      let shape = [];
      let dimensions = api.TF_NumDims(this._handle);

      for (let i = 0; i < dimensions; i++) {
        shape.push(api.TF_Dim(this._handle, i));
      }

      this._shape = shape;
    }

    return this._shape;
  }

  get type() {
    this.ensureValid();

    if (this._type === 0) {
      this._type = api.TF_TensorType(this._handle);
    }

    return this._type;
  }

  toBuffer() {
    this.ensureValid();

    let length = api.TF_TensorByteSize(this._handle);
    let data = api.TF_TensorData(this._handle);

    // Reset the current position in the buffer to be at the start.
    return data.reinterpret(length, 0);
  }

  toValue() {
    this.ensureValid();

    if (!this._value) {
      let itemSize = 4;

      let reader = '';
      if (this.type == api.Types.float) {
        reader = 'readFloat' + os.endianness();
      }
      else if (this.type == api.Types.int32) {
        reader = 'readInt32' + os.endianness();
      }
      else {
        console.log('dataType = ' + this.type);
        throw new Error('Only float and int32 tensors can be converted to arrays.');
      }

      let data = this.toBuffer();
      reader = Buffer.prototype[reader];

      let shape = this.shape;
      if (shape.length === 0) {
        // Scalar tensor value
        this._value = reader.call(data, 0);
      }
      else {
        let totalItems = shape.reduce(function(dim, items) { return dim * items}, 1);
        let list = new Array(totalItems);

        for (let i = 0; i < totalItems; i++) {
          list[i] = reader.call(data, i * itemSize);
        }

        this._value = list;
        if (shape.length > 1) {
          this._value = listToTensor(list, shape);
        }
      }
    }

    return this._value;
  }
}

module.exports = Tensor;
