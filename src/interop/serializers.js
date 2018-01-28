// serializers.js
// Implements serialization logic to convert between raw C Tensor representation and script types.
//

const api = require('./api'),
      os = require('os');

class GenericSerializer {

  fromBuffer(buffer, shape) {
    return buffer;
  }

  toBuffer(data) {
    if (Buffer.isBuffer(data)) {
      return data;
    }

    throw new Error('Unsupported Tensor data.');
  }
}

class NumberSerializer {

  constructor(readFn, size) {
    this._readFn = Buffer.prototype[readFn];
    this._size = size;
  }

  fromBuffer(buffer, shape) {
    if (shape.length === 0) {
      // Scalar tensor value
      return this._readFn.call(buffer, 0);
    }
    else {
      let array = createItemArray(shape);

      for (let i = 0; i < array.length; i++) {
        array[i] = this._readFn.call(buffer, i * this._size);
      }

      return array;
    }
  }
}

class Int32Serializer extends NumberSerializer {

  constructor() {
    super('readInt32' + os.endianness(), /* size */ 4)
  }

  toBuffer(data) {
    return api.ApiTypes.IntArray(data).buffer;
  }
}

class FloatSerializer extends NumberSerializer {

  constructor() {
    super('readFloat' + os.endianness(), /* size */ 4)
  }

  toBuffer(data) {
    return api.ApiTypes.FloatArray(data).buffer;
  }
}

class Int64Serializer {

  fromBuffer(buffer, shape) {
    // TODO: Handle representing int64 values in script (likely using the int64-buffer module)
    return buffer;
  }

  toBuffer(data) {
    return api.ApiTypes.LongLongArray(data).buffer;
  }
}


function createItemArray(shape) {
  // The number of items is the product of the dimensions specified by the shape.
  let totalItems = shape.reduce(function(dim, items) { return dim * items}, 1);
  return new Array(totalItems);
}


const _genericSerializer = new GenericSerializer();
const _serializers = {};
_serializers[api.Types.int32.toString()] = new Int32Serializer();
_serializers[api.Types.int64.toString()] = new Int64Serializer();
_serializers[api.Types.float.toString()] = new FloatSerializer();

function createSerializer(type) {
  return _serializers[type.toString()] || _genericSerializer;
}


module.exports = {
  create: createSerializer
};
