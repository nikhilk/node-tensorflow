// serializers.js
// Implements serialization logic to convert between raw C Tensor representation and script types.
//

'use strict';

const api = require('./api'),
      os = require('os'),
      ref = require('ref');

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

class StringSerializer {

  // TODO: Handle the case where the input is not a string, but represented as a Buffer.

  // Strings are encoded as an array of uint64 values containing offsets into the buffer for
  // each string. Each string is a 7-bit encoded length prefix followed by the bytes.

  fromBuffer(buffer, shape) {
    let array = createItemArray(shape);
    let count = array.length;

    let header = 8 * count;
    for (let i = 0; i < count; i++) {
      let offset = ref.readUInt64LE(buffer, i * 8);
      let nextOffset = i === count - 1 ? buffer.size : ref.readUInt64LE(buffer, i * 8 + 8)

      let srcLength = nextOffset - offset;
      let srcBuffer = buffer.reinterpret(srcLength, offset + header);

      let decodedBuffer = ref.alloc(api.ApiTypes.BytePtr);
      let decodedLength = ref.alloc(api.ApiTypes.Size);
      api.TF_StringDecode(srcBuffer, srcLength, decodedBuffer, decodedLength, api.Status);

      decodedLength = decodedLength.deref();
      decodedBuffer = decodedBuffer.deref();
      array[i] = decodedBuffer.reinterpret(decodedLength, 0).toString('binary');
    }

    return array;
  }

  toBuffer(data) {
    let maxLength = 0;
    let size = 0;
    data = data.map((s) => {
      let length = Buffer.byteLength(s, 'binary');
      let encodedLength = api.TF_StringEncodedSize(length);

      let item = { s: s, offset: size, length: length, encodedLength: encodedLength };
      size += encodedLength;

      maxLength = Math.max(maxLength, length);
      return item;
    });

    // Add for the list of offsets (uint64 value per string)
    let header = 8 * data.length;
    size += header;

    // Allocate a buffer that is large enough to hold the longest string; Add 1 for trailing null.
    let srcBuffer = new Buffer(maxLength + 1);

    let buffer = new Buffer(size);
    data.forEach((item, i) => {
      ref.writeUInt64LE(buffer, i * 8, item.offset);

      ref.writeCString(srcBuffer, 0, item.s, 'binary');

      let destBuffer = ref.reinterpret(buffer, item.encodedLength, item.offset + header);
      api.TF_StringEncode(srcBuffer, item.length, destBuffer, item.encodedLength, api.Status);
    });

    return buffer;
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
_serializers[api.Types.string.toString()] = new StringSerializer();

function createSerializer(type) {
  return _serializers[type.toString()] || _genericSerializer;
}


module.exports = {
  create: createSerializer
};
