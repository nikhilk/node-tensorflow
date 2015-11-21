// ffiplus.js
// Helper methods for working with the ffi module.
//

var ffi = require('ffi');

function Library(name) {
  this._name = name;
  this._exports = { };
  this._aliases = { };
}

Library.prototype.export = function(name, rawName, returnType, argType) {
  var argTypes;
  if (argType) {
    argTypes = Array.prototype.slice.call(arguments).slice(3);
  }
  else {
    argTypes = [];
  }

  this._exports[rawName] = [ returnType, argTypes ];
  this._aliases[name] = rawName;

  return this;
}

Library.prototype.create = function() {
  var lib = ffi.Library(this._name, this._exports);
  for (var alias in this._aliases) {
    lib[alias] = lib[this._aliases[alias]];
  }

  return lib;
}

ffi.defineLibrary = function(name) {
  return new Library(name);
}

module.exports = ffi;

