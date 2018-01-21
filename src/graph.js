// graph.js
// Implements the Graph class to represent a Graph built from a GraphDef.
//

'use strict';

const api = require('./interop/api'),
      fs = require('fs'),
      session = require('./session');


class Graph {

  constructor(graphHandle) {
    this._graphHandle = graphHandle;
    this._opCache = {};

    this._sessions = [];
  }

  delete() {
    if (this._sessions) {
      this._sessions.forEach((session) => session.delete());
      this._sessions = null;
    }

    if (this._graphHandle) {
      api.TF_DeleteGraph(this._graphHandle);
      this._graphHandle = null;
    }
  }

  createSession() {
    this._ensureValid();

    if (this._sessions === null) {
      this._sessions = [];
    }

    let s = session.create(this._graphHandle, this._opCache);
    this._sessions.push(s);

    return s;
  }

  _ensureValid() {
    if (!this._graphHandle) {
      throw new Error('The Graph instance has been deleted.');
    }
  }
}


function createGraph(graphDef) {
  let protobuf = loadGraphDef(graphDef);

  let graphDefBuffer = api.TF_NewBufferFromString(protobuf, protobuf.length);
  let graphDefOptions = api.TF_NewImportGraphDefOptions();

  let graphHandle = api.TF_NewGraph();
  api.TF_GraphImportGraphDef(graphHandle, graphDefBuffer, graphDefOptions, api.Status);

  api.TF_DeleteImportGraphDefOptions(graphDefOptions);
  api.TF_DeleteBuffer(graphDefBuffer);

  if (api.TF_GetCode(api.Status) !== api.StatusCodes.ok) {
    api.TF_DeleteGraph(graphHandle);

    let error = api.TF_Message(api.Status);
    throw new Error(error);
  }

  return new Graph(graphHandle);
}

function loadGraphDef(graphDef) {
  if (graphDef.constructor == String) {
    return fs.readFileSync(graphDef);
  }
  else if (Buffer.isBuffer(graphDef)) {
    return graphdef;
  }
  else {
    let ProtobufWriter = require('pbf');

    let writer = new ProtobufWriter();
    api.Protos.GraphDef.write(graphDef, writer);

    return writer.finish();
  }
}


module.exports = {
  create: createGraph
};
