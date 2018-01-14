// session.js
// Defines the Session class.
//

'use strict';

const api = require('./interop/api'),
      Graph = require('./graph'),
      Tensor = require('./tensor');

function createRunParameters(graph, inputs, outputs, targets) {
  let params = {
    inputOps: null,
    inputTensors: null,
    inputs: 0,
    outputOps: null,
    outputTensors: null,
    outputs: 0,
    targetOps: null,
    targets: 0
  };

  if (inputs) {
    params.inputOps = [];
    params.inputTensors = [];

    for (let op in inputs) {
      let parts = op.split(':');
      let name = parts[0];
      let opReference = graph.ops[name];

      if (!opReference) {
        throw new Error(`The input "${name}" wasn't loaded or doesn't exist in the graph.`);
      }

      params.inputOps.push(api.ApiTypes.OperationValue({op: opReference, index: parts[1] || 0}));
      params.inputTensors.push(inputs[op].handle);

      params.inputs++;
    }

    params.inputOps = api.ApiTypes.OperationValueArray(params.inputOps);
    params.inputTensors = api.ApiTypes.TensorArray(params.inputTensors);
  }

  if (outputs) {
    params.outputs = outputs.length;
    params.outputOps = outputs.map((o) => {
      let parts = o.split(':');
      let name = parts[0];
      if (!graph.ops[name]) {
        throw new Error(`The output "${name}" wasn't loaded or doesn't exist in the graph.`);
      }
      return api.ApiTypes.OperationValue({op: graph.ops[name], index: parts[1] || 0});
    });
    params.outputOps = api.ApiTypes.OperationValueArray(params.outputOps);
    params.outputTensors = api.ApiTypes.TensorArray(params.outputs);
  }

  if (targets) {
    params.targets = targets.length;
    params.targetOps = targets.map((t) => {
      if (!graph.ops[t]) {
        throw new Error(`The target "${t}" wasn't loaded or doesn't exist in the graph.`);
      }
      return graph.ops[t];
    });
    params.targetOps = api.ApiTypes.OperationArray(params.targetOps);
  }

  return params;
}

class Session extends api.Reference {

  constructor(handle, graph, ownGraph) {
    super(handle);
    this._graph = graph;
    this._ownGraph = ownGraph;
  }

  static fromGraph(graph, ownGraph) {
    if (!graph.isValid) {
      throw new Error('Referenced graph object has been deleted.');
    }

    ownGraph = ownGraph || false;

    let status = api.TF_NewStatus();
    let sessionOptions = api.TF_NewSessionOptions();
    let sessionHandle = api.TF_NewSession(graph.handle, sessionOptions, status);
    let code = api.TF_GetCode(status);

    api.TF_DeleteSessionOptions(sessionOptions);
    api.TF_DeleteStatus(status);
    if (code === api.StatusCodes.ok) {
      return new Session(sessionHandle, graph, ownGraph);
    }
    else {
      if (ownGraph) {
        graph.delete();
      }

      throw new Error('Unable to create session');
    }
  }

  static fromGraphDef(graphDef, operations) {
    let graph = Graph.fromGraphDef(graphDef, operations);
    return Session.fromGraph(graph, /* ownGraph */ true);
  }

  get graph() {
    return this._graph;
  }

  delete() {
    // Overridden, as TF_DeleteSession doesn't follow the pattern of other Delete APIs.
    if (this.isValid) {
      let status = api.TF_NewStatus();
      api.TF_DeleteSession(this._handle, status);
      api.TF_DeleteStatus(status);

      this._handle = null;
    }

    if (this._ownGraph && this._graph) {
      this._graph.delete();
      this._graph = null;
    }
  }

  run(inputs, outputs, targets) {
    let params = createRunParameters(this._graph, inputs, outputs, targets);

    let status = api.TF_NewStatus();
    api.TF_SessionRun(this._handle,
                      /* options */ null,
                      params.inputOps, params.inputTensors, params.inputs,
                      params.outputOps, params.outputTensors, params.outputs,
                      params.targetOps, params.targets,
                      /* metadata */ null,
                      status)
    let code = api.TF_GetCode(status);
    let error = null;

    if (code !== api.StatusCodes.ok) {
      let message = api.TF_Message(status);
      error = new Error(message);
    }

    api.TF_DeleteStatus(status);

    if (error) {
      throw error;
    }

    if (params.outputs) {
      let results = {};

      outputs.forEach((name, i) => {
        results[name] = new Tensor(params.outputTensors[i]);
      });

      return results;
    }

    return null;
  }
}

module.exports = Session;
