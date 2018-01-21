// session.js
// Implements the Session class to wrap a TensorFlow session, and session.run functionality.
//

'use strict';

const api = require('./interop/api'),
      tensor = require('./tensor');


class Session {

  constructor(sessionHandle, graphHandle, graphOps) {
    this._sessionHandle = sessionHandle;
    this._graphHandle = graphHandle;
    this._graphOps = graphOps;
  }

  delete() {
    if (this._sessionHandle) {
      api.TF_DeleteSession(this._sessionHandle, api.Status);      
      this._sessionHandle = null;
    }
  }

  _ensureValid() {
    if (!this._sessionHandle) {
      throw new Error('The Session instance has been closed and deleted.');
    }
  }

  run(inputs, outputs, targets) {
    this._ensureValid();

    let singleOutput = false;
    if (outputs && !Array.isArray(outputs)) {
      outputs = [outputs];
      singleOutput = true;
    }
    if (targets && !Array.isArray(targets)) {
      targets = [targets];
    }

    let params = createRunParameters(this._graphHandle, this._graphOps,
                                     inputs, outputs, targets);

    api.TF_SessionRun(this._sessionHandle,
                      /* options */ null,
                      params.inputOps, params.inputTensors, params.inputs,
                      params.outputOps, params.outputTensors, params.outputs,
                      params.targetOps, params.targets,
                      /* metadata */ null,
                      api.Status);

    if (params.inputs) {
      for (let i = 0; i < params.inputs; i++) {
        api.TF_DeleteTensor(params.inputTensors[i]);
      }
    }

    let code = api.TF_GetCode(api.Status);
    if (code !== api.StatusCodes.ok) {
      let message = api.TF_Message(api.Status);
      throw new Error(message);
    }

    let results = undefined;
    if (params.outputs) {
      results = createRunResults(outputs, params.outputTensors, singleOutput);

      for (let i = 0; i < params.outputs; i++) {
        api.TF_DeleteTensor(params.outputTensors[i]);
      }
    }

    return results;
  }
}


function createSession(graphHandle, graphOps) {
  let sessionOptions = api.TF_NewSessionOptions();
  let sessionHandle = api.TF_NewSession(graphHandle, sessionOptions, api.Status);
  let code = api.TF_GetCode(api.Status);

  api.TF_DeleteSessionOptions(sessionOptions);

  if (api.TF_GetCode(api.Status) !== api.StatusCodes.ok) {
    let error = api.TF_Message(api.Status);
    throw new Error(error);
  }

  return new Session(sessionHandle, graphHandle, graphOps);
}

function createRunParameters(graphHandle, ops, inputs, outputs, targets) {
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
      let opReference = resolveOp(graphHandle, ops, name);

      params.inputOps.push(api.ApiTypes.OperationValue({op: opReference, index: parts[1] || 0}));
      params.inputTensors.push(tensor.toHandle(inputs[op]));

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
      let opReference = resolveOp(graphHandle, ops, name)

      return api.ApiTypes.OperationValue({op: opReference, index: parts[1] || 0});
    });
    params.outputOps = api.ApiTypes.OperationValueArray(params.outputOps);
    params.outputTensors = api.ApiTypes.TensorArray(params.outputs);
  }

  if (targets) {
    params.targets = targets.length;
    params.targetOps = targets.map((name) => {
      return resolveOp(graphHandle, ops, name);
    });
    params.targetOps = api.ApiTypes.OperationArray(params.targetOps);
  }

  return params;
}

function createRunResults(outputs, outputTensors, singleOutput) {
  if (singleOutput) {
    return tensor.fromHandle(outputTensors[0]);
  }

  let results = {};
  outputs.forEach((name, i) => {
    results[name] = tensor.fromHandle(outputTensors[i]);
  });

  return results;
}

function resolveOp(graphHandle, opCache, name) {
  let op = opCache[name];
  if (op !== undefined) {
    return op;
  }

  op = api.TF_GraphOperationByName(graphHandle, name);
  if (op && !op.isNull()) {
    opCache[name] = op;
    return op;
  }

  throw new Error(`An operation with the name "${name}" was not found in the graph.`);
}


module.exports = {
  create: createSession
};
