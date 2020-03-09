class Utils {
  constructor() {
    this.rawModel;
    this.labels;
    this.model;
    this.inputTensor;
    this.outputTensor;
    this.referenceTensor;
    this.modelFile;
    this.labelsFile;
    this.inputSize;
    this.outputSize;
    this.preOptions;
    this.postOptions;
    this.updateProgress;
    this.backend = '';
    this.prefer = '';
    this.initialized = false;
    this.loaded = false;
    this.resolveGetRequiredOps = null;
    this.outstandingRequest = null;
    this.frameError = {};
    this.totalError = {};
  }

  async loadModel(model) {
    if (this.loaded && this.modelFile === model.modelFile) {
      return 'LOADED';
    }
    // reset all states
    this.loaded = this.initialized = false;
    this.backend = this.prefer = '';

    // set new model params
    this.inputSize = model.inputSize;
    this.outputSize = model.outputSize;
    this.sampleRate = model.sampleRate;
    this.modelFile = model.modelFile;
    this.labelsFile = model.labelsFile;
    this.preOptions = model.preOptions || {};
    this.postOptions = model.postOptions || {};
    this.isQuantized = model.isQuantized || false;
    let typedArray;
    if (this.isQuantized) {
      typedArray = Uint8Array;
    } else {
      typedArray = Float32Array;
    }
    this.inputTensor = new typedArray(this.inputSize.reduce((a, b) => a * b));
    this.outputTensor = new typedArray(this.outputSize.reduce((a, b) => a * b));
    this.referenceTensor = new typedArray(this.outputSize.reduce((a, b) => a * b));

    let arrayBuffer = await this.loadUrl(this.modelFile, true, true);
    let resultBytes = new Uint8Array(arrayBuffer);

    switch (this.modelFile.split('.').pop()) {
      case 'tflite':
        let flatBuffer = new flatbuffers.ByteBuffer(resultBytes);
        this.rawModel = tflite.Model.getRootAsModel(flatBuffer);
        this.rawModel._rawFormat = 'TFLITE';
        printTfLiteModel(this.rawModel);
        break;
      case 'onnx':
        let err = onnx.ModelProto.verify(resultBytes);
        if (err) {
          throw new Error(`Invalid model ${err}`);
        }
        this.rawModel = onnx.ModelProto.decode(resultBytes);
        this.rawModel._rawFormat = 'ONNX';
        printOnnxModel(this.rawModel);
        break;
      case 'bin':
        const networkFile = this.modelFile.replace(/bin$/, 'xml');
        const networkText = await this.loadUrl(networkFile, false, false);
        const weightsBuffer = resultBytes.buffer;
        this.rawModel = new OpenVINOModel(networkText, weightsBuffer);
        this.rawModel._rawFormat = 'OPENVINO';
        break;
      default:
        throw new Error('Unrecognized model format');
    }
    this.loaded = true;
    return 'SUCCESS';
  }

  async init(backend, prefer) {
    if (!this.loaded) {
      return 'NOT_LOADED';
    }
    if (this.initialized && backend === this.backend && prefer === this.prefer) {
      return 'INITIALIZED';
    }
    this.initialized = false;
    this.backend = backend;
    this.prefer = prefer;
    let configs = {
      rawModel: this.rawModel,
      backend: backend,
      prefer: prefer,
      softmax: this.postOptions.softmax || false,
    };
    switch (this.rawModel._rawFormat) {
      case 'TFLITE':
        this.model = new TFliteModelImporter(configs);
        break;
      case 'ONNX':
        this.model = new OnnxModelImporter(configs);
        break;
      case 'OPENVINO':
        this.model = new OpenVINOModelImporter(configs);
        break;
    }
    let result = await this.model.createCompiledModel();
    console.log(`compilation result: ${result}`);
    let start = performance.now();
    result = await this.model.compute([this.inputTensor], [this.outputTensor]);
    let elapsed = performance.now() - start;
    console.log(`warmup time: ${elapsed.toFixed(2)} ms`);
    this.initialized = true;

    if (this.resolveGetRequiredOps) {
      this.resolveGetRequiredOps(this.model.getRequiredOps());
    }

    return 'SUCCESS';
  }

  async getRequiredOps() {
    if (!this.initialized) {
      return new Promise(resolve => this.resolveGetRequiredOps = resolve);
    } else {
      return this.model.getRequiredOps();
    }
  }

  getSubgraphsSummary() {
    if (this.model._backend !== 'WebML' &&
        this.model &&
        this.model._compilation &&
        this.model._compilation._preparedModel) {
      return this.model._compilation._preparedModel.getSubgraphsSummary();
    } else {
      return [];
    }
  }

  async predict(ark) {
    if (!this.initialized) return;
    await this.prepareInputTensor(this.inputTensor, ark);
    let start = performance.now();
    await this.model.compute([this.inputTensor], [this.outputTensor]);
    let elapsed = performance.now() - start;
    console.log('Output:', this.outputTensor)
    return {
      time: elapsed.toFixed(2),
      result: this.outputTensor
    };
  }

  async loadUrl(url, binary, progress) {
    return new Promise((resolve, reject) => {
      if (this.outstandingRequest) {
        this.outstandingRequest.abort();
      }
      let request = new XMLHttpRequest();
      this.outstandingRequest = request;
      request.open('GET', url, true);
      if (binary) {
        request.responseType = 'arraybuffer';
      }
      request.onload = function(ev) {
        this.outstandingRequest = null;
        if (request.readyState === 4) {
          if (request.status === 200) {
            resolve(request.response);
          } else {
            reject(new Error('Failed to load ' + url + ' status: ' + request.status));
          }
        }
      };
      if (progress && typeof this.updateProgress !== 'undefined') {
        request.onprogress = this.updateProgress;
      }
      request.send();
    });
  }

  async prepareInputTensor(tensor, ark) {
    let request = new Request(ark);
    let response = await fetch(request);
    let arkFileBuffer = await response.arrayBuffer();
    let value = new Float32Array(arkFileBuffer);
    let suba = value.subarray(6, 6 + 440);
    tensor.set(suba,0);
    console.log("Input tensor", tensor);
  }

  downloadArkFile() {
    if (!this.outputTensor) {
        console.error("saveArkFile: No data");
        return;
    }
    let Data = new Float32Array(this.outputTensor);
    let filename = "output.ark";
    var blob = new Blob([Data], { type: 'text/json' }),
        e = document.createEvent('MouseEvents'),
        a = document.createElement('a');
    a.download = filename;
    a.href = window.URL.createObjectURL(blob);
    a.dataset.downloadurl = ['text/json', a.download, a.href];
    e.initMouseEvent('click', true, false, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
    a.dispatchEvent(e);
    console.log("Converted output tensor to ${filename}.");
  }

  async loadReferenceArk(referenceTensor,ark) {
    let request = new Request(ark);
    let response = await fetch(request);
    let arkFileBuffer = await response.arrayBuffer();
    let value = new Float32Array(arkFileBuffer);
    let suba = value.subarray(6, 6 + 3425);
    referenceTensor.set(suba,0);
    console.log("reference tensor", referenceTensor);
  }
  async initError(tensor) {
    tensor.numScores = 0,
    tensor.numErrors = 0,
    tensor.threshold = 0.001,
    tensor.maxError = 0.0,
    tensor.rmsError = 0.0,
    tensor.sumError = 0.0,
    tensor.sumRmsError = 0.0,
    tensor.sumSquaredError = 0.0,
    tensor.maxRelError = 0.0,
    tensor.sumRelError = 0.0,
    tensor.sumSquaredRelError = 0.0
}

  async compareScores(outputTensor,referenceTensor,numRows,numColumns) {
  let numErrors = 0; 
  await this.initError(this.frameError);
  for (let i =0; i< numRows; i++) {
    for (let j =0; j< numColumns; j++) {
      let score = outputTensor[i*numColumns+j];
      let refScore = referenceTensor[i * numColumns + j];
      let error = Math.abs(refScore - score);
      let rel_error = error / ((Math.abs(refScore)) + 1e-20);
      let squared_error = error * error;
      let squared_rel_error = rel_error * rel_error;
      this.frameError.numScores++;
      this.frameError.sumError += error;
      this.frameError.sumSquaredError += squared_error;
      if (error > this.frameError.maxError) {
        this.frameError.maxError = error;
      }
      this.frameError.sumRelError += rel_error;
      this.frameError.sumSquaredRelError += squared_rel_error;
      if (rel_error > this.frameError.maxRelError) {
        this.frameError.maxRelError = rel_error;
      }
      if (error > this.frameError.threshold) {
        numErrors++;
      }
    }
    }
    this.frameError.rmsError = Math.sqrt(this.frameError.sumSquaredError / (numRows * numColumns));
    this.frameError.sumRmsError += this.frameError.rmsError;
    this.frameError.numErrors = numErrors;
    return numErrors;
  }

  async UpdateScoreError(frameError,totalError) {
    totalError.numErrors += frameError.numErrors;
    totalError.numScores += frameError.numScores;
    totalError.sumRmsError += frameError.rmsError;
    totalError.sumError += frameError.sumError;
    totalError.sumSquaredError += frameError.sumSquaredError;
    if (frameError.maxError > totalError.maxError) {
      totalError.maxError = frameError.maxError;
    }
    totalError.sumRelError += frameError.sumRelError;
    totalError.sumSquaredRelError += frameError.sumSquaredRelError;
    if (frameError.maxRelError > totalError.maxRelError) {
      totalError.maxRelError = frameError.maxRelError;
    }
  }

  async printReferenceCompareResults(totalError,framesNum) {//framesNum equals to number of frames in one utterance
  console.log("         max error: ", totalError.maxError);
  console.log("         avg error: ", totalError.sumError / totalError.numScores);
  console.log("         avg error of: ", totalError.sumRmsError / framesNum);  
  console.log("         stdev error: ", await this.StdDevError(totalError));
  }

  async StdDevError(totalError) {
  let result = Math.sqrt(totalError.sumSquaredError / totalError.numScores
               - (totalError.sumError / totalError.numScores) * (totalError.sumError / totalError.numScores));
  return result;
  }

  deleteAll() {
    if (this.model._backend != 'WebML') {
      this.model._compilation._preparedModel._deleteAll();
    }
  }

}
