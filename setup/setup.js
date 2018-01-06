// setup.js
// Installs pre-built TensorFlow libraries released by TensorFlow.
// For more details, see https://www.tensorflow.org/install/install_c
//

'use strict';

const fs = require('fs'),
      https = require('https'),
      os = require('os'),
      path = require('path'),
      processes = require('child_process'),
      zlib = require('zlib');

const libPlatform = os.platform();

// TODO: Add support for GPU-enabled TensorFlow builds
// TODO: Add support for GPU - is there a way to detect NVIDIA GPU availability and/or
//       relevant driver/software and automatically install the right one?
//       For now, we'll use an environment variable, and otherwise default to CPU-only.
const libType = process.env['TENSORFLOW_LIB_TYPE'] || 'cpu';

// TODO: Add support for specifying the version.
//       One way is to have this node package version match, but that seems like it may not pan
//       out always.
//       For now, we'll use an environment variable, and otherwise default to 1.4.1.
const libVersion = process.env['TENSORFLOW_LIB_VERSION'] || '1.4.1';

function isInstallationRequired() {
  let libPath = process.env['TENSORFLOW_LIB_PATH'] || null;
  if (!libPath) {
    return true;
  }

  if (!fs.existsSync(path.join(libPath, 'libtensorflow.so'))) {
    console.log(`libtensorflow.so was not found at "${libPath}"`);
    process.exit(1);
  }
  if (!fs.existsSync(path.join(libPath, 'libtensorflow_framework.so'))) {
    console.log(`libtensorflow_framework.so was not found at "${libPath}"`);
    process.exit(1);
  }

  console.log(`TensorFlow libraries are already available at ${libPath}.`);
  return false;
}

function getSourceUrl() {
}

function downloadPackage(url, downloadPath, cb) {
  console.log(`Downloading ...\n${url}\n  --> ${downloadPath} ...`);

  var file = fs.createWriteStream(downloadPath);
  var request = https.get(url, function(response) {
    response.on('data', function(chunk) {
      file.write(chunk);
    })
    .on('end', function() {
      file.end(cb);
    })
  });

  request.on('error', function(e) {
    fs.unlink(downloadPath);
    cb(e);
  });
}

function expandPackage(tarPath, expandPath) {
  console.log('Expanding and installing ...')
  try {
    processes.execSync(`tar -C ${expandPath} -xzf ${tarPath}`);
  }
  catch (e) {
    console.log('Unable to setup TensorFlow libraries.');
    process.exit(1);
  }
}

function install() {
  if ((libPlatform != 'linux') && (libPlatform != 'darwin')) {
    console.log('Only Linux and Mac OS platforms are supported.\n' +
                'See https://www.tensorflow.org/install/install_c for more information');
    process.exit(1);
  }

  let url = 'https://storage.googleapis.com/tensorflow/libtensorflow/' +
            `libtensorflow-${libType}-${libPlatform}-x86_64-${libVersion}.tar.gz`;
  let tarPath = path.join(os.tmpdir(), 'tensorflow.tar.gz');
  let installPath = path.join(__dirname, '..');

  downloadPackage(url, tarPath, function(e) {
    if (e) {
      console.log(e.message);
      process.exit(1);
    }

    expandPackage(tarPath, installPath);
  });
}


if (isInstallationRequired()) {
  install();
}
