# Samples List

This directory contains a few samples of using TensorFlow with node.js.

## Graphs
The graphs set of samples demonstrate using TensorFlow graphs.

### basic
'Hello World' style sample, demonstrating graph loading and execution.

### matrix
Builds on the above to use matrices instead of scalar tensors, as well as load graphs containing
namescopes. This sample will be updated to demonstrate feeding in tensors when executing graphs.

# Running Samples

Within each sample directory:

    npm install
    npm run -s sample

This will install the tensorflow node.js package along with associated TensorFlow binaries.
Once installed, running will first run the Python code to produce the TensorFlow artifact
(eg. graph) and then run the node.js sample.

## If you're making changes to the tensorflow package ...
... and you want to run the samples without first publishing the package to npm, you can
create a package from your environment and install using that to run the samples.

    # Build local tensorflow-\<version\>.tgz package
    # From the root of the repo directory...
    npm pack

    cd samples/graphs/basic
    npm install ../../../tensorflow-<version>.tgz

This will install from your local package (even if the versions have not changed).

Note that it will record this in the sample's package.json file. Be sure to revert the package.json
file to avoid committing this change.
