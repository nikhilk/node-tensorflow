#!/bin/sh

SCRIPT=$0
if [ "$SCRIPT" == "-bash" ]; then
  SCRIPT=${BASH_SOURCE[0]}
fi
REPO_DIR=$(git rev-parse --show-toplevel)

docker run -i -v $REPO_DIR:/repo/nodetf -t node-tensorflow

