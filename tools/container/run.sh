#!/bin/sh

SCRIPT=$0
if [ "$SCRIPT" == "-bash" ]; then
  SCRIPT=${BASH_SOURCE[0]}
fi
REPO_DIR=$(git rev-parse --show-toplevel)

docker run -it --rm --name tf-env -v $REPO_DIR:/repo -p 8080:8080 tf-env
