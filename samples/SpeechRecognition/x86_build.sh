#!/bin/bash

CURRENT_PATH=$(readlink -f "$(dirname "$0")")
BASEDIR=$CURRENT_PATH/../..
LIB_DIR=$BASEDIR/build

cmake  .. \
    -DARMNN_LIB_DIR=$LIB_DIR \
    -DBUILD_UNIT_TESTS=0
#make -j12
