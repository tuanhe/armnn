#!/bin/bash

CURRENT_PATH=$(cd $(dirname $0); pwd)
ARMNN_ROOT_DIR=$CURRENT_PATH/../..


echo $CURRENT_PATH
echo $ARMNN_ROOT_DIR

cmake .. \
    -DOPENCV_ROOT_DIR=$ARMNN_ROOT_DIR/../opencv  \
    -DARMNN_LIB_DIR=$ARMNN_ROOT_DIR/build 
#
#make -j12