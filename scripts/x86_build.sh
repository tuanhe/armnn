#!/bin/bash

CURRENT_PATH=$(readlink -f "$(dirname "$0")")
BASEDIR=$CURRENT_PATH/../..

echo $CURRENT_PATH
echo $BASEDIR

cmake .. \
    -DBOOST_ROOT=$BASEDIR/boost_x86_install/ \
    -DBUILD_TF_LITE_PARSER=1 \
	-DTF_LITE_GENERATED_PATH=$BASEDIR/tflite \
	-DFLATBUFFERS_ROOT=$BASEDIR/flatbuffers \
	-DOPENCV_ROOT=$BASEDIR/opencv/share/OpenCV \
	-DARMNNREF=1 \
    -DBUILD_TESTS=1 \
	-DBUILD_SAMPLE_APP=1 


make -j12