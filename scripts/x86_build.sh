#!/bin/bash

SYSTEM=`uname  -s`

if [ $SYSTEM = "Darwin" ] ; then
    CURRENT_PATH=$(cd $(dirname $0); pwd)
elif
    [ $SYSTEM = "Linux" ] ; then
   	CURRENT_PATH=$(readlink -f "$(dirname "$0")")
else
    echo  "Unkown platform"
fi

BASEDIR=$CURRENT_PATH/../../armnn-dev

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