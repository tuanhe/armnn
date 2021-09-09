//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <algorithm>
#include <getopt.h>
#include <numeric>
#include <signal.h>
#include <string>
#include <sys/time.h>
#include <vector>

#include <armnn/Utils.hpp>
#include <armnn/BackendId.hpp>
#include <armnn/BackendRegistry.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnnTfLiteParser/ITfLiteParser.hpp>

#include "opencv2/opencv.hpp"

const char* class_names[] = 
{   "background",
	"person", "bicycle", "car", "motorcycle", "airplane", "bus",
	"train", "truck", "boat", "traffic light", "fire hydrant",
	"stop sign", "parking meter", "bench", "bird", "cat", "dog",
	"horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
	"skis", "snowboard", "sports ball", "kite", "baseball bat",
	"baseball glove", "skateboard", "surfboard", "tennis racket",
	"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
	"banana", "apple", "sandwich", "orange", "broccoli", "carrot",
	"hot dog", "pizza", "donut", "cake", "chair", "couch",
	"potted plant", "bed", "dining table", "toilet", "tv", "laptop",
	"mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
	"toaster", "sink", "refrigerator", "book", "clock", "vase",
	"scissors", "teddy bear", "hair drier", "toothbrush"
};

const unsigned char colors[81][3] = {
	{56, 0, 255},
	{226, 255, 0},
	{0, 94, 255},
	{0, 37, 255},
	{0, 255, 94},
	{255, 226, 0},
	{0, 18, 255},
	{255, 151, 0},
	{170, 0, 255},
	{0, 255, 56},
	{255, 0, 75},
	{0, 75, 255},
	{0, 255, 169},
	{255, 0, 207},
	{75, 255, 0},
	{207, 0, 255},
	{37, 0, 255},
	{0, 207, 255},
	{94, 0, 255},
	{0, 255, 113},
	{255, 18, 0},
	{255, 0, 56},
	{18, 0, 255},
	{0, 255, 226},
	{170, 255, 0},
	{255, 0, 245},
	{151, 255, 0},
	{132, 255, 0},
	{75, 0, 255},
	{151, 0, 255},
	{0, 151, 255},
	{132, 0, 255},
	{0, 255, 245},
	{255, 132, 0},
	{226, 0, 255},
	{255, 37, 0},
	{207, 255, 0},
	{0, 255, 207},
	{94, 255, 0},
	{0, 226, 255},
	{56, 255, 0},
	{255, 94, 0},
	{255, 113, 0},
	{0, 132, 255},
	{255, 0, 132},
	{255, 170, 0},
	{255, 0, 188},
	{113, 255, 0},
	{245, 0, 255},
	{113, 0, 255},
	{255, 188, 0},
	{0, 113, 255},
	{255, 0, 0},
	{0, 56, 255},
	{255, 0, 113},
	{0, 255, 188},
	{255, 0, 94},
	{255, 0, 18},
	{18, 255, 0},
	{0, 255, 132},
	{0, 188, 255},
	{0, 245, 255},
	{0, 169, 255},
	{37, 255, 0},
	{255, 0, 151},
	{188, 0, 255},
	{0, 255, 37},
	{0, 255, 0},
	{255, 0, 170},
	{255, 0, 37},
	{255, 75, 0},
	{0, 0, 255},
	{255, 207, 0},
	{255, 0, 226},
	{255, 245, 0},
	{188, 255, 0},
	{0, 255, 18},
	{0, 255, 75},
	{0, 255, 151},
	{255, 56, 0},
	{245, 255, 0}
};

struct Box2f
{
    float cx;
    float cy;
    float w;
    float h;
};

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<float> maskdata;
    cv::Mat mask;
};