//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
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
#include <armnnOnnxParser/IOnnxParser.hpp>

#define CV 1

#if CV
#include "opencv2/opencv.hpp"

//using namespace cv;

#else
//image process
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>
#endif

// Application parameters
std::vector<armnn::BackendId> default_preferred_backends_order = {armnn::Compute::CpuRef};
std::vector<armnn::BackendId> preferred_backends_order;
std::string model_file_str;
std::string preferred_backend_str;
std::string input_file_str;
int nb_loops = 1;

uint32_t INPUT_W = 416;
uint32_t INPUT_H = 416;
uint32_t nClasses = 3;
float mean[3] = {0.485, 0.456, 0.406};
float stdev[3] = {0.229, 0.224, 0.225};

static void print_help(char** argv)
{
    std::cout <<
        "Usage: " << argv[0] << " -m <model .tflite>\n"
        "\n"
        "-m --model_file <.tflite file path>:  .tflite model to be executed\n"
        "-b --backend <device>:                preferred backend device to run layers on by default. Possible choices: "
                                               << armnn::BackendRegistryInstance().GetBackendIdsAsString() << "\n"
        "                                      (by default CpuAcc, CpuRef)\n"
        "-l --loops <int>:                     provide the number of times the inference will be executed\n"
        "                                      (by default nb_loops=1)\n"
        "-i --input <image>:                   provide the input image file\n"
        "--help:                               show this help\n";
    exit(1);
}

void process_args(int argc, char** argv)
{
    const char* const short_opts = "m:b:l:i:h";
    const option long_opts[] = {
        {"model_file",   required_argument, nullptr, 'm'},
        {"backend",      required_argument, nullptr, 'b'},
        {"loops",        required_argument, nullptr, 'l'},
        {"input",        required_argument, nullptr, 'i'},
        {"help",         no_argument,       nullptr, 'h'},
        {nullptr,        no_argument,       nullptr, 0}
    };

    while (true)
    {
        const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

        if (-1 == opt)
        {
            break;
        }

        switch (opt)
        {
        case 'm':
            model_file_str = std::string(optarg);
            std::cout << "model file set to: " << model_file_str << std::endl;
            break;
        case 'b':
            preferred_backend_str = std::string(optarg);
            // Overwrite the backend
            preferred_backends_order.push_back(preferred_backend_str);

            std::cout << "backend device set to:" << preferred_backend_str << std::endl;;
            break;
        case 'l':
            nb_loops = std::stoi(optarg);
            std::cout << "benchmark will execute " << nb_loops << " inference(s)" << std::endl;
            break;
        case 'i':
            input_file_str = std::string(optarg);
            std::cout << "The input file: " << input_file_str << std::endl;
            break;
        case 'h': // -h or --help
        case '?': // Unrecognized option
        default:
            print_help(argv);
            break;
        }
    }

    if (model_file_str.empty() || input_file_str.empty())
    {
        print_help(argv);
    }
}

cv::Mat static_resize(const char* input_image_path)
{
    cv::Mat image = cv::imread(input_image_path);
    float r = std::min(INPUT_W / (image.cols * 1.0), INPUT_H / (image.rows * 1.0));
    int unpad_w = r * image.cols;
    int unpad_h = r * image.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(image, re, re.size());
    cv::Mat out(INPUT_W, INPUT_H, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

void yolox_normalize(cv::Mat& img, void* data)
{
    float* buffer = reinterpret_cast<float*>(data);
	cvtColor(img, img, cv::COLOR_BGR2RGB);
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t h = 0; h < img_h; h++) {
        for (size_t w = 0; w < img_w; w++) {
            for (size_t c = 0; c < channels; c++) {
                int in_index = h * img_w * channels + w * channels + c;
                float a = (float)(((float)img.at<cv::Vec3b>(h, w)[c]/255 - mean[c])/stdev[c]);
                printf("%le \n",  a);
                buffer[in_index] = a;
            }
        }
    }
}

static int read_bin_file(const char* file, void* inputData, uint32_t inputDataSize)
{
    char * buffer;
    long size;
    std::ifstream in (file, std::ios::in| std::ios::binary |std::ios::ate);
    size = in.tellg();
    in.seekg (0, std::ios::beg);
    buffer = new char [size];
    in.read (buffer, size);
    in.close();
    memcpy(inputData, buffer, size);
    delete[] buffer;

    return 0;
}

static int image_pre_process(const char* file, armnn::TensorInfo& tensorInfo, void* data)
{
    cv::Mat resized_img = static_resize(file);
    yolox_normalize(resized_img, data);
    return 0;
}

static int image_post_process(const char* file, void* data)
{
	cv::Mat mask = cv::Mat(INPUT_W, INPUT_H, CV_8UC1, data);
    cv::imwrite("debug0.jpg", mask);
    cv::cvtColor(mask, mask, cv::COLOR_GRAY2RGB);
    cv::imwrite("debug1.jpg", mask);
    
    cv::Mat srcTest = cv::imread(file);
    int origWidth = srcTest.cols;
    int origHeight = srcTest.rows;
    std::cout << "origWidth" << origWidth << "\n";
    std::cout << "origHeight" << origHeight << "\n";
    cv::resize(mask, mask, cv::Size(origWidth, origHeight), 0, 0, cv::INTER_CUBIC);
    cv::imwrite("debug2.jpg", mask);
    cv::threshold(mask, mask, 128, 255, cv::THRESH_BINARY);
    cv::imwrite("debug3.jpg", mask);
    
    //seems no useful
    //cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    //cv::dilate(mask, mask, element);    
    cv::Mat bgMask = ~mask;
    cv::Mat result;
    cv::GaussianBlur(bgMask, bgMask, cv::Size(), 10);
    cv::add(srcTest, bgMask, result);
    
    cv::imwrite("seg.jpg", result);

    return 0;
}

static int write2buffer(std::string backendDev, void* data, uint32_t size)
{
    std::string out_name = backendDev + ".bin";
    
    out_name = std::to_string(size) + out_name;
    const char* out_file = out_name.data();
    FILE* fp = fopen(out_file, "wb");
    if (fp == nullptr)
    {
        std::cout << "open output file fail\n";
        return -1;
    }
    
    fwrite(data, 1, size, fp);
    fflush(fp);
    fclose(fp);
    return 0;
}

int main(int argc, char* argv[])
{
    std::vector<double> inferenceTimes;
    armnn::ConfigureLogging(true, true, armnn::LogSeverity::Info);

    // Get options
    process_args(argc, argv);

    // Create the runtime
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));


    // Create Parser
    armnnTfLiteParser::ITfLiteParserPtr armnnparser(armnnTfLiteParser::ITfLiteParser::Create());

    // Create a network
    armnn::INetworkPtr network = armnnparser->CreateNetworkFromBinaryFile(model_file_str.c_str());
    if (!network)
    {
        throw armnn::Exception("Failed to create an ArmNN network");
    }

    //network->PrintGraph();

    // Optimize the network
    if (preferred_backends_order.size() == 0)
    {
        preferred_backends_order = default_preferred_backends_order;
    }
    armnn::IOptimizedNetworkPtr optimizedNet = armnn::Optimize(*network,
                                                               preferred_backends_order,
                                                               runtime->GetDeviceSpec());
    //network->PrintGraph();
    //optimizedNet->PrintGraph();
    
    armnn::NetworkId networkId;

    // Load the network in to the runtime
    runtime->LoadNetwork(networkId, std::move(optimizedNet));

    // Check the number of subgraph
    if (armnnparser->GetSubgraphCount() != 1)
    {
        std::cout << "Model with more than 1 subgraph is not supported by this benchmark application.\n";
        exit(0);
    }
    size_t subgraphId = 0;

    // Set up the input network
    std::cout << "\nModel information:" << std::endl;
    std::vector<armnnTfLiteParser::BindingPointInfo> inputBindings;
    std::vector<armnn::TensorInfo>                   inputTensorInfos;
    std::vector<std::string> inputTensorNames = armnnparser->GetSubgraphInputTensorNames(subgraphId);
    for (unsigned int i = 0; i < inputTensorNames.size() ; i++)
    {
        std::cout << "inputTensorNames[" << i << "] = " << inputTensorNames[i] << std::endl;
        armnnTfLiteParser::BindingPointInfo inputBinding = armnnparser->GetNetworkInputBindingInfo(
                                                                           subgraphId,
                                                                           inputTensorNames[i]);
        armnn::TensorInfo inputTensorInfo = runtime->GetInputTensorInfo(networkId, inputBinding.first);
        std::cout << "inputTensorsize[" << i << "] = " << inputTensorInfo.GetNumBytes() << std::endl;
        inputBindings.push_back(inputBinding);
        inputTensorInfos.push_back(inputTensorInfo);
    }

    // Set up the output network
    std::vector<armnnTfLiteParser::BindingPointInfo> outputBindings;
    std::vector<armnn::TensorInfo>                   outputTensorInfos;
    std::vector<std::string> outputTensorNames = armnnparser->GetSubgraphOutputTensorNames(subgraphId);
    for (unsigned int i = 0; i < outputTensorNames.size() ; i++)
    {
        std::cout << "outputTensorNames[" << i << "] = " << outputTensorNames[i] << std::endl;
        armnnTfLiteParser::BindingPointInfo outputBinding = armnnparser->GetNetworkOutputBindingInfo(
                                                                             subgraphId,
                                                                             outputTensorNames[i]);
        armnn::TensorInfo outputTensorInfo = runtime->GetOutputTensorInfo(networkId, outputBinding.first);
        std::cout << "outputTensorsize[" << i << "] = " << outputTensorInfo.GetNumBytes() << std::endl;
        outputBindings.push_back(outputBinding);
        outputTensorInfos.push_back(outputTensorInfo);
    }

    // Allocate input tensors
    unsigned int nb_inputs = armnn::numeric_cast<unsigned int>(inputTensorInfos.size());
    armnn::InputTensors inputTensors;
    std::vector<std::vector<float>> in;
    for (unsigned int i = 0 ; i < nb_inputs ; i++)
    {
        std::vector<float> in_data(inputTensorInfos.at(i).GetNumElements());
        in.push_back(in_data);
        inputTensors.push_back({ inputBindings[i].first, armnn::ConstTensor(inputBindings[i].second, in[i].data()) });
    }

    // Allocate output tensors
    unsigned int nb_ouputs = armnn::numeric_cast<unsigned int>(outputTensorInfos.size());
    armnn::OutputTensors outputTensors;
    std::vector<std::vector<float>> out;
    for (unsigned int i = 0; i < nb_ouputs ; i++)
    {
        std::vector<float> out_data(outputTensorInfos.at(i).GetNumElements());
        out.push_back(out_data);
        outputTensors.push_back({ outputBindings[i].first, armnn::Tensor(outputBindings[i].second, out[i].data()) });
    }
    
    image_pre_process(input_file_str.c_str(), inputTensorInfos[0], in[0].data());
    //read_bin_file(input_file_str.c_str(), in[0].data(), inputTensorInfos[0].GetNumElements());
    write2buffer("input", in[0].data(), inputTensorInfos.at(0).GetNumBytes());

    // Run the inferences
    std::cout << "\ninferences are running: " << std::flush;
    for (int i = 0 ; i < nb_loops ; i++)
    {
        runtime->EnqueueWorkload(networkId, inputTensors, outputTensors);

        std::cout << "# " << std::flush;
    }
    std::cout << "\n";

    write2buffer("CpuRef", out[0].data(), outputTensorInfos.at(0).GetNumBytes());

    float* ptr = reinterpret_cast<float*>(out[0].data());
    for(uint32_t i = 0 ; i <  outputTensorInfos.at(0).GetNumBytes()/4; i++)
    {
        printf("%e \n", ptr[i]);
    }

    //image_post_process(input_file_str.c_str(), out[0].data());

    return 0;
}
