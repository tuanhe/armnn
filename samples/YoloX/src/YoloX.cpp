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

#include "YoloX.hpp"

// Application parameters
std::vector<armnn::BackendId> default_preferred_backends_order = {armnn::Compute::CpuRef};
std::vector<armnn::BackendId> preferred_backends_order;
std::string model_file_str;
std::string preferred_backend_str;
std::string input_file_str;
int nb_loops = 1;

#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.3

static const int INPUT_W = 416;
static const int INPUT_H = 416;
static const int NUM_CLASSES = 80;

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


static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (auto stride : strides)
    {
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                grid_strides.push_back((GridAndStride){g0, g1, stride});
            }
        }
    }
}


static void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, float* feat_blob, float prob_threshold, std::vector<Object>& objects)
{

    const int num_anchors = grid_strides.size();

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        const int basic_pos = anchor_idx * (NUM_CLASSES + 5);

        // yolox/models/yolo_head.py decode logic
        float x_center = (feat_blob[basic_pos+0] + grid0) * stride;
        float y_center = (feat_blob[basic_pos+1] + grid1) * stride;
        float w = exp(feat_blob[basic_pos+2]) * stride;
        float h = exp(feat_blob[basic_pos+3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = feat_blob[basic_pos+4];
        for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        } // class loop

    } // point anchor loop
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}


static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void decode_outputs(float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h) 
{
    std::vector<Object> proposals;
    std::vector<int> strides = {8, 16, 32};
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(INPUT_W, INPUT_H, strides, grid_strides);
    generate_yolox_proposals(grid_strides, prob,  BBOX_CONF_THRESH, proposals);
    std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, NMS_THRESH);


    int count = picked.size();

    std::cout << "num of boxes: " << count << std::endl;

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x) / scale;
        float y0 = (objects[i].rect.y) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
}


static int image_pre_process(const char* file, armnn::TensorInfo& tensorInfo, void* data)
{
    cv::Mat resized_img = static_resize(file);
    yolox_normalize(resized_img, data);
    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
        float c_mean = cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5){
            txt_color = cv::Scalar(0, 0, 0);
        }else{
            txt_color = cv::Scalar(255, 255, 255);
        }

        cv::rectangle(image, obj.rect, color * 255, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = obj.rect.x;
        int y = obj.rect.y + 1;
        //int y = obj.rect.y - label_size.height - baseLine;
        if (y > image.rows)
            y = image.rows;
        //if (x + label_size.width > image.cols)
            //x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
    }

    cv::imwrite("yolox_ed.jpg" , image);
    fprintf(stderr, "save vis file\n");
    /* cv::imshow("image", image); */
    /* cv::waitKey(0); */
}

static int image_post_process(const char* file, void* data)
{
    std::vector<Object> objects;
    float* output_buffer = reinterpret_cast<float*>(data);
    cv::Mat image = cv::imread(file);
    float scale = std::min(INPUT_W / (image.cols * 1.0), INPUT_H / (image.rows * 1.0));

    decode_outputs(output_buffer, objects, scale,  image.cols,  image.rows);
    //cv::imwrite("seg.jpg", result);
    draw_objects(image, objects);

    return 0;
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
    //write2buffer("input", in[0].data(), inputTensorInfos.at(0).GetNumBytes());

    // Run the inferences
    std::cout << "\ninferences are running: " << std::flush;
    for (int i = 0 ; i < nb_loops ; i++)
    {
        runtime->EnqueueWorkload(networkId, inputTensors, outputTensors);

        std::cout << "# " << std::flush;
    }
    std::cout << "\n";

    //write2buffer("CpuRef", out[0].data(), outputTensorInfos.at(0).GetNumBytes());
    //float* ptr = reinterpret_cast<float*>(out[0].data());
    //for(uint32_t i = 0 ; i <  outputTensorInfos.at(0).GetNumBytes()/4; i++)
    //{
    //    printf("%e \n", ptr[i]);
    //}

    image_post_process(input_file_str.c_str(), out[0].data());

    return 0;
}
