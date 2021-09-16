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

#include "YoloV5.hpp"

// Application parameters
std::vector<armnn::BackendId> default_preferred_backends_order = {armnn::Compute::CpuRef};
std::vector<armnn::BackendId> preferred_backends_order;
std::string model_file_str = "yolov5s_new_416x416_model_float32.tflite";
std::string preferred_backend_str;
std::string input_file_str;
int nb_loops = 1;

static const int INPUT_W = 416;
static const int INPUT_H = 416;
static const int NUM_CLASSES = 80;

float mean[3] = {0.485, 0.456, 0.406};
float stdev[3] = {0.229, 0.224, 0.225};

float threshold_nms_iou = 0.25f;
float threshold_box_confidence = 0.25f;
float threshold_class_confidence = 0.2f;

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
        std::cout << "inputTensor Shape :" << inputTensorInfo.GetShape() << std::endl;
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
        std::cout << "outputTensor Shape :" << outputTensorInfo.GetShape() << std::endl;
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
    
    cv::Mat original_image = cv::imread(input_file_str);
    image_pre_process(original_image, inputTensorInfos[0], in[0].data());
    
    // Run the inferences
    std::cout << "\ninferences are running: " << std::flush;
    for (int i = 0 ; i < nb_loops ; i++)
    {
        runtime->EnqueueWorkload(networkId, inputTensors, outputTensors);

        std::cout << "# " << std::flush;
    }
    std::cout << "\n";

    image_post_process_one_tensor(original_image, outputTensors);

    return 0;
}

int image_pre_process(const cv::Mat& src, armnn::TensorInfo& tensorInfo, void* data)
{
    cv::Mat img;
    cv::cvtColor(src, img, cv::COLOR_BGR2RGB);

    int w, h, x, y;
    float r_w = INPUT_W / (img.cols*1.0);
    float r_h = INPUT_H / (img.rows*1.0);
    if (r_h > r_w) {
        w = INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (INPUT_H - h) / 2;
    } else {
        w = r_h * img.cols;
        h = INPUT_H;
        x = (INPUT_W - w) / 2;
        y = 0;
    }
    
    cv::Mat re(h, w, CV_8UC3);
    //cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_AREA);
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    //cv::imwrite("debug1.jpg", out);

    cv::Mat fimage;
    out.convertTo(fimage, CV_32FC3, 1.f/255, 0);
    memcpy(data, fimage.data, INPUT_H * INPUT_W * 3 * sizeof(float));

    return 0;
}

void nms_cpu(std::vector<Bbox> &bboxes, float threshold) {
    if (bboxes.empty()){
        return ;
    }
    // 1.之前需要按照score排序
    std::sort(bboxes.begin(), bboxes.end(), [&](Bbox b1, Bbox b2){return b1.score>b2.score;});
    // 2.先求出所有bbox自己的大小
    std::vector<float> area(bboxes.size());
    for (int i=0; i<bboxes.size(); ++i){
        area[i] = (bboxes[i].xmax - bboxes[i].xmin) * (bboxes[i].ymax - bboxes[i].ymin);
    }
    // 3.循环
    for (int i=0; i<bboxes.size(); ++i){
        for (int j=i+1; j<bboxes.size(); ){
            float left = std::max(bboxes[i].xmin, bboxes[j].xmin);
            float right = std::min(bboxes[i].xmax, bboxes[j].xmax);
            float top = std::max(bboxes[i].ymin, bboxes[j].ymin);
            float bottom = std::min(bboxes[i].ymax, bboxes[j].ymax);
            float width = std::max(right - left + 1, 0.f);
            float height = std::max(bottom - top + 1, 0.f);
            float u_area = height * width;
            float iou = (u_area) / (area[i] + area[j] - u_area);
            if (iou>=threshold){
                bboxes.erase(bboxes.begin()+j);
                area.erase(area.begin()+j);
            }else{
                ++j;
            }
        }
    }
}

bool parse_yolov5_result(float* data, uint32_t num_boxes, int32_t src_w, int32_t src_h, std::vector<Bbox>& bboxes)
{
    for(uint32_t idx = 0; idx < num_boxes; idx++)
    {
        uint32_t index = idx * 85;
        float confidence = data[index + 4];
        //printf("%f \n", confidence);
        if(confidence <= threshold_box_confidence)
            continue;

        //find out the label index
        int32_t class_id = 0;
        float confi = 0;
        //argmax
        for (int32_t class_index = 0; class_index < NUM_CLASSES; class_index++) {
            float confidence_of_class = data[index + 5 + class_index];
            if (confidence_of_class > confi) {
                confi = confidence_of_class;
                class_id = class_index;
            }
        }

        //std::cout << "confidential :" << confi << "\n";
        //对于边框置信度小于阈值的边框,不关心其他数值,不进行计算减少计算量
        //float cof = box_prob * max_prob;                
        //if(cof < threshold_box_confidence)
        //    continue;

        //xywh - > (x1,y1),(x2,y2)
        float x = data[index + 0] * src_w;
        float y = data[index + 1] * src_h;
        float w = data[index + 2] * src_w;
        float h = data[index + 3] * src_h;

        Bbox box; 
        box.xmin = x  - w/2;
        box.ymin = y  - h/2;
        box.xmax = x  + w/2;
        box.ymax = y  + h/2;
        box.score = confidence;
        box.label_idx = class_id;
        bboxes.push_back(box);
        //printf("%f  %f  %f  %f \n", box.xmin, box.ymin , box.xmax, box.ymax);
    }
    return true;
}

float compute_iou(const Bbox& obj0, const Bbox& obj1)
{
    int32_t interx0 = (std::max)(obj0.xmin, obj1.xmin);
    int32_t intery0 = (std::max)(obj0.ymin, obj1.ymin);
    int32_t interx1 = (std::min)(obj0.xmax, obj1.xmax);
    int32_t intery1 = (std::min)(obj0.ymax, obj1.ymax);
    if (interx1 < interx0 || intery1 < intery0) 
        return 0.f;

    int32_t area0 = (obj0.ymax - obj0.ymin) * (obj0.xmax - obj0.xmin);
    int32_t area1 = (obj1.ymax - obj1.ymin) * (obj1.xmax - obj1.xmin);
    int32_t areaInter = (interx1 - interx0) * (intery1 - intery0);
    int32_t areaSum = area0 + area1 - areaInter;

    return static_cast<float>(areaInter) / areaSum;
}

void nms_new(std::vector<Bbox>& bbox_list, std::vector<Bbox>& bbox_nms_list, float threshold_nms_iou)
{
    std::sort(bbox_list.begin(), bbox_list.end(), [](Bbox const& lhs, Bbox const& rhs) {
        //if (lhs.w * lhs.h > rhs.w * rhs.h) return true;
        if (lhs.score > rhs.score) 
            return true;
        return false;
        });

    std::unique_ptr<bool[]> is_merged(new bool[bbox_list.size()]);
    for (size_t i = 0; i < bbox_list.size(); i++) 
        is_merged[i] = false;
    for (size_t index_high_score = 0; index_high_score < bbox_list.size(); index_high_score++) {
        std::vector<Bbox> candidates;
        if (is_merged[index_high_score]) 
            continue;
        candidates.push_back(bbox_list[index_high_score]);
        for (size_t index_low_score = index_high_score + 1; index_low_score < bbox_list.size(); index_low_score++) {
            if (is_merged[index_low_score]) 
                continue;
            //if (check_class_id && bbox_list[index_high_score].class_id != bbox_list[index_low_score].class_id) 
            //    continue;
            if (compute_iou(bbox_list[index_high_score], bbox_list[index_low_score]) > threshold_nms_iou) {
                candidates.push_back(bbox_list[index_low_score]);
                is_merged[index_low_score] = true;
            }
        }

        bbox_nms_list.push_back(candidates[0]);
    }
}

void scale_box(std::vector<Bbox>& bbox_list, int32_t origin_img_width, int32_t origin_img_height, int32_t target_w, int32_t target_h)
{
    float ratio = float(std::max(target_w, target_h)) / float(std::max(origin_img_width, origin_img_height));
    int32_t pad_w = (target_w - origin_img_width * ratio) / 2 ;
    int32_t pad_h = (target_h - origin_img_height * ratio) / 2 ;
     
    for(auto& box : bbox_list )
    {
        box.xmin = ( box.xmin- pad_w ) /ratio; 
        box.ymin = ( box.ymin- pad_h ) /ratio;
        box.xmax = ( box.xmax- pad_w ) /ratio;
        box.ymax = ( box.ymax- pad_h ) /ratio;

        //clip 
        box.xmin = std::max(std::min(static_cast<float>(origin_img_width ), box.xmin), 0.f);
        box.ymin = std::max(std::min(static_cast<float>(origin_img_height), box.ymin), 0.f);
        box.xmax = std::max(std::min(static_cast<float>(origin_img_width ), box.xmax), 0.f);
        box.ymax = std::max(std::min(static_cast<float>(origin_img_height), box.ymax), 0.f);

        //printf("%f  %f  %f   %f \n", box.xmin, box.ymin, box.xmax, box.ymax);
    }
}

int image_post_process_one_tensor(const cv::Mat& img, armnn::OutputTensors& outputTensors)
{
    int img_width = img.cols;
    int img_height = img.rows;

    std::vector<Bbox> bboxes;

    for(auto ot : outputTensors){
        auto tensor = ot.second;
        //std::cout << "Shape [ " << tensor.GetShape() << " ] \n";
        float* buffer = reinterpret_cast<float*>(tensor.GetMemoryArea());
        uint32_t num_boxes = tensor.GetShape()[1];
        parse_yolov5_result(buffer, num_boxes,  INPUT_W, INPUT_H, bboxes);
    }

    std::vector<Bbox> nms_bboxes;
    //nms_new(bboxes, nms_bboxes, threshold_nms_iou);
    nms_cpu(bboxes, threshold_nms_iou);
    
    scale_box(bboxes, img_width, img_height, INPUT_W, INPUT_H);
    
    //draw picture
    for(auto box : bboxes) 
    {   
        cv::rectangle(img, cv::Point(box.xmin, box.ymin), cv::Point(box.xmax, box.ymax), cv::Scalar(255, 204,0), 2);
        cv::putText(img, std::to_string(box.score) + " " + class_names[box.label_idx], cv::Point(box.xmin, box.ymin), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
    }
    cv::imwrite("YoloV5_output.jpg", img);
    return 0;
}