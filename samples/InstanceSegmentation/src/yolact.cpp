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

#include "Yolact.hpp"
#include "opencv2/opencv.hpp"

//using namespace cv;
// Application parameters
std::vector<armnn::BackendId> default_preferred_backends_order = {armnn::Compute::CpuRef};
std::vector<armnn::BackendId> preferred_backends_order;
std::string model_file_str;
std::string preferred_backend_str;
std::string input_file_str;
int nb_loops = 1;

uint32_t nW = 550;
uint32_t nH = 550;
uint32_t nClasses = 3;
const float MEANS[3] = { 123.68, 116.78, 103.94 };
const float STD[3] = { 58.40, 57.12, 57.38 };
const float confidence_thresh = 0.5f;
const float nms_thresh = 0.5f;
const int keep_top_k = 200;

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

    if (model_file_str.empty())
    {
        print_help(argv);
    }
}

static int image_pre_process(const char* file, void* data)
{
    int nC = 3;
    cv::Mat src = cv::imread(file);
    
    cv::Mat image;
	cv::resize(src, image, cv::Size(nW, nH), 0, 0, cv::INTER_LINEAR);
	int cnls = image.type();
	if (cnls == CV_8UC1) {
		cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
	}
	else if (cnls == CV_8UC3) {
		cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	}
	else if (cnls == CV_8UC4) {
		cv::cvtColor(image, image, cv::COLOR_BGRA2RGB);
	}

    cv::Mat fimage;
    image.convertTo(fimage, CV_32FC3);

    int i = 0, j = 0;
	for (i = 0; i < fimage.rows; i++)
	{
		float* pdata = (float*)(fimage.data + i * fimage.step);
		for (j = 0; j < fimage.cols; j++)
		{
			pdata[0] = (pdata[0] - MEANS[0]) / STD[0];
			pdata[1] = (pdata[1] - MEANS[1]) / STD[1];
			pdata[2] = (pdata[2] - MEANS[2]) / STD[2];
			pdata += 3;
		}
	}

    // Copy image into input tensor
    memcpy(data, fimage.data, sizeof(float) * nW * nW * nC);
    return 0;
}


static std::vector<Box2f> generate_priorbox(int num_priores)
{
    const int conv_ws[5] = {69, 35, 18, 9, 5};
    const int conv_hs[5] = {69, 35, 18, 9, 5};

    const float aspect_ratios[3] = {1.f, 0.5f, 2.f};
    const float scales[5] = {24.f, 48.f, 96.f, 192.f, 384.f};

    int index = 0;
    std::vector<Box2f> priorboxes(num_priores);

    for (uint32_t p = 0; p < 5; p++)
	{
		int conv_w = conv_ws[p];
		int conv_h = conv_hs[p];

		float scale = scales[p];

		for (int i = 0; i < conv_h; i++)
		{
			for (int j = 0; j < conv_w; j++)
			{
				// +0.5 because priors are in center-size notation
				float cx = (j + 0.5f) / conv_w;
				float cy = (i + 0.5f) / conv_h;

				for (int k = 0; k < 3; k++)
				{
					float ar = aspect_ratios[k];

					ar = sqrt(ar);

					float w = scale * ar / nW;
					float h = scale / ar / nH;

					// This is for backward compatability with a bug where I made everything square by accident
					// cfg.backbone.use_square_anchors:
					h = w;

                    Box2f& priorbox = priorboxes[index];
                    priorbox.cx = cx;
                    priorbox.cy = cy;
                    priorbox.w = w;
                    priorbox.h = h;
                    index += 1;
				}
			}
		}
	}
    return priorboxes;
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void fast_nms(std::vector<std::vector<Object> >& class_candidates, std::vector<Object>& objects,
                     const float iou_thresh, const int nms_top_k, const int keep_top_k)
{
    for (int i = 0; i < (int)class_candidates.size(); i++)
    {
        std::vector<Object>& candidate = class_candidates[i];
        std::sort(candidate.begin(), candidate.end(), [](const Object& a, const Object& b) { return a.prob > b.prob; });
        if (candidate.size() == 0)
            continue;

        if (nms_top_k != 0 && nms_top_k > candidate.size())
        {
            candidate.erase(candidate.begin() + nms_top_k, candidate.end());
        }

        objects.push_back(candidate[0]);
        const int n = candidate.size();
        std::vector<float> areas(n);
        std::vector<int> keep(n);
        for (int j = 0; j < n; j++)
        {
            areas[j] = candidate[j].rect.area();
        }
        std::vector<std::vector<float> > iou_matrix;
        for (int j = 0; j < n; j++)
        {
            std::vector<float> iou_row(n);
            for (int k = 0; k < n; k++)
            {
                float inter_area = intersection_area(candidate[j], candidate[k]);
                float union_area = areas[j] + areas[k] - inter_area;
                iou_row[k] = inter_area / union_area;
            }
            iou_matrix.push_back(iou_row);
        }
        for (int j = 1; j < n; j++)
        {
            std::vector<float>::iterator max_value;
            max_value = std::max_element(iou_matrix[j].begin(), iou_matrix[j].begin() + j - 1);
            if (*max_value <= iou_thresh)
            {
                objects.push_back(candidate[j]);
            }
        }
    }
    std::sort(objects.begin(), objects.end(), [](const Object& a, const Object& b) { return a.prob > b.prob; });
    
    if (objects.size() > keep_top_k)
        objects.resize(keep_top_k);
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    cv::Mat image = bgr.clone();

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        if (obj.prob < 0.15)
            continue;

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob, obj.rect.x, obj.rect.y,
                obj.rect.width, obj.rect.height);

        const unsigned char* color = colors[color_index % 81];
        color_index++;

        cv::rectangle(image, obj.rect, cv::Scalar(color[0], color[1], color[2]));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0));

        // draw mask
        for (int y = 0; y < image.rows; y++)
        {
            const uchar* mp = obj.mask.ptr(y);
            uchar* p = image.ptr(y);
            for (int x = 0; x < image.cols; x++)
            {
                if (mp[x] == 255)
                {
                    p[0] = cv::saturate_cast<uchar>(p[0] * 0.5 + color[0] * 0.5);
                    p[1] = cv::saturate_cast<uchar>(p[1] * 0.5 + color[1] * 0.5);
                    p[2] = cv::saturate_cast<uchar>(p[2] * 0.5 + color[2] * 0.5);
                }
                p += 3;
            }
        }
    }

    cv::imwrite("yolact_out.jpg", image);
}

static int image_post_process(const char* image_path, armnn::OutputTensors& outputTensors)
{
    cv::Mat bgr = cv::imread(image_path);
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    float_t* mask_maps = reinterpret_cast<float*>(outputTensors.at(0).second.GetMemoryArea());
    float_t* location = reinterpret_cast<float*>(outputTensors.at(1).second.GetMemoryArea());
    float_t* mask = reinterpret_cast<float*>(outputTensors.at(2).second.GetMemoryArea());
    float_t* confidence = reinterpret_cast<float*>(outputTensors.at(3).second.GetMemoryArea());

    int num_class = 81;
    int num_priors = 19248;
    std::vector<Box2f> priorboxes = generate_priorbox(num_priors);

    std::vector<std::vector<Object>> class_candidates;
    class_candidates.resize(num_class);

    for (int i = 0; i < num_priors; i++)
    {
        const float* conf = confidence + i * 81;
        const float* loc = location + i * 4;
        const float* maskdata = mask + i * 32;
        Box2f& priorbox = priorboxes[i];

        int label = 0;
        float score = 0.f;
        for (int j = 1; j < num_class; j++)
        {
            float class_score = conf[j];
            if (class_score > score)
            {
                label = j;
                score = class_score;
            }
        }

        if (label == 0 || score <= confidence_thresh)
            continue;

        float var[4] = {0.1f, 0.1f, 0.2f, 0.2f};

        float bbox_cx = var[0] * loc[0] * priorbox.w + priorbox.cx;
        float bbox_cy = var[1] * loc[1] * priorbox.h + priorbox.cy;
        float bbox_w = (float)(exp(var[2] * loc[2]) * priorbox.w);
        float bbox_h = (float)(exp(var[3] * loc[3]) * priorbox.h);

        float obj_x1 = bbox_cx - bbox_w * 0.5f;
        float obj_y1 = bbox_cy - bbox_h * 0.5f;
        float obj_x2 = bbox_cx + bbox_w * 0.5f;
        float obj_y2 = bbox_cy + bbox_h * 0.5f;

        //clip
        obj_x1 = std::max(std::min(obj_x1 * bgr.cols, (float)(bgr.cols - 1)), 0.f);
        obj_y1 = std::max(std::min(obj_y1 * bgr.rows, (float)(bgr.rows - 1)), 0.f);
        obj_x2 = std::max(std::min(obj_x2 * bgr.cols, (float)(bgr.cols - 1)), 0.f);
        obj_y2 = std::max(std::min(obj_y2 * bgr.rows, (float)(bgr.rows - 1)), 0.f);

        Object obj;
        obj.rect = cv::Rect_<float>(obj_x1, obj_y1, obj_x2 - obj_x1 + 1, obj_y2 - obj_y1 + 1);
        obj.label = label;
        obj.prob = score;

        obj.maskdata = std::vector<float>(maskdata, maskdata + 32);

        class_candidates[label].push_back(obj);
    }

    std::vector<Object> darw_objects;
    darw_objects.clear();
    fast_nms(class_candidates, darw_objects, nms_thresh, 0, keep_top_k);

    for (int i = 0; i < darw_objects.size(); i++)
    {
        Object& obj = darw_objects[i];

        cv::Mat mask1(138, 138, CV_32FC1);
        {
            mask1 = cv::Scalar(0.f);

            for (int p = 0; p < 32; p++)
            {
                const float* maskmap = mask_maps + p;
                float coeff = obj.maskdata[p];
                float* mp = (float*)mask1.data;

                // mask += m * coeff
                for (int j = 0; j < 138 * 138; j++)
                {
                    mp[j] += maskmap[j * 32] * coeff;
                }
            }
        }

        cv::Mat mask2;
        cv::resize(mask1, mask2, cv::Size(img_w, img_h));

        // crop obj box and binarize
        obj.mask = cv::Mat(img_h, img_w, CV_8UC1);
        {
            obj.mask = cv::Scalar(0);

            for (int y = 0; y < img_h; y++)
            {
                if (y < obj.rect.y || y > obj.rect.y + obj.rect.height)
                    continue;

                const float* mp2 = mask2.ptr<const float>(y);
                uchar* bmp = obj.mask.ptr<uchar>(y);

                for (int x = 0; x < img_w; x++)
                {
                    if (x < obj.rect.x || x > obj.rect.x + obj.rect.width)
                        continue;

                    bmp[x] = mp2[x] > 0.5f ? 255 : 0;
                }
            }
        }
    }

    draw_objects(bgr, darw_objects);
    
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

    network->PrintGraph();

    // Optimize the network
    if (preferred_backends_order.size() == 0)
    {
        preferred_backends_order = default_preferred_backends_order;
    }
    armnn::IOptimizedNetworkPtr optimizedNet = armnn::Optimize(*network,
                                                               preferred_backends_order,
                                                               runtime->GetDeviceSpec());
    //network->PrintGraph();
    optimizedNet->PrintGraph();
    
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
    
    image_pre_process(input_file_str.c_str(), in[0].data());

    // Run the inferences
    std::cout << "\ninferences are running: " << std::flush;
    for (int i = 0 ; i < nb_loops ; i++)
    {
        runtime->EnqueueWorkload(networkId, inputTensors, outputTensors);

        std::cout << "# " << std::flush;
    }
    std::cout << "\n";


    for(uint32_t i = 0; i < nb_ouputs; ++i )
    {
        std::cout << "Output tensor" << i << " shape:" <<  outputTensors.at(i).second.GetShape() << std::endl;
    }
    
    image_post_process(input_file_str.c_str(), outputTensors);

    return 0;
}
