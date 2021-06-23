#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"
#include "tensorflow/lite/model.h"

// Load Label function
std::vector<std::string> load_labels(std::string labels_file){
    
    std::ifstream file(labels_file.c_str()); // Open the Label File
    if (!file.is_open()){
        std::cerr << "unable to open file" << std::endl;
        exit(-1);
    }
    std::string label_str;
    std::vector<std::string> labels;

    while (std::getline(file, label_str)) // Read all lines from file
    {
        if (label_str.size() > 0) labels.push_back(label_str);
    }   

    file.close(); // Close The File
    return labels;
}

int main(int argc, char** argv)
{
    // Load Model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("C:/Users/Administrator/Karthick/Git/tflite/models/classification/mobilenet_v1_1.0_224_quant.tflite");
    if (model == nullptr) {
        std::cerr << "failed to load model" << std::endl;
        exit(-1);
    }
    
    // Initiate Interpreter
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
    if (interpreter == nullptr) {
        std::cerr << "failed to initiate interpreter" << std::endl;
        exit(-1);
    } 
    interpreter->AllocateTensors();
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "failed to allocate tensor" << std::endl;
        exit(-1);
    }  
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(4); 
    
    // Get Input Tensor Dimensions
    int input = interpreter->inputs()[0];
    auto height = interpreter->tensor(input)->dims->data[1];
    auto width = interpreter->tensor(input)->dims->data[2];
    auto channels = interpreter->tensor(input)->dims->data[3];
    
    // Load Input Image
    cv::Mat image;
    auto frame = cv::imread("C:/Users/Administrator/Downloads/sample.jpg");
    if (frame.empty()) {
        std::cerr << "Can not load picture!" << std::endl;
        exit(-1);
    }

    // Copy image to input tensor
    cv::resize(frame, image, cv::Size(width, height), cv::INTER_NEAREST);
    memcpy(interpreter->typed_input_tensor<unsigned char>(0), image.data, image.total() * image.elemSize());

    // Inference
    std::chrono::steady_clock::time_point start, end;
    start = std::chrono::steady_clock::now();
    interpreter->Invoke();
    end = std::chrono::steady_clock::now();
    auto inference_time = std::chrono::duration_cast <std::chrono::milliseconds> (end - start).count();

    // Get Output
    int output = interpreter->outputs()[0];
    TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
    auto output_size = output_dims->data[output_dims->size - 1];
    std::vector<std::pair<float, int>> top_results;
    float threshold = 0.001f;

    switch (interpreter->tensor(output)->type) {
        case kTfLiteFloat32:
            tflite::label_image::get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size,
                3, threshold, &top_results, kTfLiteFloat32);
            break;
        case kTfLiteUInt8:
            tflite::label_image::get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0), output_size,
                3, threshold, &top_results, kTfLiteUInt8);
            break;
        default:
            std::cerr << "cannot handle output type " << interpreter->tensor(output)->type << std::endl;
            exit(-1);
    }


    // Print inference ms in input image
    int y0 = 10;
    int dy = 22;
    int i = 1;
    int y = y0 + i++ * dy;
    cv::putText(frame, "Inference Time in ms: " + std::to_string(inference_time), cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

    // Load Labels
    auto labels = load_labels("C:/Users/Administrator/Karthick/Git/tflite/models/classification/labels_mobilenet_quant_v1_224.txt");

    // Print labels with confidence in input image
    for (const auto& result : top_results) {
        const float confidence = result.first;
        const int index = result.second;
        std::string output = "Label : " + labels[index] + " Confidence : " + std::to_string(confidence);
        int y = y0 + i++ * dy;
        cv::putText(frame, output, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    }

    // Display image
    cv::imshow("Output", frame);
    cv::waitKey(0);
    
    return 0;
}