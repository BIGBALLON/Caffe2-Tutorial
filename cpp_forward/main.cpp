/*******************************************************
 * Copyright (C) 2018-2019 bigballon <fm.bigballon@gmail.com>
 * 
 * This file is a caffe2 C++ image classification test 
 * by using pre-trained cifar10 model.
 *
 * Feel free to modify if you need.
 *******************************************************/
#include "caffe2/core/common.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/workspace.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/init.h"

// headers for opencv [P.S]
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <iostream>
#include <map>

// define flags
CAFFE2_DEFINE_string(init_net, "./init_net.pb",
                     "The given path to the init protobuffer.");
CAFFE2_DEFINE_string(predict_net, "./predict_net.pb",
                     "The given path to the predict protobuffer.");
CAFFE2_DEFINE_string(file, "./image_file.jpg", "The image file.");

// define functons
void forwardTest();
void loadImage(std::string file_name, float* imgBinary);

// main function
int main(int argc, char** argv) {
    caffe2::GlobalInit(&argc, &argv);
    forwardTest();
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}

void loadImage(std::string file_name, float* imgBinary){

    auto image = cv::imread(file_name);    // CV_8UC3
    std::cout << "== image size: " << image.size()
              << " ==" << std::endl;

    // scale image to fit
    cv::Size scale(32,32);
    cv::resize(image, image, scale);
    std::cout << "== simply resize: " << image.size() 
              << " ==" << std::endl;
    
    // convert [unsigned int] to [float]
    image.convertTo(image, CV_32FC3);

    // convert NHWC to NCHW
    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);
    std::vector<float> data;
    for (auto &c : channels) {
        data.insert(data.end(), (float *)c.datastart, (float *)c.dataend);
    }
    
    // do normalization & copy to imgBinary
    int dim = 0;
    float image_mean[3] = {113.865, 122.95, 125.307};
    float image_std[3] = {66.7048, 62.0887, 62.9932};
        
    for(auto i = 0; i < data.size();++i){
        if(i > 0 && i % (32*32) == 0) dim++;
        imgBinary[i] = (data[i] - image_std[dim]) / image_mean[dim];
        // std::cout << imgBinary[i] << std::endl;
    }
}

void forwardTest(){

    // define a caffe2 Workspace
    caffe2::Workspace workSpace;

    // define initNet and predictNet
    caffe2::NetDef initNet, predictNet;

    // read protobuf
    CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_init_net, &initNet));
    CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_predict_net, &predictNet));

    // set device type, I just test GPU version
    predictNet.mutable_device_option()->set_device_type(caffe2::CUDA);
    initNet.mutable_device_option()->set_device_type(caffe2::CUDA);

    // load network
    CAFFE_ENFORCE(workSpace.RunNetOnce(initNet));
    CAFFE_ENFORCE(workSpace.CreateNet(predictNet));

    // load image from file, then convert it to float array.
    float imgBinary[3 * 32 * 32];
    loadImage(FLAGS_file, imgBinary);

    // define a Tensor which is used to stone input data
    caffe2::TensorCPU input;
    input.Resize(std::vector<caffe2::TIndex>({1, 3, 32, 32}));
    input.ShareExternalPointer(imgBinary);

    // get "data" blob
    auto data = workSpace.GetBlob("data")->GetMutable<caffe2::TensorCUDA>();
    // copy from input data
    data->CopyFrom(input);

    // forward
    workSpace.RunNet(predictNet.name());

    // get softmax blob and show the results
    std::vector<std::string> labelName = {"airplane","automobile","bird","cat","deer",
        "dog","frog","horse","ship","truck"};

    auto softmax = caffe2::TensorCPU(workSpace.GetBlob("softmax")->Get<caffe2::TensorCUDA>());
    
    std::vector<float> probs(softmax.data<float>(),
        softmax.data<float>() + softmax.size());
    
    auto max = std::max_element(probs.begin(), probs.end());
    auto index = std::distance(probs.begin(), max);
    std::cout << "== predicted label: " << labelName[index] 
              << " ==\n== with probability: " << (*max * 100)
              << "% ==" << std::endl;
}
