#ifndef PCREG_LIB_H
#define PCREG_LIB_H

#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <Eigen/Eigen>

#include "utils/grid_subsampling/grid_subsampling.h"


class ONNXModel{
private:
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::Session* session;
    std::vector<const char*> input_node_names;
    std::vector<Ort::AllocatedStringPtr> input_names_ptr;
    std::vector<const char*> output_node_names;
    std::vector<Ort::AllocatedStringPtr> output_names_ptr;
    std::vector<std::vector<int64_t>> input_node_dims;
    std::vector<std::vector<int64_t>> output_node_dims;
    std::string _model_path;

public:
    ONNXModel(std::string model_path){
        _model_path = model_path;
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session = new Ort::Session(env, _model_path.c_str(), session_options);

        // print number of model input nodes
        const size_t num_input_nodes = session->GetInputCount();
        input_names_ptr.reserve(num_input_nodes);
        input_node_names.reserve(num_input_nodes);

        // iterate over all input nodes
        for (size_t i = 0; i < num_input_nodes; i++) {
            // print input node names
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_node_names.push_back(input_name.get());
            output_names_ptr.push_back(std::move(input_name));

            // print input node types
            auto type_info = session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            // print input shapes/dims
            input_node_dims.push_back(tensor_info.GetShape());
        }

        // do the same for output nodes
        const size_t num_output_nodes = session->GetOutputCount();
        output_names_ptr.reserve(num_output_nodes);
        output_node_names.reserve(num_output_nodes);

        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            output_node_names.push_back(output_name.get());
            output_names_ptr.push_back(std::move(output_name));

            auto type_info = session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            output_node_dims.push_back(tensor_info.GetShape());
        }
    }
    void summarize(){
        std::cout << "model: " << _model_path << std::endl;
        std::cout << "Number of inputs = " << input_names_ptr.size() << std::endl;
        for(size_t i = 0; i < input_names_ptr.size(); i++){
            std::cout << "Input " << i << ": " << input_names_ptr[i].get();
            std::cout << " shape: ";
            for(size_t j = 0; j < input_node_dims[i].size(); j++){
                std::cout << input_node_dims[i][j];
                if (j < input_node_dims[i].size() - 1) std::cout << "x";
            }
            std::cout << std::endl;
        }
        std::cout << "Number of outputs = " << output_names_ptr.size() << std::endl;
        for(size_t i = 0; i < output_names_ptr.size(); i++){
            std::cout << "Output " << i << ": " << output_names_ptr[i].get();
            std::cout << " shape: ";
            for(size_t j = 0; j < output_node_dims[i].size(); j++){
                std::cout << output_node_dims[i][j];
                if (j < output_node_dims[i].size() - 1) std::cout << "x";
            }
            std::cout << std::endl;
        }
        std::cout << "##########################################" << std::endl;

    }
    void run(){
        // Ort::Value input_tensor(allocator, input_node_dims[0], input_node_dims[0].size());
        // Ort::Value output_tensor(allocator, output_node_dims[0], output_node_dims[0].size());
        // Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        // float* input_tensor_data = input_tensor.GetTensorMutableData<float>();
        // float* output_tensor_data = output_tensor.GetTensorMutableData<float>();
        // for (size_t i = 0; i < input_node_dims[0][1]; i++) {
        //     input_tensor_data[i] = i;
        // }
        // Ort::RunOptions run_options;
        // session->Run(run_options, input_names_ptr.data(), &input_tensor, 1, output_names_ptr.data(), &output_tensor, 1);
        // for (size_t i = 0; i < output_node_dims[0][1]; i++) {
        //     std::cout << output_tensor_data[i] << std::endl;
        // }
    }
    ~ONNXModel(){
        delete session;
    }
};

struct PCRegIO{

};

class PCRegLib {
private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr source, target;
    ONNXModel* backbone = nullptr;
    ONNXModel* transformer = nullptr;
    ONNXModel* matching = nullptr;

private:
    int _load_pcd(std::string src, std::string tgt);
    int _batch_grid_subsampling_kpconv(
        vector<PointXYZ>& pc, 
        vector<PointXYZ>& pc_sub, 
        vector<int>& original_batches,
        vector<int>& subsampled_batches,
        float sampledl
    );
    // int _hierarchical_downsample_pcd(
    //     pcl::PointCloud<pcl::PointXYZ>::Ptr& src,
    //     pcl::PointCloud<pcl::PointXYZ>::Ptr& tgt,
    // );
public:
    PCRegLib();
    void init();
    ~PCRegLib();
    int run_registration(
        std::string src, 
        std::string tgt, 
        Eigen::Matrix4f& res
    );
};

#endif