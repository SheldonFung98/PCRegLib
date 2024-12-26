#ifndef PCREG_LIB_H
#define PCREG_LIB_H

#include <iostream>
#include <onnxruntime_cxx_api.h>

class ONNXModel{
private:
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::Session* session;
    std::vector<Ort::AllocatedStringPtr> input_names_ptr;
    std::vector<const char*> input_node_names;
    std::vector<std::vector<int64_t>> input_node_dims;
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
            input_names_ptr.push_back(std::move(input_name));

            // print input node types
            auto type_info = session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            // print input shapes/dims
            input_node_dims.push_back(tensor_info.GetShape());
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
    }
    ~ONNXModel(){
        delete session;
    }
};

class PCRegLib {
private:
    std::string source;
    std::string target;
    ONNXModel* backbone;

public:
    PCRegLib();
    void init();
    void load_pcd(std::string src, std::string tgt);
    ~PCRegLib();
};

#endif