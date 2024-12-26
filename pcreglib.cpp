#include "pcreglib.h"


PCRegLib::PCRegLib() {
}

void PCRegLib::init() {
    backbone = new ONNXModel("models/backbone.onnx");
    backbone->summarize();
    std::cout << "PCRegLib initialized" << std::endl;
}

void PCRegLib::load_pcd(std::string src, std::string tgt) {
    source = src;
    target = tgt;
    std::cout << "Source: " << source << std::endl;
    std::cout << "Target: " << target << std::endl;
}

PCRegLib::~PCRegLib() {
}