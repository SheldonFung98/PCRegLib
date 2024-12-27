#include "pcreglib.h"

PCRegLib::PCRegLib() : source(new pcl::PointCloud<pcl::PointXYZ>), target(new pcl::PointCloud<pcl::PointXYZ>) {
}

void PCRegLib::init() {
    std::cout << "Loading models" << std::endl << std::endl;
    backbone = new ONNXModel("models/backbone.onnx");
    backbone->summarize();
    transformer = new ONNXModel("models/transformer.onnx");
    transformer->summarize();
    matching = new ONNXModel("models/matching.onnx");
    matching->summarize();
    std::cout << "PCRegLib initialized" << std::endl;
}

int PCRegLib::_load_pcd(std::string src, std::string tgt) {
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(src, *source) == -1){
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return -1;
    } else {
          std::cout << "Loaded "
            << source->width * source->height
            << " data points from test_pcd.pcd with the following fields: "
            << std::endl;
    }
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(tgt, *target) == -1){
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return -1;
    } else {
          std::cout << "Loaded "
            << target->width * target->height
            << " data points from test_pcd.pcd with the following fields: "
            << std::endl;
    }
    return 0;
}

int PCRegLib::_batch_grid_subsampling_kpconv(vector<PointXYZ>& pc, vector<PointXYZ>& pc_sub, vector<int>& original_batches, vector<int>& subsampled_batches, float sampledl){
	vector<float> original_features, subsampled_features;
	vector<int> original_classes, subsampled_classes;
    int max_p = 0;

	batch_grid_subsampling(
        pc,
		pc_sub,
        original_features,
        subsampled_features,
        original_classes,
        subsampled_classes,
        original_batches,
        subsampled_batches,
        sampledl,
        max_p
    );
}

PCRegLib::~PCRegLib() {
    delete backbone;
    delete transformer;
    delete matching;
    std::cout << "PCRegLib destroyed" << std::endl;
}

int PCRegLib::run_registration(std::string src, std::string tgt, Eigen::Matrix4f& res){
    if (_load_pcd(src, tgt) == -1) return -1;
    std::cout << "Running registration" << std::endl;
    return 0;
}
