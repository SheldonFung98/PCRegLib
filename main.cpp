#include <iostream>
#include "pcreglib.h"

int main(int argc, char const *argv[]){
    
    if (argc < 3) {
        std::cerr << "Usage: " 
            << argv[0] 
            << " <source> <target>" 
            << std::endl;
        return 1;
    }

    std::string sourceFile = argv[1];
    std::string targetFile = argv[2];

    PCRegLib pcreglib;
    pcreglib.init();
    Eigen::Matrix4f transformation_pred;
    pcreglib.run_registration(sourceFile, targetFile, transformation_pred);
    return 0;
}
