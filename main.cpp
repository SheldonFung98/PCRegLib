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
    pcreglib.load_pcd(sourceFile, targetFile);

    return 0;
}
