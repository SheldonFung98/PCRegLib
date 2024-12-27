cmake_minimum_required(VERSION 3.16.3)

project(main)

find_package(PCL 1.7 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    utils/
    /media/sf_sharefolder/onnxruntime-linux-x64-gpu-1.15.0/include
)
add_definitions(${PCL_DEFINITIONS})

add_executable(main 
    main.cpp
    pcreglib.cpp
    utils/grid_subsampling/grid_subsampling.cpp
    utils/cloud/cloud.cpp
)
target_link_libraries(main
    /media/sf_sharefolder/onnxruntime-linux-x64-gpu-1.15.0/lib/libonnxruntime.so
    /media/sf_sharefolder/onnxruntime-linux-x64-gpu-1.15.0/lib/libonnxruntime_providers_cuda.so
    /media/sf_sharefolder/onnxruntime-linux-x64-gpu-1.15.0/lib/libonnxruntime_providers_shared.so
    /media/sf_sharefolder/onnxruntime-linux-x64-gpu-1.15.0/lib/libonnxruntime_providers_tensorrt.so
    ${PCL_LIBRARIES}
)