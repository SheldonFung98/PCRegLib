cmake_minimum_required(VERSION 3.16.3)

project(main)

include_directories(
    /media/sf_sharefolder/onnxruntime-linux-x64-gpu-1.15.0/include
)

add_executable(main 
    main.cpp
    pcreglib.cpp
)

target_link_libraries(main
    /media/sf_sharefolder/onnxruntime-linux-x64-gpu-1.15.0/lib/libonnxruntime.so
    /media/sf_sharefolder/onnxruntime-linux-x64-gpu-1.15.0/lib/libonnxruntime_providers_cuda.so
    /media/sf_sharefolder/onnxruntime-linux-x64-gpu-1.15.0/lib/libonnxruntime_providers_shared.so
    /media/sf_sharefolder/onnxruntime-linux-x64-gpu-1.15.0/lib/libonnxruntime_providers_tensorrt.so
)