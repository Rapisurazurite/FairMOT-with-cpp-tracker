cmake_minimum_required(VERSION 3.16)
project(tracklet)

find_package(CUDA REQUIRED)


set(PYTHON_EXECUTABLE /home/ubuntu/anaconda3/envs/Openpcdet/bin/python3.8)
include_directories("/home/ubuntu/anaconda3/envs/Openpcdet/lib/python3.8/site-packages/torch/include")
include_directories("/home/ubuntu/anaconda3/envs/Openpcdet/lib/python3.8/site-packages/torch/include/torch/csrc/api/include/")
include_directories("/usr/include/eigen3")


add_subdirectory(extern/pybind11)


## PROJECT pybind_cuda
set(module_name tracklet)
set(pybind_cuda_SRCS main.cpp src/cpp_extension.cpp src/KalmanFilter.cpp src/basetrack.cpp src/Ndarray.cpp)

pybind11_add_module(${module_name} ${pybind_cuda_SRCS})
add_custom_target(
        pytorch_build_cu
        COMMAND chmod a+x ${CMAKE_SOURCE_DIR}/cmake_setup.sh
        COMMAND ${CMAKE_SOURCE_DIR}/cmake_setup.sh ${CMAKE_SOURCE_DIR} build CUDA --module_name ${module_name}_cu --src ${pybind_cuda_SRCS}
)

add_custom_target(
        pytorch_build
        COMMAND chmod a+x ${CMAKE_SOURCE_DIR}/cmake_setup.sh
        COMMAND ${CMAKE_SOURCE_DIR}/cmake_setup.sh ${CMAKE_SOURCE_DIR} install --module_name ${module_name} --src ${pybind_cuda_SRCS}
)