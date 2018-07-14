# University Tuebingen, 2018
cmake_minimum_required( VERSION 2.8 )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}../../cmake/modules)

project(TFCustomOp)

find_package(TENSORFLOW 1.9 EXACT REQUIRED)
find_package(CUDA 9.0 EXACT REQUIRED)

message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_SAMPLE_INC: ${CUDA_SAMPLE_INC}")

# set necessary flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -march=native -fopenmp -D_GLIBCXX_USE_CXX11_ABI=${TENSORFLOW_ABI}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -fPIC --shared -D_GLIBCXX_USE_CXX11_ABI=${TENSORFLOW_ABI}" )
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11  --expt-relaxed-constexpr -D GOOGLE_CUDA=1 --gpu-architecture=sm_52 -D_GLIBCXX_USE_CXX11_ABI=${TENSORFLOW_ABI}" )

include_directories(SYSTEM "${CUDA_INCLUDE_DIRS}/../../")
include_directories(SYSTEM ${TENSORFLOW_INCLUDE_DIR})
include_directories(SYSTEM "kernels")

add_tensorflow_operation("matrix_add")
