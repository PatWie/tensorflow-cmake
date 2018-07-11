# University Tuebingen, 2018
cmake_minimum_required( VERSION 2.8 )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tensorflow_config.txt")
  include(${CMAKE_CURRENT_SOURCE_DIR}/tensorflow_config.txt)

  message(STATUS "use TensorFlow_ABI: ${TensorFlow_ABI}")
  message(STATUS "use TensorFlow_INCLUDE_DIRS: ${TensorFlow_INCLUDE_DIRS}")
  message(STATUS "use TensorFlow_LIBRARIES: ${TensorFlow_LIBRARIES}")
else()
  message(FATAL_ERROR "You need to run \"python configure.py\" first")
endif()


project(TFCustomOp)

# use cuda 9.0
find_package(CUDA 9.0 EXACT REQUIRED)
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_SAMPLE_INC: ${CUDA_SAMPLE_INC}")

# set necessary flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -march=native -fopenmp -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -fPIC --shared -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}" )
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11  --expt-relaxed-constexpr -D GOOGLE_CUDA=1 --gpu-architecture=sm_52 -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}" )

include_directories(SYSTEM "${CUDA_INCLUDE_DIRS}/../../")
include_directories(SYSTEM ${TensorFlow_INCLUDE_DIRS})
include_directories(SYSTEM "kernels")

macro(add_tf_operation arg)
  message(STATUS "will build \"${arg}\" operation")

  cuda_add_library(${arg}_op_cu SHARED kernels/${arg}_kernel.cu)
  set_target_properties(${arg}_op_cu PROPERTIES PREFIX "")

  add_library(${arg}_op SHARED kernels/${arg}_op.cc kernels/${arg}_kernel.cc ops/${arg}.cc )

  set_target_properties(${arg}_op PROPERTIES PREFIX "")
  target_link_libraries(${arg}_op LINK_PUBLIC ${arg}_op_cu ${TensorFlow_LIBRARIES})
endmacro()

add_tf_operation("matrix_add")
