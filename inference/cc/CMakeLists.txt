cmake_minimum_required( VERSION 2.8 )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- BEGIN: get TensorFlow related informations (ABI, include-path, lib) --------
execute_process(
  COMMAND python -c "import tensorflow as tf; print(tf.__cxx11_abi_flag__); print(tf.sysconfig.get_include()); print(tf.sysconfig.get_lib() + '/libtensorflow_framework.so')"
  OUTPUT_VARIABLE TensorFlow_INFORMATION
  OUTPUT_STRIP_TRAILING_WHITESPACE)

string(REPLACE "\n" ";" TensorFlow_INFORMATION_LIST ${TensorFlow_INFORMATION})
list(GET TensorFlow_INFORMATION_LIST 0 TensorFlow_ABI)
list(GET TensorFlow_INFORMATION_LIST 1 TensorFlow_INCLUDE_DIRS)
list(GET TensorFlow_INFORMATION_LIST 2 TensorFlow_LIBRARIES)

message(STATUS "TensorFlow-ABI: ${TensorFlow_ABI}")
message(STATUS "TensorFlow-INCLUDE_DIRS: ${TensorFlow_INCLUDE_DIRS}")
message(STATUS "TensorFlow-LIBRARIES: ${TensorFlow_LIBRARIES}")
# --- END: get TensorFlow related informations (ABI, include-path, lib) --------

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}" )


project( TFCustomOp )

# include_directories(SYSTEM "${TensorFlow_INCLUDE_DIRS}/external/nsync/public")
include_directories(SYSTEM ${TensorFlow_INCLUDE_DIRS})

add_executable (inference_cc inference_cc.cc)
TARGET_LINK_LIBRARIES(inference_cc -Wl,--allow-multiple-definition -Wl,--whole-archive "$ENV{TensorFlow_GIT_REPO}/bazel-bin/tensorflow/libtensorflow_cc.so" -Wl,--no-whole-archive)
TARGET_LINK_LIBRARIES(inference_cc -Wl,--allow-multiple-definition -Wl,--whole-archive "$ENV{TensorFlow_GIT_REPO}/bazel-bin/tensorflow/libtensorflow_framework.so" -Wl,--no-whole-archive)
