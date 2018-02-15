cmake_minimum_required( VERSION 2.8 )


if(DEFINED ENV{TensorFlow_GIT_REPO})
  message(STATUS "FOUND TENSORFLOW GIT-REPO in $ENV{TensorFlow_GIT_REPO}")
else()
    message(FATAL_ERROR "TensorFlow_GIT_REPO must be defined, see README.md")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tensorflow_config.txt")
  include(${CMAKE_CURRENT_SOURCE_DIR}/tensorflow_config.txt)

  message(STATUS "TensorFlow_ABI: ${TensorFlow_ABI}")
  message(STATUS "TensorFlow_INCLUDE_DIRS: ${TensorFlow_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "You need to run \"python configure.py\" first")
endif()

project( TFCustomOp )

include_directories(SYSTEM $ENV{TensorFlow_GIT_REPO}/bazel-tensorflow)

add_executable (inference_c inference_c.c)
TARGET_LINK_LIBRARIES(inference_c -Wl,--allow-multiple-definition -Wl,--whole-archive "$ENV{TensorFlow_GIT_REPO}/bazel-bin/tensorflow/libtensorflow.so" -Wl,--no-whole-archive)
TARGET_LINK_LIBRARIES(inference_c -Wl,--allow-multiple-definition -Wl,--whole-archive "$ENV{TensorFlow_GIT_REPO}/bazel-bin/tensorflow/libtensorflow_framework.so" -Wl,--no-whole-archive)
