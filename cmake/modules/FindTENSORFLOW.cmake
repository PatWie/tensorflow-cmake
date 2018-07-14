# Patrick Wieschollek, <mail@patwie.com>
# FindTENSORFLOW
# -------------
#
# Find TensorFlow library and includes
#
# This module will set the following variables in your project:
#
# ``TENSORFLOW_VERSION``
#   exact TensorFlow version obtained from runtime
# ``TENSORFLOW_ABI``
#   ABI specification of TensorFlow library
# ``TENSORFLOW_INCLUDE_DIR``
#   where to find tensorflow header files
# ``TENSORFLOW_LIBRARY``
#   the libraries to link against to use TENSORFLOW.
# ``TENSORFLOW_FOUND TRUE``
#   If false, do not try to use TENSORFLOW.
#
#
# USAGE
# ------
# add "list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}../../path/to/this/file)" to your project
#
# "add_tensorflow_operation" is a macro to compile a custom operation
#
# add_tensorflow_operation("<op-name>") expects the following files to exists:
#   - kernels/<op-name>_kernel.cc
#   - kernels/<op-name>_kernel.cu
#   - kernels/<op-name>_op.cc
#   - kernels/<op-name>_op.h
#   - ops/<op-name>.cc


message(STATUS "Detecting TensorFlow info")
execute_process(
  COMMAND python -c "import tensorflow as tf; print(tf.__version__); print(tf.__cxx11_abi_flag__); print(tf.sysconfig.get_include()); print(tf.sysconfig.get_lib() + '/libtensorflow_framework.so')"
  OUTPUT_VARIABLE TF_INFORMATION_STRING
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Detecting TensorFlow info - done")

string(REPLACE "\n" ";" TF_INFORMATION_LIST ${TF_INFORMATION_STRING})
list(GET TF_INFORMATION_LIST 0 TF_VERSION)
list(GET TF_INFORMATION_LIST 1 TF_ABI)
list(GET TF_INFORMATION_LIST 2 TF_INCLUDE_DIR)
list(GET TF_INFORMATION_LIST 3 TF_LIBRARY)

set(_packageName "TF")
if (DEFINED TF_VERSION)
    string (REGEX MATCHALL "[0-9]+" _versionComponents "${TF_VERSION}")
    list (LENGTH _versionComponents _len)
    if (${_len} GREATER 0)
        list(GET _versionComponents 0 TF_VERSION_MAJOR)
    endif()
    if (${_len} GREATER 1)
        list(GET _versionComponents 1 TF_VERSION_MINOR)
    endif()
    if (${_len} GREATER 2)
        list(GET _versionComponents 2 TF_VERSION_PATCH)
    endif()
    if (${_len} GREATER 3)
        list(GET _versionComponents 3 TF_VERSION_TWEAK)
    endif()
    set (TF_VERSION_COUNT ${_len})
else()
    set (TF_VERSION_COUNT 0)
endif()

if("${TF_VERSION_MAJOR}.${TF_VERSION_MINOR}" STREQUAL "${TENSORFLOW_FIND_VERSION_MAJOR}.${TENSORFLOW_FIND_VERSION_MINOR}")
  set(TENSORFLOW_VERSION ${TF_VERSION})
  set(TENSORFLOW_ABI ${TF_ABI})
  set(TENSORFLOW_INCLUDE_DIR ${TF_INCLUDE_DIR})
  set(TENSORFLOW_LIBRARY ${TF_LIBRARY})
  set(TENSORFLOW_FOUND TRUE)
  message(STATUS "Found TensorFlow: (found suitable exact version \"${TENSORFLOW_VERSION}\")")
  message(STATUS "TensorFlow-ABI is ${TENSORFLOW_ABI}")
  message(STATUS "TensorFlow-INCLUDE_DIR is ${TENSORFLOW_INCLUDE_DIR}")
  message(STATUS "TensorFlow-LIBRARY is ${TENSORFLOW_LIBRARY}")
else()
  set(TENSORFLOW_FOUND FALSE)

endif()

macro(add_tensorflow_operation op_name)
  message(STATUS "will build custom TensorFlow operation \"${op_name}\"")

  cuda_add_library(${op_name}_op_cu SHARED kernels/${op_name}_kernel.cu)
  set_target_properties(${op_name}_op_cu PROPERTIES PREFIX "")

  add_library(${op_name}_op SHARED kernels/${op_name}_op.cc kernels/${op_name}_kernel.cc ops/${op_name}.cc )

  set_target_properties(${op_name}_op PROPERTIES PREFIX "")
  target_link_libraries(${op_name}_op LINK_PUBLIC ${op_name}_op_cu ${TENSORFLOW_LIBRARY})
endmacro()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  TENSORFLOW
  FOUND_VAR TENSORFLOW_FOUND
  REQUIRED_VARS
    TENSORFLOW_LIBRARY
    TENSORFLOW_INCLUDE_DIR
  VERSION_VAR
    TENSORFLOW_VERSION
  )