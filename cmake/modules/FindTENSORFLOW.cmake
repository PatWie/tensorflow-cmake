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
# ``TENSORFLOW_SOURCE_DIR``
#   Path to source of TensorFlow, when env-var 'TENSORFLOW_SOURCE_DIR' is set and path exists
# ``TENSORFLOW_C_LIBRARY``
#   Path to libtensorflow_cc.so (require env-var 'TENSORFLOW_BUILD_DIR')
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

# export TENSORFLOW_SOURCE_DIR=/graphics/opt/opt_Ubuntu16.04/tensorflow/src
# detect TensorFlow git repository
set(TENSORFLOW_HAS_SOURCE FALSE)
set(TENSORFLOW_SOURCE_DIR "")
if(EXISTS "$ENV{TENSORFLOW_SOURCE_DIR}/README.md")
  if(EXISTS "$ENV{TENSORFLOW_SOURCE_DIR}/WORKSPACE")
    set(TENSORFLOW_HAS_SOURCE TRUE)
  endif()
endif()

if(TENSORFLOW_HAS_SOURCE)
  set(TENSORFLOW_SOURCE_DIR $ENV{TENSORFLOW_SOURCE_DIR})
  message(STATUS "TensorFlow-SOURCE-DIRECTORY is ${TENSORFLOW_SOURCE_DIR}")
else()
  message(STATUS "No TensorFlow source repository detected")
endif()

# detect libtensorflow_cc.so
# export TENSORFLOW_BUILD_DIR=/graphics/opt/opt_Ubuntu16.04/tensorflow/build/v1.9.0/
find_library(TENSORFLOW_C_LIBRARY
  NAMES libtensorflow_cc.so
  PATHS $ENV{TENSORFLOW_BUILD_DIR}
  DOC "TensorFlow CC library." )

if(TENSORFLOW_C_LIBRARY)
  message(STATUS "TensorFlow-CC-LIBRARY is ${TENSORFLOW_C_LIBRARY}")
else()
  message(STATUS "No TensorFlow-CC-LIBRARY detected")
endif()

macro(TENSORFLOW_REQUIRE_C_LIBRARY)
  if(TENSORFLOW_C_LIBRARY)
  else()
    message(FATAL_ERROR "Project requires libtensorflow_cc.so, please specify the path in ENV-VAR 'TENSORFLOW_BUILD_DIR'")
  endif()
endmacro()

macro(TENSORFLOW_REQUIRE_SOURCE)
  if(TENSORFLOW_HAS_SOURCE)
  else()
    message(FATAL_ERROR "Project requires TensorFlow source directory, please specify the path in ENV-VAR 'TENSORFLOW_SOURCE_DIR'")
  endif()
endmacro()

macro(add_tensorflow_operation op_name)
  message(STATUS "will build custom TensorFlow operation \"${op_name}\"")

  cuda_add_library(${op_name}_op_cu SHARED kernels/${op_name}_kernel.cu)
  set_target_properties(${op_name}_op_cu PROPERTIES PREFIX "")

  add_library(${op_name}_op SHARED kernels/${op_name}_op.cc kernels/${op_name}_kernel.cc ops/${op_name}.cc )

  set_target_properties(${op_name}_op PROPERTIES PREFIX "")
  target_link_libraries(${op_name}_op LINK_PUBLIC ${op_name}_op_cu ${TENSORFLOW_LIBRARY})
endmacro()

# simplify TensorFlow dependencies
add_library(TENSORFLOW_DEP INTERFACE)
TARGET_INCLUDE_DIRECTORIES(TENSORFLOW_DEP INTERFACE ${TENSORFLOW_SOURCE_DIR})
TARGET_INCLUDE_DIRECTORIES(TENSORFLOW_DEP INTERFACE ${TENSORFLOW_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(TENSORFLOW_DEP INTERFACE -Wl,--allow-multiple-definition -Wl,--whole-archive ${TENSORFLOW_C_LIBRARY} -Wl,--no-whole-archive)
TARGET_LINK_LIBRARIES(TENSORFLOW_DEP INTERFACE -Wl,--allow-multiple-definition -Wl,--whole-archive ${TENSORFLOW_LIBRARY} -Wl,--no-whole-archive)


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