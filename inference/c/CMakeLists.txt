cmake_minimum_required( VERSION 2.8 )

project( TFCustomOp )

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake/modules)
find_package(TensorFlow REQUIRED)

TensorFlow_REQUIRE_C_LIBRARY()
TensorFlow_REQUIRE_SOURCE()

ADD_EXECUTABLE (inference_c inference_c.c)
# no need for ${TENSORFLOW_INCLUDE_DIR} from pip
TARGET_INCLUDE_DIRECTORIES(inference_c PRIVATE ${TensorFlow_SOURCE_DIR})
TARGET_LINK_LIBRARIES(inference_c PRIVATE TensorFlow_DEP)
