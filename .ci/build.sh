# contains README.md of TensorFlow
export TENSORFLOW_SOURCE_DIR=/tensorflow
# contains libtensorflow_cc.so
export TENSORFLOW_C_LIBRARY=/tensorflow/bazel-bin/tensorflow/
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}


cd inference/cc/
cmake .
make

cd ../c/
cmake .
make

cd ../examples/keras/
cmake .
make