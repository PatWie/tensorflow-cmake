export TensorFlow_GIT_REPO=/tensorflow
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

cd inference/cc/
cp ../../.ci/tensorflow_config.example.txt tensorflow_config.txt
cmake .
make

cd ../c/
cp ../../.ci/tensorflow_config.example.txt tensorflow_config.txt
cmake .
make

cd ../examples/keras/
cp ../../.ci/tensorflow_config.example.txt tensorflow_config.txt
cmake .
make