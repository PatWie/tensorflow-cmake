# TensorFlow C++ Collection (TensorFlow >= v1.9.0)

[![Build Status TensorFlow v1.9 ](https://ci.patwie.com/api/badges/PatWie/tensorflow_inference/status.svg)](http://ci.patwie.com/PatWie/tensorflow_inference)
Just a dead-simple way to run saved models from tensorflow in different languages **without** messing around with bazel.

| Example | requires TensorFlow source | Explanation |
| ------ | ------ | ------ |
| [custom operation](./custom_ops) | no | build a custom op for TensorFLow in C++/CUDA
| [inference  (C++, C, Go)](./inference) | yes | running inference code using CMake in C/C++/Go/Python
| [keras cpp-inference example](./examples/keras) | yes | running inference code using CMake in C++ from a keras-model
| [simple example](./examples/simple) | yes | running the C++ example from TensorFlow code using CMake
| [OpenCV example](./examples/resize) | yes | running a C++ example using TensorFlow in combination with OpenCV to resize an image (uses CMake)

## Details

### Custom Operation

This example illustrates the process of creating a custom operation using C++ and CUDA. It is *not* intended to show an implemenation obtaining peak-performance. INstead it is just a boilerplate-template.

```console
user@host $ pip install tensorflow-gpu --user # just the pip package is needed
user@host $ cd custom_op/user_ops
user@host $ cmake .
user@host $ make
user@host $ python test_matrix_add.py
user@host $ cd ..
user@host $ python example.py
```
### Inference

This example creates a model in Python, saves the graph to disk and loads it in C/C+/Go/Python to perform inference. As they are based on the C-API they require the `libtensorflow_cc.so` library which is *not* shipped in the pip-package. Hence, you will need to build TensorFlow from source beforehand,  like

```console
user@host $ ls ${TENSORFLOW_SOURCE_DIR}

ACKNOWLEDGMENTS     bazel-genfiles      configure          pip
ADOPTERS.md         bazel-out           configure.py       py.pynano
ANDROID_NDK_HOME    bazel-tensorflow    configure.py.bkp   README.md
...
user@host $ cd ${TENSORFLOW_SOURCE_DIR}
user@host $  ./configure
user@host $  # ... or whatever options you used here
user@host $ bazel build -c opt --copt=-mfpmath=both --copt=-msse4.2 --config=cuda //tensorflow:libtensorflow.so
user@host $ bazel build -c opt --copt=-mfpmath=both --copt=-msse4.2 --config=cuda //tensorflow:libtensorflow_cc.so
user@host $ export TENSORFLOW_BUILD_DIR=bazel-bin/tensorflow/
```

#### 1. Save Model

We just run a very basic model

```python
x = tf.placeholder(tf.float32, shape=[1, 2], name='input')
output = tf.identity(tf.layers.dense(x, 1), name='output')
```

Therefore, just save the model like you normally do. This is done in `example.py` besides some outputs

```console
user@host $ python example.py

[<tf.Variable 'dense/kernel:0' shape=(2, 1) dtype=float32_ref>, <tf.Variable 'dense/bias:0' shape=(1,) dtype=float32_ref>]
input            [[1. 1.]]
output           [[2.1909506]]
dense/kernel:0   [[0.9070684]
 [1.2838823]]
dense/bias:0     [0.]
```

#### 2. Run Inference

These bindings are tested on the [9d419e4511 TensorFlow](https://github.com/tensorflow/tensorflow/commit/995d836e9ba7cbee56948f73bdbd099d419e4511) commit.

##### Python

```console
user@host $ python python/inference.py

[<tf.Variable 'dense/kernel:0' shape=(2, 1) dtype=float32_ref>, <tf.Variable 'dense/bias:0' shape=(1,) dtype=float32_ref>]
input            [[1. 1.]]
output           [[2.1909506]]
dense/kernel:0   [[0.9070684]
 [1.2838823]]
dense/bias:0     [0.]
```

##### C++

```console
user@host $ cd cc
user@host $ cmake .
user@host $ make
user@host $ cd ..
user@host $ ./cc/inference_cc

input           Tensor<type: float shape: [1,2] values: [1 1]>
output          Tensor<type: float shape: [1,1] values: [2.19095063]>
dense/kernel:0  Tensor<type: float shape: [2,1] values: [0.907068372][1.28388226]>
dense/bias:0    Tensor<type: float shape: [1] values: 0>
```

##### C

```console
user@host $ cd c
user@host $ cmake .
user@host $ make
user@host $ cd ..
user@host $ ./c/inference_c

2.190951

```


##### Go

```console
user@host $ export LIBRARY_PATH=${TensorFlow_GIT_REPO}/bazel-bin/tensorflow:$LIBRARY_PATH
user@host $ export LD_LIBRARY_PATH=${TensorFlow_GIT_REPO}/bazel-bin/tensorflow:$LD_LIBRARY_PATH
user@host $ go get github.com/tensorflow/tensorflow/tensorflow/go
user@host $ cd go
user@host $ go build inference_go.go
user@host $ cd ../
user@host $ ./inference_go

input           [[1 1]]
output          [[2.1909506]]
dense/kernel:0  [[0.9070684] [1.2838823]]
dense/bias:0    [0]
```