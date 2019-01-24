# TensorFlow CMake/C++ Collection

Looking at the official docs: What do you see? The usual fare?
Now, guess what: This is a bazel-free zone. We use CMake here!

This collection contains **reliable** and **dead-simple** examples to use TensorFlow in C, C++, Go and Python: load a pre-trained model or compile a custom operation with or without CUDA. All builds are tested against the two most recent stable TensorFlow version and rely on CMake with a custom [FindTensorFlow.cmake](https://github.com/PatWie/tensorflow-cmake/blob/master/cmake/modules/FindTensorFlow.cmake). This cmake includes common work arounds for bugs in specific TF versions.

The implementation is tested against the following versions

| TensorFlow  v1.9.0 | TensorFlow  v1.10.0 | TensorFlow  v1.11.0 | TensorFlow  v1.12.0 |
| ------ | ------ | ------ | ------ |
| [![Build Status TensorFlow](https://ci.patwie.com/api/badges/PatWie/tensorflow-cmake/status.svg)](http://ci.patwie.com/PatWie/tensorflow-cmake) | [![Build Status TensorFlow](https://ci.patwie.com/api/badges/PatWie/tensorflow-cmake/status.svg)](http://ci.patwie.com/PatWie/tensorflow-cmake) | [![Build Status TensorFlow](https://ci.patwie.com/api/badges/PatWie/tensorflow-cmake/status.svg)](http://ci.patwie.com/PatWie/tensorflow-cmake) | [![Build Status TensorFlow](https://ci.patwie.com/api/badges/PatWie/tensorflow-cmake/status.svg)](http://ci.patwie.com/PatWie/tensorflow-cmake) |

It contains the following examples.

| Example| Explanation |
| ------ | ------ |
| [custom operation](./custom_op)   | build a custom operation for TensorFLow in C++/CUDA (requires only pip) |
| [inference  (C++)](./inference/cc) | run inference in C++ |
| [inference  (C)](./inference/c) | run inference in C |
| [inference  (Go)](./inference/go) | run inference in Go |
| [event writer](./examples/event_writer)  | write event files for TensorBoard in C++ |
| [keras cpp-inference example](./examples/keras)  | run a Keras-model in C++ |
| [simple example](./examples/simple)  | create and run a TensorFlow graph in C++ |
| [resize image example](./examples/resize)  | resize an image in TensorFlow with/without OpenCV |


## Custom Operation

This example illustrates the process of creating a custom operation using C++/CUDA and CMake. It is *not* intended to show an implementation obtaining peak-performance. Instead, it is just a boilerplate-template.

```console
user@host $ pip install tensorflow-gpu --user # solely the pip package is needed
user@host $ cd custom_op/user_ops
user@host $ cmake .
user@host $ make
user@host $ python test_matrix_add.py
user@host $ cd ..
user@host $ python example.py
```

## TensorFlow Graph within C++

This example illustrates the process of loading an image (using OpenCV or TensorFlow), resizing the image  saving the image as a JPG or PNG (using OpenCV or TensorFlow).

```console
user@host $ cd examples/resize
user@host $ export TENSORFLOW_BUILD_DIR=...
user@host $ export TENSORFLOW_SOURCE_DIR=...
user@host $ cmake .
user@host $ make
```


## TensorFlow-Serving

There are two examples demonstrating the handling of TensorFlow-Serving: using a vector input and using an encoded image input.

```console
server@host $ CHOOSE=basic # or image
server@host $ cd serving/${CHOOSE}/training
server@host $ python create.py # create some model
server@host $ cd serving/server/
server@host $ ./run.sh # start server

# some some queries

client@host $ cd client/bash
client@host $ ./client.sh
client@host $ cd client/python
# for the basic-example
client@host $ python client_rest.py
client@host $ python client_grpc.py
# for the image-example
client@host $ python client_rest.py /path/to/img.[png,jpg]
client@host $ python client_grpc.py /path/to/img.[png,jpg]
```

## Inference

Create a model in Python, save the graph to disk and load it in C/C+/Go/Python to perform inference. As these examples are based on the TensorFlow C-API they require the `libtensorflow_cc.so` library which is *not* shipped in the pip-package (tensorfow-gpu). Hence, you will need to build TensorFlow from source beforehand, e.g.,

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

user@host $ export TENSORFLOW_BUILD_DIR=/tensorflow_dist
user@host $ mkdir ${TENSORFLOW_BUILD_DIR}
user@host $ cp ${TENSORFLOW_SOURCE_DIR}/bazel-bin/tensorflow/*.so ${TENSORFLOW_BUILD_DIR}/
user@host $ cp ${TENSORFLOW_SOURCE_DIR}/bazel-genfiles/tensorflow/cc/ops/*.h ${TENSORFLOW_BUILD_DIR}/includes/tensorflow/cc/ops/
```

### 1. Save Model

We just run a very basic model

```python
x = tf.placeholder(tf.float32, shape=[1, 2], name='input')
output = tf.identity(tf.layers.dense(x, 1), name='output')
```

Therefore, save the model like you regularly do. This is done in `example.py` besides some outputs

```console
user@host $ python example.py

[<tf.Variable 'dense/kernel:0' shape=(2, 1) dtype=float32_ref>, <tf.Variable 'dense/bias:0' shape=(1,) dtype=float32_ref>]
input            [[1. 1.]]
output           [[2.1909506]]
dense/kernel:0   [[0.9070684]
 [1.2838823]]
dense/bias:0     [0.]
```

### 2. Run Inference

#### Python

```console
user@host $ python python/inference.py

[<tf.Variable 'dense/kernel:0' shape=(2, 1) dtype=float32_ref>, <tf.Variable 'dense/bias:0' shape=(1,) dtype=float32_ref>]
input            [[1. 1.]]
output           [[2.1909506]]
dense/kernel:0   [[0.9070684]
 [1.2838823]]
dense/bias:0     [0.]
```

#### C++

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

#### C

```console
user@host $ cd c
user@host $ cmake .
user@host $ make
user@host $ cd ..
user@host $ ./c/inference_c

2.190951

```


#### Go

```console
user@host $ go get github.com/tensorflow/tensorflow/tensorflow/go
user@host $ cd go
user@host $ ./build.sh
user@host $ cd ../
user@host $ ./inference_go

input           [[1 1]]
output          [[2.1909506]]
dense/kernel:0  [[0.9070684] [1.2838823]]
dense/bias:0    [0]
```
