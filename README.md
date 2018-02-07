# Running Inference in TensorFlow in C/C+/Go/Python

Just a dead-simple way to run saved models from tensorflow in different languages.

### Requiements

It assumes that you have install TensorFlow from source using

```console
  ./configure
  # ...
  bazel build -c opt --copt=-mfpmath=both --copt=-msse4.2 --config=cuda //tensorflow:libtensorflow.so
  bazel build -c opt --copt=-mfpmath=both --copt=-msse4.2 --config=cuda //tensorflow:libtensorflow_cc.so
```

Further, we ned to the path to TensorFlow git repository such that we can link to these libraries. There, set the path such that the output looks like

```console
user@host $ export TensorFlow_GIT_REPO=/path/to/tensorflow/git
user@host $ ls TensorFlow_GIT_REPO

ACKNOWLEDGMENTS     bazel-genfiles      configure          pip
ADOPTERS.md         bazel-out           configure.py       py.pynano
ANDROID_NDK_HOME    bazel-tensorflow    configure.py.bkp   README.md
...

```


### 1. Save Model

We just run the very basic model

```python
x = tf.placeholder(tf.float32, shape=[1, 2], name='input')
output = tf.identity(tf.layers.dense(x, 1), name='output')
```

Therefore, just save the model like you normally do, e.g.,

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

These bindings are tested on the [9d419e4511](https://github.com/tensorflow/tensorflow/commit/995d836e9ba7cbee56948f73bdbd099d419e4511) commit.

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
