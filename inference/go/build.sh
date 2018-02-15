export LD_LIBRARY_PATH=${TensorFlow_GIT_REPO}/bazel-bin/tensorflow/:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${TensorFlow_GIT_REPO}/bazel-bin/tensorflow:${LIBRARY_PATH}
go build inference_go.go