
# 1.9
export CUDA_VERSION=9.0
export CUDA_POSTFIX=9-0
export TF_VERSION=1.9.0
export BAZEL_VERSION=0.11.0
export CUB_VERSION=1.8.0
export OPENCV_VERSION=3.4.2
export GO_VERSION=1.10
export CUDNN_VERSION=7.0.5.15-1+cuda${CUDA_VERSION}

# 1.10
export CUDA_VERSION=9.2
export CUDA_POSTFIX=9-2
export TF_VERSION=1.10.0
export BAZEL_VERSION=0.16.1
export CUB_VERSION=1.8.0
export OPENCV_VERSION=3.4.2
export GO_VERSION=1.10
export GO_VERSION=1.10
export CUDNN_VERSION=7.1.4.18-1+cuda${CUDA_VERSION}

# 1.11
export CUDA_VERSION=9.2
export CUDA_POSTFIX=9-2
export TF_VERSION=1.11.0
export BAZEL_VERSION=0.16.1
export CUB_VERSION=1.8.0
export OPENCV_VERSION=3.4.2
export GO_VERSION=1.11
export CUDNN_VERSION=7.0.5.15-1+cuda${CUDA_VERSION}


# tensorflow:ubuntu16.04-cuda9.2-bazel0.16.0-dependencies
# tensorflow:ubuntu16.04-cuda9.2-bazel0.16.0-tensorflow1.10.0
# tensorflow:ubuntu16.04-cuda9.2-bazel0.16.0-tensorflow1.10.0-opencv
# # tensorflow:ubuntu16.04-cuda9.2-v1.10.0-opencv
# # tensorflow:ubuntu16.04-cuda9.2-v1.10.0-go

sudo docker build \
  --build-arg BAZEL_VERSION=${BAZEL_VERSION} \
  --build-arg CUDA_VERSION=${CUDA_VERSION} \
  --build-arg CUDA_POSTFIX=${CUDA_POSTFIX} \
  --build-arg CUDNN_VERSION=${CUDNN_VERSION} \
  -t tensorflow:ubuntu16.04-cuda${CUDA_VERSION}-bazel${BAZEL_VERSION}-dependencies \
  -f Dockerfile.tensorflow-dependencies .

sudo docker build \
  --build-arg BAZEL_VERSION=${BAZEL_VERSION} \
  --build-arg CUDA_VERSION=${CUDA_VERSION} \
  --build-arg CUDA_POSTFIX=${CUDA_POSTFIX} \
  --build-arg CUDNN_VERSION=${CUDNN_VERSION} \
  --build-arg TF_VERSION=${TF_VERSION} \
  --build-arg CUB_VERSION=${CUB_VERSION} \
  -t tensorflow:ubuntu16.04-cuda${CUDA_VERSION}-bazel${BAZEL_VERSION}-tensorflow${TF_VERSION} \
  -f Dockerfile.tensorflow-plain .


sudo docker build \
  --build-arg BAZEL_VERSION=${BAZEL_VERSION} \
  --build-arg CUDA_VERSION=${CUDA_VERSION} \
  --build-arg CUDA_POSTFIX=${CUDA_POSTFIX} \
  --build-arg CUDNN_VERSION=${CUDNN_VERSION} \
  --build-arg TF_VERSION=${TF_VERSION} \
  --build-arg CUB_VERSION=${CUB_VERSION} \
  --build-arg OPENCV_VERSION=${OPENCV_VERSION} \
  -t tensorflow:ubuntu16.04-cuda${CUDA_VERSION}-bazel${BAZEL_VERSION}-tensorflow${TF_VERSION}-opencv${OPENCV_VERSION} \
  -f Dockerfile.tensorflow-opencv .


sudo docker build \
  --build-arg BAZEL_VERSION=${BAZEL_VERSION} \
  --build-arg CUDA_VERSION=${CUDA_VERSION} \
  --build-arg CUDA_POSTFIX=${CUDA_POSTFIX} \
  --build-arg CUDNN_VERSION=${CUDNN_VERSION} \
  --build-arg TF_VERSION=${TF_VERSION} \
  --build-arg CUB_VERSION=${CUB_VERSION} \
  --build-arg GO_VERSION=${GO_VERSION} \
  -t tensorflow:ubuntu16.04-cuda${CUDA_VERSION}-bazel${BAZEL_VERSION}-tensorflow${TF_VERSION}-go${GO_VERSION} \
  -f Dockerfile.tensorflow-go .













