ARG CUDA_VERSION
ARG TF_VERSION
ARG BAZEL_VERSION
ARG CUB_VERSION
FROM tensorflow:ubuntu16.04-cuda${CUDA_VERSION}-bazel${BAZEL_VERSION}-dependencies
LABEL maintainer="Patrick Wieschollek <mail@patwie.com>"

ARG TF_VERSION
ARG BAZEL_VERSION
ARG CUB_VERSION

# Download and build TensorFlow.
WORKDIR /tensorflow
RUN git clone --branch=v${TF_VERSION} --depth=1 https://github.com/tensorflow/tensorflow.git .

# Configure the build for our CUDA configuration.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV TF_NEED_CUDA 1
ENV TF_CUDA_COMPUTE_CAPABILITIES=3.0,3.5,5.2,6.0,6.1
ENV TF_CUDA_VERSION=${CUDA_VERSION}
ENV TF_NCCL_VERSION=1.3
ENV TF_CUDNN_VERSION=7

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

RUN LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    tensorflow/tools/ci_build/builds/configured GPU \
    bazel build -c opt --copt=-mavx --config=cuda \
       --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" \
       tensorflow/tools/pip_package:build_pip_package && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip

RUN LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    tensorflow/tools/ci_build/builds/configured GPU \
    bazel build -c opt --copt=-mavx --config=cuda \
       --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" \
       //tensorflow:libtensorflow_cc.so

RUN LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    tensorflow/tools/ci_build/builds/configured GPU \
    bazel build -c opt --copt=-mavx --config=cuda \
       --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" \
       //tensorflow:libtensorflow.so


RUN pip --no-cache-dir install --upgrade /tmp/pip/tensorflow-*.whl
#RUN rm /usr/local/cuda/lib64/stubs/libcuda.so.1
#RUN rm -rf /tmp/pip
#RUN rm -rf /root/.cache

RUN mkdir /tensorflow_dist
RUN mkdir -p /tensorflow_dist/includes/tensorflow/cc/ops/
RUN cp /tensorflow/bazel-genfiles/tensorflow/cc/ops/*.h /tensorflow_dist/includes/tensorflow/cc/ops/
RUN cp /tensorflow/bazel-bin/tensorflow/*.so /tensorflow_dist/

WORKDIR /extra
RUN wget https://codeload.github.com/NVlabs/cub/zip/${CUB_VERSION}
RUN unzip ${CUB_VERSION}

WORKDIR /root

