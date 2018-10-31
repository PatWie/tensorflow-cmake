ARG BAZEL_VERSION
ARG CUDA_VERSION
ARG CUDA_POSTFIX
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu16.04

ARG BAZEL_VERSION
ARG CUDA_POSTFIX
ARG CUDNN_VERSION
RUN echo "BAZEL_VERSION" ${BAZEL_VERSION}
RUN echo "CUDA_VERSION" ${CUDA_VERSION}
RUN echo "CUDA_POSTFIX" ${CUDA_POSTFIX}
RUN echo "CUDNN_VERSION" ${CUDNN_VERSION}
LABEL maintainer="Patrick Wieschollek <mail@patwie.com>"


RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA_POSTFIX} \
        cuda-cublas-dev-${CUDA_POSTFIX} \
        cuda-cudart-dev-${CUDA_POSTFIX} \
        cuda-cufft-dev-${CUDA_POSTFIX} \
        cuda-curand-dev-${CUDA_POSTFIX} \
        cuda-cusolver-dev-${CUDA_POSTFIX} \
        cuda-cusparse-dev-${CUDA_POSTFIX} \
        curl \
        git \
        nano \
        cmake \
        libcudnn7=${CUDNN_VERSION} \
        libcudnn7-dev=${CUDNN_VERSION} \
        libnccl2 \
        libnccl-dev \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        wget \
        && \
    find /usr/local/cuda-9.0/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a
    # rm -rf /var/lib/apt/lists/* && \

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        matplotlib \
        mock \
        numpy \
        scipy \
        sklearn \
        pandas \
        keras_applications \
        keras_preprocessing \
        && \
    python -m ipykernel.kernelspec

# Set up Bazel.

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc

# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc


WORKDIR /
RUN echo https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh

