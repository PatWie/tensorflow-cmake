ARG UBUNTU_VERSION=16.04

FROM nvidia/cuda:10.0-base-ubuntu${UBUNTU_VERSION} as base

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-10-0 \
        cuda-cublas-dev-10-0 \
        cuda-cudart-dev-10-0 \
        cuda-cufft-dev-10-0 \
        cuda-curand-dev-10-0 \
        cuda-cusolver-dev-10-0 \
        cuda-cusparse-dev-10-0 \
        libcudnn7=7.4.1.5-1+cuda10.0 \
        libcudnn7-dev=7.4.1.5-1+cuda10.0 \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        wget \
        git \
        && \
    find /usr/local/cuda-10.0/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a



# Configure the build for our CUDA configuration.
ENV CI_BUILD_PYTHON python
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV TF_NEED_CUDA 1
ENV TF_NEED_TENSORRT 1
ENV TF_CUDA_COMPUTE_CAPABILITIES=3.5,5.2,6.0,6.1,7.0
ENV TF_CUDA_VERSION=10.0
ENV TF_CUDNN_VERSION=7

# Check out TensorFlow source code if --build_arg CHECKOUT_TENSORFLOW=1
ARG CHECKOUT_TF_SRC=0
ARG CHECKOUT_TF_VERSION=r1.14

RUN test "${CHECKOUT_TF_SRC}" -eq 1 && git clone --branch=${CHECKOUT_TF_VERSION} --depth=1 https://github.com/tensorflow/tensorflow.git /tensorflow_src

ARG USE_PYTHON_3_NOT_2
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    openjdk-8-jdk \
    ${PYTHON}-dev \
    swig

RUN ${PIP} --no-cache-dir install \
    Pillow \
    h5py \
    keras_applications \
    keras_preprocessing \
    matplotlib \
    mock \
    numpy \
    scipy \
    sklearn \
    future \
    pandas \
    && test "${USE_PYTHON_3_NOT_2}" -eq 1 && true || ${PIP} --no-cache-dir install \
    enum34

# Install bazel
ARG BAZEL_VERSION=0.24.1
RUN mkdir /bazel && \
    wget -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    wget -O /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    chmod +x /bazel/installer.sh && \
    /bazel/installer.sh && \
    rm -f /bazel/installer.sh


# Install Tensorflow
ENV TF_NEED_TENSORRT 0
ENV TF_CUDA_COMPUTE_CAPABILITIES=5.2
WORKDIR /tensorflow_src
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


RUN mkdir /tensorflow_dist
RUN mkdir -p /tensorflow_dist/includes/tensorflow/cc/ops/
RUN cp /tensorflow_src/bazel-genfiles/tensorflow/cc/ops/*.h /tensorflow_dist/includes/tensorflow/cc/ops/
RUN cp /tensorflow_src/bazel-bin/tensorflow/*.so /tensorflow_dist/

# Load CUBS
ARG CUB_VERSION=1.8.0

WORKDIR /extra
RUN wget https://codeload.github.com/NVlabs/cub/zip/${CUB_VERSION}
RUN unzip ${CUB_VERSION}

WORKDIR /root


# Install OpenCV

ARG OPENCV_VERSION=3.4.1
WORKDIR /tmp
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O opencv3.zip
RUN unzip -q opencv3.zip
RUN mv opencv-${OPENCV_VERSION} /opencv

RUN mkdir -p /opencv/build
WORKDIR /opencv/build

RUN apt-get update && apt-get install -y \
    cmake

RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D BUILD_PYTHON_SUPPORT=OFF \
        -D WITH_CUDA=OFF \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D INSTALL_C_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        #-D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_NEW_PYTHON_SUPPORT=OFF \
        -D WITH_IPP=OFF \
        -D WITH_V4L=ON ..
RUN make -j
RUN make install


RUN ldconfig

WORKDIR /root


# Install go

ARG GO_VERSION=1.11
WORKDIR /tmp
RUN wget https://dl.google.com/go/go${GO_VERSION}.linux-amd64.tar.gz
RUN tar -xvf go${GO_VERSION}.linux-amd64.tar.gz
RUN mv go /usr/local
ENV GOROOT=/usr/local/go

WORKDIR /root


ENV GOPATH=/gocode
ENV PATH=$GOROOT/bin:$PATH
