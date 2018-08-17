// 2018, Patrick Wieschollek <mail@patwie.com>

#include <cstring>

#include "matrix_add_op.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {
namespace functor {

template <typename Dtype>
struct MatrixAddFunctor<CPUDevice, Dtype> {
  static void launch(::tensorflow::OpKernelContext* ctx, const Tensor& Xt,
                     const Tensor& Yt, Tensor* Zt, Dtype bias) {
    const auto X = Xt.tensor<Dtype, 4>();
    const auto Y = Yt.tensor<Dtype, 4>();
    auto Z = Zt->tensor<Dtype, 4>();

    Z.setZero();

    // get dimensions
    const int N = Xt.dim_size(0);
    const int H = Xt.dim_size(1);
    const int W = Xt.dim_size(2);
    const int C = Xt.dim_size(3);

    // the computation (easy to read)
    for (int n = 0; n < N; ++n)
      for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
          for (int c = 0; c < C; ++c)
            Z(n, h, w, c) = X(n, h, w, c) + Y(n, h, w, c) + bias;
  }
};

template struct MatrixAddFunctor<CPUDevice, int32>;
template struct MatrixAddFunctor<CPUDevice, uint32>;
template struct MatrixAddFunctor<CPUDevice, float>;
template struct MatrixAddFunctor<CPUDevice, double>;

template <typename Dtype>
struct MatrixAddGrad<CPUDevice, Dtype> {
  static void launch(::tensorflow::OpKernelContext* ctx, const Tensor& topdiff_,
                     Tensor* grad_mA_, Tensor* grad_mB_) {
    const int W = topdiff_.NumElements();

    grad_mA_->flat<Dtype>().setZero();
    grad_mB_->flat<Dtype>().setZero();

    const Dtype* topdiff = topdiff_.flat<Dtype>().data();
    Dtype* grad_X = grad_mA_->flat<Dtype>().data();
    Dtype* grad_Y = grad_mB_->flat<Dtype>().data();

    std::memcpy(grad_X, topdiff, W * sizeof(Dtype));
    std::memcpy(grad_Y, topdiff, W * sizeof(Dtype));
    // for (int i = 0; i < W; ++i) {
    //   grad_X[i] = topdiff[i];
    //   grad_Y[i] = topdiff[i];
    // }
  }
};

template struct MatrixAddGrad<CPUDevice, float>;
template struct MatrixAddGrad<CPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow