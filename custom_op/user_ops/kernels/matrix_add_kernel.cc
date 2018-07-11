// ComputerGraphics Tuebingen, 2017

#include <cstring>

#include "tensorflow/core/framework/op.h"
#include "matrix_add_op.h"

namespace tensorflow {

namespace functor {

template <typename Dtype>
struct MatrixAddFunctor<CPUDevice, Dtype> {
  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const Tensor& mA_,
                   const Tensor& mB_,
                   Tensor *mC_,
                   Dtype bias) {
    auto mA = mA_.tensor<Dtype, 4>();
    auto mB = mB_.tensor<Dtype, 4>();
    auto mC = mC_->tensor<Dtype, 4>();

    mC.setZero();

    // get dimensions
    const int B = mA_.dim_size(0);
    const int M = mA_.dim_size(1);
    const int N = mA_.dim_size(2);
    const int D = mA_.dim_size(3);

    // the computation (easy to read)
    for (int b = 0; b < B; ++b)
      for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
          for (int d = 0; d < D; ++d)
            mC(b, r, c, d) = mA(b, r, c, d) + mB(b, r, c, d) + bias;
  }
};

template struct MatrixAddFunctor<CPUDevice, int>;
template struct MatrixAddFunctor<CPUDevice, float>;
template struct MatrixAddFunctor<CPUDevice, double>;


template <typename Dtype>
struct MatrixAddGrad<CPUDevice, Dtype> {
  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const Tensor& topdiff_,
                   Tensor *grad_mA_,
                   Tensor *grad_mB_) {
    const int N = topdiff_.NumElements();

    grad_mA_->flat<Dtype>().setZero();
    grad_mB_->flat<Dtype>().setZero();

    const Dtype* topdiff = topdiff_.flat<Dtype>().data();
    Dtype* grad_mA = grad_mA_->flat<Dtype>().data();
    Dtype* grad_mB = grad_mB_->flat<Dtype>().data();

    std::memcpy(grad_mA, topdiff, N * sizeof(Dtype));
    std::memcpy(grad_mB, topdiff, N * sizeof(Dtype));
    // for (int i = 0; i < N; ++i) {
    //   grad_mA[i] = topdiff[i];
    //   grad_mB[i] = topdiff[i];
    // }
  }
};

template struct MatrixAddGrad<CPUDevice, int>;
template struct MatrixAddGrad<CPUDevice, float>;
template struct MatrixAddGrad<CPUDevice, double>;


} // namespace functor
} // namespace tensorflow