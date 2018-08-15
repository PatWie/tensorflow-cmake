// 2018, Patrick Wieschollek <mail@patwie.com>

#ifndef MATRIX_ADD_KERNELS_MATRIX_ADD_OP_H_
#define MATRIX_ADD_KERNELS_MATRIX_ADD_OP_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Device, typename Dtype>
struct MatrixAddFunctor {
  static void launch(::tensorflow::OpKernelContext* ctx, const Tensor& mA_,
                     const Tensor& mB_, Tensor* mC_, Dtype bias);
};

template <typename Device, typename Dtype>
struct MatrixAddGrad {
  static void launch(::tensorflow::OpKernelContext* ctx, const Tensor& topdiff_,
                     Tensor* grad_mA_, Tensor* grad_mB_);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // MATRIX_ADD_KERNELS_MATRIX_ADD_OP_H_
