// ComputerGraphics Tuebingen, 2018

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <stdio.h>

#include "matrix_add_op.h"

namespace tensorflow {

// Forward-Pass (CPU, GPU)
// --------------------------------------------------
template<typename Device, typename Dtype>
class MatrixAddOp: public OpKernel {
 public:
  explicit MatrixAddOp(OpKernelConstruction* ctx) :
    OpKernel(ctx) {
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("bias", &bias_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& mA = ctx->input(0);
    const Tensor& mB = ctx->input(1);

    OP_REQUIRES(ctx, mA.shape() == mB.shape(), errors::InvalidArgument("Input shapes have to be the same"));

    const int B = mA.dim_size(0);
    const int M = mA.dim_size(1);
    const int N = mA.dim_size(2);
    const int D = mA.dim_size(3);

    TensorShape output_shape({B, M, N, D});
    // same as: output_shape.AddDim(B); ....

    Tensor* mC = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &mC));
    // same as "OP_REQUIRES_OK(ctx,ctx->allocate_output(0, mA.tensor<Dtype, 4>().shape(), &mC));"

    ::tensorflow::functor::MatrixAddFunctor<Device, Dtype>()(ctx,
        mA, mB, mC, bias_);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MatrixAddOp);
  float bias_;
};

// Backward-Pass (CPU, GPU)
// --------------------------------------------------
template<typename Device, typename Dtype>
class MatrixAddGradOp: public OpKernel {
 public:
  explicit MatrixAddGradOp(OpKernelConstruction* ctx) :
    OpKernel(ctx) {
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& mA = ctx->input(0);
    const Tensor& mB = ctx->input(1);
    const Tensor& topdiff = ctx->input(2);

    Tensor* grad_mA = nullptr;
    Tensor* grad_mB = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, mA.shape(), &grad_mA));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, mB.shape(), &grad_mB));

    ::tensorflow::functor::MatrixAddGrad<Device, Dtype>()(ctx,
        topdiff, grad_mA, grad_mB);
  }
};


#define OPNAME(NAME) NAME ## Op
#define REGISTER(NAME, Dtype)                                          \
  REGISTER_KERNEL_BUILDER(                                             \
      Name(#NAME).Device(DEVICE_CPU).TypeConstraint<Dtype>("T"),       \
      OPNAME(NAME)<CPUDevice, Dtype>);                                 \
  REGISTER_KERNEL_BUILDER(                                             \
      Name(#NAME).Device(DEVICE_GPU).TypeConstraint<Dtype>("T"),       \
      OPNAME(NAME)<GPUDevice, Dtype>);


REGISTER(MatrixAdd, int);
REGISTER(MatrixAdd, float);
REGISTER(MatrixAdd, double);
REGISTER(MatrixAddGrad, float);
REGISTER(MatrixAddGrad, double);



}  // namespace tensorflow
