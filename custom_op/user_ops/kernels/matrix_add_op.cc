// 2018, Patrick Wieschollek <mail@patwie.com>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

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

    if (!ctx->status().ok()) {
      return;
    }

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

    ::tensorflow::functor::MatrixAddFunctor<Device, Dtype>::launch(ctx,
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

    if (!ctx->status().ok()) {
      return;
    }

    Tensor* grad_mA = nullptr;
    Tensor* grad_mB = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, mA.shape(), &grad_mA));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, mB.shape(), &grad_mB));

    ::tensorflow::functor::MatrixAddGrad<Device, Dtype>::launch(ctx,
        topdiff, grad_mA, grad_mB);
  }
};


// Register the CPU kernels.
#define REGISTER_MATRIXADD_OP_CPU(T)                                     \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("MatrixAdd").Device(DEVICE_CPU).TypeConstraint<T>("T"),       \
      MatrixAddOp<CPUDevice, T>)

#define REGISTER_MATRIXADD_GRAD_OP_CPU(T)                                \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("MatrixAddGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),   \
      MatrixAddGradOp<CPUDevice, T>)

// see "tensorflow/core/framework/register_types.h"
TF_CALL_uint32(REGISTER_MATRIXADD_OP_CPU);
TF_CALL_int32(REGISTER_MATRIXADD_OP_CPU);
TF_CALL_double(REGISTER_MATRIXADD_OP_CPU);
TF_CALL_float(REGISTER_MATRIXADD_OP_CPU);

TF_CALL_double(REGISTER_MATRIXADD_GRAD_OP_CPU);
TF_CALL_float(REGISTER_MATRIXADD_GRAD_OP_CPU);
#undef REGISTER_MATRIXADD_OP_CPU
#undef REGISTER_MATRIXADD_GRAD_OP_CPU

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_MATRIXADD_OP_GPU(T)                                     \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("MatrixAdd").Device(DEVICE_GPU).TypeConstraint<T>("T"),       \
      MatrixAddOp<GPUDevice, T>)                                         \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("MatrixAddGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"),   \
      MatrixAddGradOp<GPUDevice, T>)

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_MATRIXADD_OP_GPU);
#undef REGISTER_MATRIXADD_OP_GPU
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
