// 2018, Patrick Wieschollek <mail@patwie.com>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "matrix_add_op.h"

namespace tensorflow {

// Forward-Pass (CPU, GPU)
// --------------------------------------------------
template <typename Device, typename Dtype>
class MatrixAddOp : public OpKernel {
 public:
  explicit MatrixAddOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bias", &bias_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& X = ctx->input(0);
    const Tensor& Y = ctx->input(1);

    if (!ctx->status().ok()) {
      return;
    }

    OP_REQUIRES(ctx, X.shape() == Y.shape(),
                errors::InvalidArgument("Input shapes have to be the same"));

    const int N = X.dim_size(0);
    const int H = X.dim_size(1);
    const int W = X.dim_size(2);
    const int C = X.dim_size(3);

    TensorShape output_shape({N, H, W, C});
    // same as: output_shape.AddDim(N); ....

    Tensor* Z = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &Z));
    // same as "OP_REQUIRES_OK(ctx,ctx->allocate_output(0, X.tensor<Dtype,
    // 4>().shape(), &Z));"

    ::tensorflow::functor::MatrixAddFunctor<Device, Dtype>::launch(ctx, X, Y, Z,
                                                                   bias_);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MatrixAddOp);
  float bias_;
};

// Backward-Pass (CPU, GPU)
// --------------------------------------------------
template <typename Device, typename Dtype>
class MatrixAddGradOp : public OpKernel {
 public:
  explicit MatrixAddGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& X = ctx->input(0);
    const Tensor& Y = ctx->input(1);
    const Tensor& topdiff = ctx->input(2);

    if (!ctx->status().ok()) {
      return;
    }

    Tensor* grad_X = nullptr;
    Tensor* grad_Y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, X.shape(), &grad_X));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, Y.shape(), &grad_Y));

    ::tensorflow::functor::MatrixAddGrad<Device, Dtype>::launch(ctx, topdiff,
                                                                grad_X, grad_Y);
  }
};

#define REGISTER_CUSTOM_OP(NAME, DEVICE, T)                       \
  REGISTER_KERNEL_BUILDER(                                        \
      Name(#NAME).Device(DEVICE_##DEVICE).TypeConstraint<T>("T"), \
      NAME##Op<DEVICE##Device, T>)

REGISTER_CUSTOM_OP(MatrixAdd, CPU, uint32);
REGISTER_CUSTOM_OP(MatrixAdd, CPU, int32);
REGISTER_CUSTOM_OP(MatrixAdd, CPU, float);
REGISTER_CUSTOM_OP(MatrixAdd, CPU, double);
REGISTER_CUSTOM_OP(MatrixAddGrad, CPU, float);
REGISTER_CUSTOM_OP(MatrixAddGrad, CPU, double);

#ifdef GOOGLE_CUDA
REGISTER_CUSTOM_OP(MatrixAdd, GPU, float);
REGISTER_CUSTOM_OP(MatrixAdd, GPU, double);
REGISTER_CUSTOM_OP(MatrixAddGrad, GPU, float);
REGISTER_CUSTOM_OP(MatrixAddGrad, GPU, double);
#endif  // GOOGLE_CUDA
#undef REGISTER_CUSTOM_OP

// // Register the CPU kernels.
// #define REGISTER_ATRIXADD_OP_CPU(T)                               \
//   REGISTER_KERNEL_BUILDER(                                         \
//       Name("MatrixAdd").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
//       MatrixAddOp<CPUDevice, T>)

// #define REGISTER_ATRIXADD_GRAD_OP_CPU(T)                              \
//   REGISTER_KERNEL_BUILDER(                                             \
//       Name("MatrixAddGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
//       MatrixAddGradOp<CPUDevice, T>)

// // requires "tensorflow/core/framework/register_types.h"
// TF_CALL_uint32(REGISTER_ATRIXADD_OP_CPU);
// TF_CALL_int32(REGISTER_ATRIXADD_OP_CPU);
// TF_CALL_double(REGISTER_ATRIXADD_OP_CPU);
// TF_CALL_float(REGISTER_ATRIXADD_OP_CPU);

// TF_CALL_double(REGISTER_ATRIXADD_GRAD_OP_CPU);
// TF_CALL_float(REGISTER_ATRIXADD_GRAD_OP_CPU);
// #undef REGISTER_ATRIXADD_OP_CPU
// #undef REGISTER_ATRIXADD_GRAD_OP_CPU

// // Register the GPU kernels.
// #ifdef GOOGLE_CUDA
// #define REGISTER_ATRIXADD_OP_GPU(T)                                   \
//   REGISTER_KERNEL_BUILDER(                                             \
//       Name("MatrixAdd").Device(DEVICE_GPU).TypeConstraint<T>("T"),     \
//       MatrixAddOp<GPUDevice, T>)                                       \
//   REGISTER_KERNEL_BUILDER(                                             \
//       Name("MatrixAddGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
//       MatrixAddGradOp<GPUDevice, T>)

// TF_CALL_GPU_NUBER_TYPES_NO_HALF(REGISTER_ATRIXADD_OP_GPU);
// #undef REGISTER_ATRIXADD_OP_GPU
// #endif  // GOOGLE_CUDA

}  // namespace tensorflow
