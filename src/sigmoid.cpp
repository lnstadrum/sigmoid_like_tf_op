#define EIGEN_USE_GPU

#include "sigmoid.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

template<class Device, typename T>
class ForwardKernel : public OpKernel {
public:
    explicit ForwardKernel(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& inputTensor = context->input(0);
        auto input = inputTensor.flat<T>();

        // Create an output tensor
        Tensor* outputTensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, inputTensor.shape(), &outputTensor));
        auto output = outputTensor->flat<T>();

        // Compute the output
        kernels::sigmoidLikeForwardPass(context->eigen_device<Device>().stream(), input.data(), output.data(), input.size());
    }
};


template<class Device, typename T>
class BackwardKernel : public OpKernel {
public:
    explicit BackwardKernel(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor (x)
        const Tensor& inputTensor = context->input(0);
        auto input = inputTensor.flat<T>();

        // Grab the gradient (dL/dy)
        const Tensor& gradientTensor = context->input(1);
        auto gradient = gradientTensor.flat<T>();

        // Create an output tensor (dL/dx)
        Tensor* outputTensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, inputTensor.shape(), &outputTensor));
        auto output = outputTensor->flat<T>();

        // Compute the output
        kernels::sigmoidLikeBackwardPass(context->eigen_device<Device>().stream(), input.data(), gradient.data(), output.data(), input.size());
    }
};


// Register operations kernels
#define REGISTER_KERNEL(OP, CLASS, T) \
    REGISTER_KERNEL_BUILDER(Name(OP).Device(DEVICE_GPU).TypeConstraint<T>("T"), CLASS<Eigen::GpuDevice, T>)

REGISTER_KERNEL("SigmoidLike", ForwardKernel, float);
REGISTER_KERNEL("SigmoidLikeGradient", BackwardKernel, float);


// Register operations
REGISTER_OP("SigmoidLike")
    .Attr("T: {float}")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

REGISTER_OP("SigmoidLikeGradient")
    .Attr("T: {float}")
    .Input("input: T")
    .Input("gradient: T")
    .Output("output: T")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);