#include "sigmoid.h"


#define CLIP(X, L) \
    ((2.0f * (L)) * __saturatef((0.5f / (L)) * (X + (L))) - L)


static const size_t THREAD_COUNT = 1024;


template <typename T>
__global__ void forwardKernel(const T* in, T* out, const size_t length) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
#if __CUDA_ARCH__ >= 350
        const T x = __ldg(in + i);
#else
        const T x = in[i];
#endif
        const T y = CLIP(0.1f * x, 0.05f);
        const T z = CLIP(0.05f * x, 0.125f);
        out[i] = __saturatef(0.05f * x + y + z + 0.5f);
    }
}


template <typename T>
__global__ void backwardKernel(const T* in, const T* grad, T* out, const size_t length) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
#if __CUDA_ARCH__ >= 350
        const T x = fabsf(__ldg(in + i));
        const T g = __ldg(grad + i);
#else
        const T x = fabsf(in[i]);
        const T g = grad[i];
#endif
        if (x <= 0.5f)
            out[i] = 0.2f * g;
        else if (x <= 2.5f)
            out[i] = 0.1f * g;
        else if (x <= 6.5f)
            out[i] = 0.05f * g;
        else
            out[i] = 0;
    }
}


namespace kernels {
    void sigmoidLikeForwardPass(cudaStream_t stream, const float* input, float* output, size_t length) {
        forwardKernel<float><<<(length + THREAD_COUNT - 1) / THREAD_COUNT, THREAD_COUNT, 0, stream>>>(input, output, length);
    }

    void sigmoidLikeBackwardPass(cudaStream_t stream, const float* input, const float* gradient, float* output, size_t length) {
        backwardKernel<float><<<(length + THREAD_COUNT - 1) / THREAD_COUNT, THREAD_COUNT, 0, stream>>>(input, gradient, output, length);
    }
}