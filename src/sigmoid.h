#pragma once
#include <cstddef>
#include <cuda_runtime.h>

namespace kernels {
    void sigmoidLikeForwardPass(cudaStream_t stream, const float* input, float* output, size_t length);
    void sigmoidLikeBackwardPass(cudaStream_t stream, const float* input, const float* gradient, float* output, size_t length);
}