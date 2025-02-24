#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "clip.cuh"
#include "status.h"

template<typename Tdata>
__global__ void clip(
    Tdata const *x,
    Tdata *y,
    uint64_t data_size,
    bool has_lower_bound,
    bool has_upper_bound,
    float lower_bound,
    float upper_bound,
    uint64_t offset) {
    const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx >= data_size) return;

    Tdata value = x[idx];
    if constexpr (std::is_same<Tdata, half>::value) {
        value = (has_lower_bound && __hlt(value, __float2half(lower_bound))) ? __float2half(lower_bound) : value;
        value = (has_upper_bound && __hgt(value, __float2half(upper_bound))) ? __float2half(upper_bound) : value;
        y[idx] = value;
    } else {
        value = (has_lower_bound && value < lower_bound) ? lower_bound : value;
        value = (has_upper_bound && value > upper_bound) ? upper_bound : value;
        y[idx] = value;
    }
}

template<typename Tdata>
infiniopStatus_t clip_nv_gpu(
    ClipCudaDescriptor_t desc,
    void const *x,
    void *y,
    cudaStream_t stream) {
    const uint64_t data_size = desc->data_size;
    const uint64_t max_grid_size = desc->max_grid_size;

    auto x_ = reinterpret_cast<const Tdata *>(x);
    auto y_ = reinterpret_cast<Tdata *>(y);

    dim3 blockDims(std::min(static_cast<uint64_t>(256), data_size));
    dim3 gridDims(std::min(ROUND_UP_DIV(data_size, blockDims.x), max_grid_size));
    uint64_t step = gridDims.x * blockDims.x;

#pragma unroll
    for (uint64_t offset = 0; offset < data_size; offset += step) {
        clip<<<gridDims, blockDims, 0, stream>>>(
            x_, y_, data_size, desc->has_lower_bound, desc->has_upper_bound, desc->lower_bound, desc->upper_bound, offset);
    }

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaClip(ClipCudaDescriptor_t desc, void *y, const void *x, void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    if (desc->dtype == F16) {
        return clip_nv_gpu<half>(desc, reinterpret_cast<const half *>(x), reinterpret_cast<half *>(y), reinterpret_cast<cudaStream_t>(stream));
    }
    if (desc->dtype == F32) {
        return clip_nv_gpu<float>(desc, reinterpret_cast<const float *>(x), reinterpret_cast<float *>(y), reinterpret_cast<cudaStream_t>(stream));
    }
    return STATUS_SUCCESS;
}