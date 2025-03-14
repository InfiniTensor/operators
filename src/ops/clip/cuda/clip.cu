#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "clip.cuh"

template<typename Tdata>
__global__ void clip(
    Tdata *y,
    const Tdata *x,
    const Tdata *max_val,
    const Tdata *min_val,
    uint64_t offset) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if constexpr (std::is_same<Tdata, half>::value) {
        y[idx] = __hlt(x[idx], *min_val) ? *min_val : __hgt(x[idx], *max_val) ? *max_val : x[idx];
    } else {
        y[idx] = x[idx] < *min_val ? *min_val : x[idx] > *max_val ? *max_val : x[idx];
    }
}

template<typename Tdata>
infiniopStatus_t clip_nv_gpu(ClipCudaDescriptor_t desc, void *y, void const *x, void *stream) {
    if (desc->data_size == 0) {
        return STATUS_SUCCESS;
    }
    dim3 blockDims = dim3(std::min(static_cast<uint64_t>(256), desc->data_size));
    dim3 gridDims = dim3(std::min(ROUND_UP_DIV(desc->data_size, blockDims.x), desc->max_grid_size));
    uint64_t step = gridDims.x * blockDims.x;

    const auto x_ = reinterpret_cast<Tdata const *>(x);
    const auto y_ = reinterpret_cast<Tdata *>(y);
    const Tdata *max_val = reinterpret_cast<const Tdata *>(desc->max_ptr);
    const Tdata *min_val = reinterpret_cast<const Tdata *>(desc->min_ptr);
    
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

#pragma unroll
    for (uint64_t i = 0; i < desc->data_size; i += step) {
        clip<Tdata><<<gridDims, blockDims, 0, cuda_stream>>>(
            y_, x_, max_val, min_val, i);
    }

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaClip(ClipCudaDescriptor_t desc,
                            void *dst, void const *src,
                            void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    if (desc->dtype == F16) {
        return clip_nv_gpu<half>(desc, dst, src, stream);
    }
    if (desc->dtype == F32) {
        return clip_nv_gpu<float>(desc, dst, src, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}