#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "where.cuh"

template<typename Tdata>
__global__ void where(    
    Tdata *dst,
    const Tdata *x,
    const Tdata *y,
    const uint8_t *condition,
    uint64_t offset){
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    dst[idx] = (condition[idx]) ? y[idx] : x[idx];

}


template<typename Tdata>
infiniopStatus_t where_nv_gpu(WhereCudaDescriptor_t desc, void *dst, void const *x, void const *y, void const *condition, void *stream) {
    if (desc->data_size == 0) {
        return STATUS_SUCCESS;
    }
    dim3 blockDims = dim3(std::min(static_cast<uint64_t>(256), desc->data_size));
    dim3 gridDims = dim3(std::min(ROUND_UP_DIV(desc->data_size, blockDims.x), desc->max_grid_size));
    uint64_t step = gridDims.x * blockDims.x;

    const auto dst_ = reinterpret_cast<Tdata*>(dst);
    const auto x_ = reinterpret_cast<Tdata const*>(x);
    const auto y_ = reinterpret_cast<Tdata const*>(y);
    const auto condition_ = reinterpret_cast<uint8_t const*>(condition);
    
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

#pragma unroll
    for (uint64_t i = 0; i < desc->data_size; i += step) {
        printf("%ld\n", (int64_t)i);
        where<Tdata><<<gridDims, blockDims, 0, cuda_stream>>>(dst_, x_, y_, condition_, i);
    }

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaWhere(WhereCudaDescriptor_t desc,
                            void *dst,                            
                            void const *x,
                            void const *y,
                            void const *condition,
                            void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    if (desc->dtype == F16) {
        return where_nv_gpu<half>(desc, dst, x, y, condition, stream);
    }
    if (desc->dtype == F32) {
        return where_nv_gpu<float>(desc, dst, x, y, condition, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
