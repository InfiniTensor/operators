#include "clip.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateClipDescriptor(CudaHandle_t handle,
                                          ClipCudaDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t dst,
                                          infiniopTensorDescriptor_t src,
                                          float max,
                                          float min) {
    uint64_t ndim = src->ndim;
    if (ndim != dst->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    for (size_t i = 0; i < ndim; ++i) {
        if (src->shape[i] != dst->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (!is_contiguous(src) || !is_contiguous(dst)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (src->dt != F16 && src->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (src->dt != dst->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    
    DT dtype = src->dt;
    uint64_t data_size = std::accumulate(src->shape, src->shape + src->ndim, 1ULL, std::multiplies<uint64_t>());
    void *max_ptr, *min_ptr;
    if(dtype == F32)
    {
        checkCudaErrorWithCode(cudaMalloc(&max_ptr, sizeof(float)), STATUS_MEMORY_NOT_ALLOCATED);
        checkCudaErrorWithCode(cudaMalloc(&min_ptr, sizeof(float)), STATUS_MEMORY_NOT_ALLOCATED);
        float* max_p = static_cast<float*>(max_ptr);
        float* min_p = static_cast<float*>(min_ptr);
        checkCudaErrorWithCode(cudaMemcpy(max_p, &max, sizeof(float), cudaMemcpyHostToDevice), STATUS_MEMORY_NOT_ALLOCATED);
        checkCudaErrorWithCode(cudaMemcpy(min_p, &min, sizeof(float), cudaMemcpyHostToDevice), STATUS_MEMORY_NOT_ALLOCATED);       
    }
    else if(dtype == F16)
    {
        half max_h = __float2half(max);
        half min_h = __float2half(min);
        checkCudaErrorWithCode(cudaMalloc(&max_ptr, sizeof(half)), STATUS_MEMORY_NOT_ALLOCATED);
        checkCudaErrorWithCode(cudaMalloc(&min_ptr, sizeof(half)), STATUS_MEMORY_NOT_ALLOCATED);
        half* max_p = static_cast<half*>(max_ptr);
        half* min_p = static_cast<half*>(min_ptr);
        checkCudaErrorWithCode(cudaMemcpy(max_p, &max_h, sizeof(half), cudaMemcpyHostToDevice), STATUS_MEMORY_NOT_ALLOCATED);
        checkCudaErrorWithCode(cudaMemcpy(min_p, &min_h, sizeof(half), cudaMemcpyHostToDevice), STATUS_MEMORY_NOT_ALLOCATED);       
    }


    *desc_ptr = new ClipCudaDescriptor{
        DevNvGpu,
        dtype,
        handle->device_id,
        ndim,
        data_size,
        static_cast<uint64_t>(handle->prop.maxGridSize[0]),
        max_ptr,
        min_ptr,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyClipDescriptor(ClipCudaDescriptor_t desc) {
    checkCudaErrorWithCode(cudaFree(desc->max_ptr), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree(desc->min_ptr), STATUS_EXECUTION_FAILED);
    delete desc;
    return STATUS_SUCCESS;
}