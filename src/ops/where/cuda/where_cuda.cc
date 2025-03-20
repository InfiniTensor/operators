#include "where_cuda.h"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateWhereDescriptor(CudaHandle_t handle,
                                            WhereCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t dst,
                                            infiniopTensorDescriptor_t src1,
                                            infiniopTensorDescriptor_t src2,
                                            infiniopTensorDescriptor_t condition
                                            ) {
    if (!isValidBroadcastShape(dst, src1) || !isValidBroadcastShape(dst, src2) || !isValidBroadcastShape(dst, condition)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (dst->dt != F16 && dst->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (src1->dt != F16 && src1->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (src2->dt != F16 && src2->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (condition->dt != U8) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (!is_contiguous(dst) || !is_contiguous(src1) || !is_contiguous(src2) || !is_contiguous(condition)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (dst->dt != src1->dt || dst->dt != src2->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    uint64_t *src1_shape = new uint64_t[dst->ndim];
    uint64_t *src2_shape = new uint64_t[dst->ndim];
    uint64_t *condition_shape = new uint64_t[dst->ndim];
    uint64_t *dst_shape = new uint64_t[dst->ndim];

    int64_t *dst_strides = new int64_t[dst->ndim];
    int64_t *src1_strides = new int64_t[src1->ndim];
    int64_t *src2_strides = new int64_t[src2->ndim];
    int64_t *condition_strides = new int64_t[condition->ndim];
    uint64_t dst_ndim = dst->ndim;
    uint64_t element_num = 1;
    for (uint64_t i = 0; i < dst->ndim; i++) {
        element_num *= dst->shape[i];
    }

    memcpy(src1_shape, src1->shape, src1->ndim * sizeof(uint64_t));
    memcpy(src2_shape, src2->shape, src2->ndim * sizeof(uint64_t));
    memcpy(condition_shape, condition->shape, condition->ndim * sizeof(uint64_t));
    memcpy(dst_shape, dst->shape, dst->ndim * sizeof(uint64_t));

    memcpy(dst_strides, dst->strides, dst->ndim * sizeof(int64_t));
    memcpy(src1_strides, src1->strides, src1->ndim * sizeof(int64_t));
    memcpy(src2_strides, src2->strides, src2->ndim * sizeof(int64_t));
    memcpy(condition_strides, condition->strides, condition->ndim * sizeof(int64_t));

    uint64_t *src1_device_shape, *src2_device_shape, *condition_device_shape, *dst_device_shape;
    int64_t *dst_device_strides, *src1_device_strides, *src2_device_strides, *condition_device_strides;
    checkCudaErrorWithCode(cudaMalloc((void**)&src1_device_shape, src1->ndim * sizeof(uint64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc((void**)&src2_device_shape, src2->ndim * sizeof(uint64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc((void**)&condition_device_shape, condition->ndim * sizeof(uint64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc((void**)&dst_device_shape, dst->ndim * sizeof(uint64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    
    checkCudaErrorWithCode(cudaMalloc((void**)&dst_device_strides, dst->ndim * sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc((void**)&src1_device_strides, src1->ndim * sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc((void**)&src2_device_strides, src2->ndim * sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc((void**)&condition_device_strides, condition->ndim * sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    
    checkCudaErrorWithCode(cudaMemcpy(src1_device_shape, src1_shape, src1->ndim * sizeof(uint64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaMemcpy(src2_device_shape, src2_shape, src2->ndim * sizeof(uint64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaMemcpy(condition_device_shape, condition_shape, condition->ndim * sizeof(uint64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaMemcpy(dst_device_shape, dst->shape, dst->ndim * sizeof(uint64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);

    checkCudaErrorWithCode(cudaMemcpy(dst_device_strides, dst_strides, dst->ndim * sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaMemcpy(src1_device_strides, src1_strides, src1->ndim * sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaMemcpy(src2_device_strides, src2_strides, src2->ndim * sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaMemcpy(condition_device_strides, condition_strides, condition->ndim * sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);

    *desc_ptr = new WhereCudaDescriptor{
        handle->device,
        dst->dt,
        src1_device_shape,
        src2_device_shape,
        condition_device_shape,
        dst_device_shape,
        dst_device_strides,
        src1_device_strides,
        src2_device_strides,
        condition_device_strides,
        dst_ndim,
        src1->ndim,
        src2->ndim,
        condition->ndim,
        element_num
    };
    delete[] src1_shape;
    delete[] src2_shape;
    delete[] condition_shape;
    delete[] dst_shape;
    delete[] dst_strides;
    delete[] src1_strides;
    delete[] src2_strides;
    delete[] condition_strides;
    return STATUS_SUCCESS;
}


infiniopStatus_t cudaDestroyWhereDescriptor(WhereCudaDescriptor_t desc) {
    checkCudaErrorWithCode(cudaFree((void*)desc->src1_shape), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree((void*)desc->src2_shape), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree((void*)desc->condition_shape), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree((void*)desc->dst_shape), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree((void*)desc->dst_strides), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree((void*)desc->src1_strides), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree((void*)desc->src2_strides), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree((void*)desc->condition_strides), STATUS_EXECUTION_FAILED);
    delete desc;
    return STATUS_SUCCESS;
}
