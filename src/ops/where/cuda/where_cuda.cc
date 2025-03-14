#include "where.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateWhereDescriptor(CudaHandle_t handle,
                                          WhereCudaDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t dst,
                                          infiniopTensorDescriptor_t x,
                                          infiniopTensorDescriptor_t y,
                                          infiniopTensorDescriptor_t condition) {
    uint64_t ndim = condition->ndim;
    if (ndim != x->ndim || ndim != y->ndim || ndim != dst->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }  
    for (size_t i = 0; i < ndim; ++i) {
        if (condition->shape[i] != x->shape[i] || condition->shape[i] != y->shape[i] || condition->shape[i] != dst->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (!is_contiguous(condition) || !is_contiguous(x) || !is_contiguous(y) || !is_contiguous(dst)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (x->dt != y->dt || x->dt != dst->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (x->dt != F16 && x->dt != F32 && x->dt != U16) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (condition->dt != U8) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    
    DT dtype = x->dt;
    uint64_t data_size = std::accumulate(condition->shape, condition->shape + condition->ndim, 1ULL, std::multiplies<uint64_t>());


    *desc_ptr = new WhereCudaDescriptor{
        DevNvGpu,
        dtype,
        handle->device_id,
        ndim,
        data_size,
        static_cast<uint64_t>(handle->prop.maxGridSize[0]),
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyWhereDescriptor(WhereCudaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}