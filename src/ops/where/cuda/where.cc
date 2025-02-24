#include "where.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateWhereDescriptor(CudaHandle_t handle,
                                          WhereCudaDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t output,
                                          infiniopTensorDescriptor_t x,
                                          infiniopTensorDescriptor_t y,
                                          infiniopTensorDescriptor_t condition) {
    if (!isValidBroadcastShape(output, x) || !isValidBroadcastShape(output, y) || !isValidBroadcastShape(output, condition)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    uint64_t max_rank = std::max(std::max(x->ndim, y->ndim), condition->ndim);
    if (output->ndim != max_rank) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    uint64_t *valid_shape = new uint64_t[max_rank];
    uint64_t *x_padded_shape = new uint64_t[max_rank];
    uint64_t *y_padded_shape = new uint64_t[max_rank];
    uint64_t *condition_padded_shape = new uint64_t[max_rank];
    getBroadcastShape(x->shape, x->ndim, y->shape, y->ndim, condition->shape, condition->ndim, valid_shape, x_padded_shape, y_padded_shape, condition_padded_shape, max_rank);
    for (uint64_t i = 0; i < output->ndim; i++) {
        if (output->shape[i] != valid_shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }

    if (!is_contiguous(y) || !is_contiguous(x)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }

    if (output->dt != x->dt || output->dt != y->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    if (condition->dt != U8) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    uint64_t output_size = std::accumulate(output->shape, output->shape + output->ndim, 1ULL, std::multiplies<uint64_t>());

    uint64_t x_ndim = x->ndim;
    uint64_t y_ndim = y->ndim;
    uint64_t condition_ndim = condition->ndim;

    uint64_t *x_shapes, *y_shapes, *condition_shapes;
    int64_t *x_strides, *y_strides, *condition_strides;

    checkCudaErrorWithCode(cudaMalloc((void**)&x_shapes, x_ndim * sizeof(uint64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc((void**)&y_shapes, y_ndim * sizeof(uint64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc((void**)&condition_shapes, condition_ndim * sizeof(uint64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc((void**)&x_strides, x_ndim * sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc((void**)&y_strides, y_ndim * sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc((void**)&condition_strides, condition_ndim * sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);

    bool x_broadcast = false;
    bool y_broadcast = false;
    bool condition_broadcast = false;

    for (uint64_t i = 0; i < output->ndim; i++) {
        if (x_padded_shape[i] != x->shape[i]) {
            x_broadcast = true;
        }
        if (y_padded_shape[i] != y->shape[i]) {
            y_broadcast = true;
        }
        if (condition_padded_shape[i] != condition->shape[i]) {
            condition_broadcast = true;
        }
    }

    if(x_broadcast) {
        checkCudaErrorWithCode(cudaMemcpy(x_shapes, x->shape, x_ndim * sizeof(uint64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
        checkCudaErrorWithCode(cudaMemcpy(x_strides, x->strides, x_ndim * sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    }
    if(y_broadcast) {
        checkCudaErrorWithCode(cudaMemcpy(y_shapes, y->shape, y_ndim * sizeof(uint64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
        checkCudaErrorWithCode(cudaMemcpy(y_strides, y->strides, y_ndim * sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    }
    if(condition_broadcast) {
        checkCudaErrorWithCode(cudaMemcpy(condition_shapes, condition->shape, condition_ndim * sizeof(uint64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
        checkCudaErrorWithCode(cudaMemcpy(condition_strides, condition->strides, condition_ndim * sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    }

    *desc_ptr = new WhereCudaDescriptor{
        DevNvGpu,
        output->dt,
        handle->device_id,
        output_size,
        x_broadcast,
        x_ndim,
        x_shapes,
        x_strides,
        y_broadcast,
        y_ndim,
        y_shapes,
        y_strides,
        condition_broadcast,
        condition_ndim,
        condition_shapes,
        condition_strides,
        static_cast<uint64_t>(handle->prop.maxGridSize[0]),
    };

    delete [] valid_shape;
    delete [] x_padded_shape;
    delete [] y_padded_shape;
    delete [] condition_padded_shape;

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyWhereDescriptor(WhereCudaDescriptor_t desc) {
    checkCudaErrorWithCode(cudaFree((void*)desc->x_shapes), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree((void*)desc->x_strides), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree((void*)desc->y_shapes), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree((void*)desc->y_strides), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree((void*)desc->condition_shapes), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree((void*)desc->condition_strides), STATUS_EXECUTION_FAILED);
    delete desc;
    return STATUS_SUCCESS;
}
