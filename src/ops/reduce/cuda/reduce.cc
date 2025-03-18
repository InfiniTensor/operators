#include "reduce.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateReduceDescriptor(CudaHandle_t handle,
                                            ReduceCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t y,
                                            infiniopTensorDescriptor_t x,
                                            int *axes,
                                            uint64_t axes_ndim,
                                            int reduce_op) {
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    if (!is_contiguous(y) || !is_contiguous(x)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }

    uint64_t ndim = x->ndim;
    if (ndim > 8) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    int x_shape[CUDNN_DIM_MAX];
    for (uint64_t i = 0; i < ndim; ++i) {
        x_shape[i] = static_cast<int>(x->shape[i]);
    }

    int x_strides[CUDNN_DIM_MAX];
    for (uint64_t i = 0; i < ndim; ++i) {
        x_strides[i] = static_cast<int>(x->strides[i]);
    }

    int y_shape[CUDNN_DIM_MAX];
    for (uint64_t i = 0; i < ndim; ++i) {
        bool is_axis = false;
        for (uint64_t j = 0; j < axes_ndim; ++j) {
            if (axes[j] == i) {
                is_axis = true;
                break;
            }
        }
        if (is_axis) {
            y_shape[i] = 1;
        } else {
            y_shape[i] = static_cast<int>(x->shape[i]);
        }
    }

    int y_strides[CUDNN_DIM_MAX];
    int stride = 1;
    for (uint64_t i = ndim; i > 0; --i) {
        y_strides[i - 1] = stride;
        stride *= y_shape[i - 1];
    }

    CREATE_CHECK_ERROR(auto dt = dataTypeMap[x->dt], dt, -1, STATUS_BAD_PARAM);
    cudnnDataType_t cudnn_dt = [&] {
        switch (dt) {
            case CUDNN_DATA_HALF:
                if (handle->compute_capability_major > 5 || (handle->compute_capability_major == 5 && handle->compute_capability_minor >= 3)) {
                    return CUDNN_DATA_HALF;
                }
                return CUDNN_DATA_FLOAT;
            case CUDNN_DATA_BFLOAT16:
            case CUDNN_DATA_FLOAT:
                return CUDNN_DATA_FLOAT;
            case CUDNN_DATA_DOUBLE:
                return CUDNN_DATA_DOUBLE;
            default:
                return CUDNN_DATA_INT32;
        }
    }();

    cudnnTensorDescriptor_t x_desc;
    checkCudnnError(cudnnCreateTensorDescriptor(&x_desc));
    cudnnTensorDescriptor_t y_desc;
    checkCudnnError(cudnnCreateTensorDescriptor(&y_desc));
    if (ndim > 4) {
        checkCudnnError(cudnnSetTensorNdDescriptor(x_desc, cudnn_dt, ndim, x_shape, x_strides));
        checkCudnnError(cudnnSetTensorNdDescriptor(y_desc, cudnn_dt, ndim, y_shape, y_strides));
    } else {
        int x_shape_[4] = {1, 1, 1, 1};
        int y_shape_[4] = {1, 1, 1, 1};
        for (int i = 0; i < ndim; ++i) {
            x_shape_[4 - i - 1] = x_shape[ndim - i - 1];
            y_shape_[4 - i - 1] = y_shape[ndim - i - 1];
        }
        checkCudnnError(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, cudnn_dt, x_shape_[0], x_shape_[1], x_shape_[2], x_shape_[3]));
        checkCudnnError(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, cudnn_dt, y_shape_[0], y_shape_[1], y_shape_[2], y_shape_[3]));
    }

    cudnnReduceTensorDescriptor_t reduce_desc;
    checkCudnnError(cudnnCreateReduceTensorDescriptor(&reduce_desc));
    cudnnReduceTensorOp_t reduce_op_ = [&] {
        switch (reduce_op) {
            case 0:
                return CUDNN_REDUCE_TENSOR_MIN;
            case 1:
                return CUDNN_REDUCE_TENSOR_MAX;
            case 2:
                return CUDNN_REDUCE_TENSOR_AVG;
        }
    }();
    checkCudnnError(cudnnSetReduceTensorDescriptor(reduce_desc, reduce_op_, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));

    uint64_t workspace_size = 0;
    if (use_cudnn(handle->cudnn_handles_t, handle->device_id, nullptr,
                  [&](cudnnHandle_t handle) { return cudnnGetReductionWorkspaceSize(handle, reduce_desc, x_desc, y_desc, &workspace_size); }) != CUDNN_STATUS_SUCCESS) {
        return STATUS_EXECUTION_FAILED;
    }
    *desc_ptr = new ReduceCudaDescriptor{
        DevNvGpu,
        y->dt,
        handle->device_id,
        handle->cudnn_handles_t,
        x_desc,
        y_desc,
        reduce_desc,
        workspace_size};

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaGetReduceWorkspaceSize(ReduceCudaDescriptor_t desc, uint64_t *size) {
    *size = desc->workspace_size;
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyReduceDescriptor(ReduceCudaDescriptor_t desc) {
    checkCudnnError(cudnnDestroyReduceTensorDescriptor(desc->reduce_desc));
    checkCudnnError(cudnnDestroyTensorDescriptor(desc->y_desc));
    checkCudnnError(cudnnDestroyTensorDescriptor(desc->x_desc));
    desc->cudnn_handles_t = nullptr;
    delete desc;
    return STATUS_SUCCESS;
}
