#include "clip.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include <optional>

infiniopStatus_t cudaCreateClipDescriptor(CudaHandle_t handle,
                                          ClipCudaDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t y,
                                          infiniopTensorDescriptor_t x,
                                          float* lower_bound,
                                          float* upper_bound) {
    if (x->ndim != y->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    for (uint64_t i = 0; i < x->ndim; i++) {
        if (x->shape[i] != y->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }

    if (!is_contiguous(y) || !is_contiguous(x)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }

    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    uint64_t data_size = std::accumulate(y->shape, y->shape + y->ndim, 1ULL, std::multiplies<uint64_t>());

    bool has_lower_bound = lower_bound != nullptr;
    bool has_upper_bound = upper_bound != nullptr;
    float lower_bound_ = has_lower_bound ? *lower_bound : std::numeric_limits<float>::lowest();
    float upper_bound_ = has_upper_bound ? *upper_bound : std::numeric_limits<float>::max();

    *desc_ptr = new ClipCudaDescriptor{
        DevNvGpu,
        y->dt,
        handle->device_id,
        data_size,
        has_lower_bound,
        lower_bound_,
        has_upper_bound,
        upper_bound_,
        static_cast<uint64_t>(handle->prop.maxGridSize[0]),
    };


    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyClipDescriptor(ClipCudaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
