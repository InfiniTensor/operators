#include "clip_cuda.h"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateClipDescriptor(CudaHandle_t handle,
                                            ClipCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t x,
                                            infiniopTensorDescriptor_t y,
                                            float *min,
                                            float *max
                                            ) {
    if (x->dt != F16 && x->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (x->ndim != y->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (x->dt != y->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (!is_contiguous(x) || !is_contiguous(y)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    uint64_t element_num = 1;
    for (uint64_t i = 0; i < x->ndim; i++) {
        element_num *= x->shape[i];
    }
    uint64_t ndim = y->ndim;
    bool has_min = min != nullptr;
    bool has_max = max != nullptr;
    float min_ = has_min ? *min : std::numeric_limits<float>::lowest();
    float max_ = has_max ? *max : std::numeric_limits<float>::max();
    *desc_ptr = new ClipCudaDescriptor{
        DevNvGpu,
        x->dt,
        ndim,
        element_num,
        min_,
        max_,
        has_min,
        has_max
    };
    return STATUS_SUCCESS;
}


infiniopStatus_t cudaDestroyClipDescriptor(ClipCudaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
