#include "swiglu.h"


infiniopStatus_t ascendCreateSwiGLUDescriptor(AscendHandle_t handle,
                                              SwiGLUAscendDescriptor_t *desc_ptr,
                                              infiniopTensorDescriptor_t c_desc,
                                              infiniopTensorDescriptor_t a_desc,
                                              infiniopTensorDescriptor_t b_desc) {
    uint64_t ndim = c_desc->ndim;
    DT dtype = c_desc->dt;

    aclDataType dt;
    if (dtype_eq(dtype, F16)) {
        dt = aclDataType::ACL_FLOAT16;
    } else if (dtype_eq(dtype, F32)) {
        dt = aclDataType::ACL_FLOAT;
    } else {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    if (ndim != 2 || a_desc->ndim != 2 || b_desc->ndim != 2) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    if (c_desc->strides[1] != 1 || a_desc->strides[1] != 1 || b_desc->strides[1] != 1) {
        return STATUS_BAD_TENSOR_STRIDES;
    }

    int32_t seq_len = static_cast<int32_t>(c_desc->shape[0]),
            di = static_cast<int32_t>(c_desc->shape[1]);

    int32_t sta = static_cast<int32_t>(a_desc->strides[0]);
    int32_t stb = static_cast<int32_t>(b_desc->strides[0]);
    int32_t stc = static_cast<int32_t>(c_desc->strides[0]);

    *desc_ptr = new SwiGLUAscendDescriptor{
        handle->device,
        handle->device_id,
        dt,
        seq_len,
        di,
        sta,
        stb,
        stc};
    return STATUS_SUCCESS;
}

infiniopStatus_t ascendSwiGLU(SwiGLUAscendDescriptor_t desc,
                              void *c,
                              void const *a,
                              void const *b,
                              void *stream) {
    auto seq_len = desc->seq_len,
         di = desc->di;

    auto sta = desc->sta,
         stb = desc->stb,
         stc = desc->stc;

    auto dt = desc->dtype;
    
    // Set device
    // aclrtSetDevice(desc->device_id);

    return swiglu_kernel_do(c, (void *) a, (void *) b, 1.0, seq_len, di, sta, stb, stc, dt, stream);
}

infiniopStatus_t ascendDestroySwiGLUDescriptor(SwiGLUAscendDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
