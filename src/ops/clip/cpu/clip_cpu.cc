#include "clip_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <algorithm>

infiniopStatus_t cpuCreateClipDescriptor(infiniopHandle_t,
                                        ClipCpuDescriptor_t *desc_ptr,
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
    *desc_ptr = new ClipCpuDescriptor{
        DevCpu,
        dtype,
        data_size,
        max,
        min,
    };
    
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyClipDescriptor(ClipCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

template<typename Tdata>

void clip_cpu(ClipCpuDescriptor_t desc, void *dst, void const *src)
{
    auto dstPtr = reinterpret_cast<Tdata*>(dst);
    auto srcPtr = reinterpret_cast<Tdata const*>(src);
    if constexpr (std::is_same<Tdata, uint16_t>::value) 
    {
        for(uint64_t i = 0; i < desc->data_size; i++) {
            float val = f16_to_f32(srcPtr[i]);
            float clipped_val = std::min(std::max(val, desc->min), desc->max);
            dstPtr[i] = f32_to_f16(clipped_val);
        }
    }
    else{
        for(uint64_t i = 0; i < desc->data_size; i++){
            dstPtr[i] = std::min(std::max(srcPtr[i], desc->min), desc->max);
        }
    }

}
infiniopStatus_t cpuClip(ClipCpuDescriptor_t desc,
    void *dst, void const *src,
    void *stream) {
        if(desc->dtype == F32)
        {
            clip_cpu<float>(desc, dst, src);
        }
        else if(desc->dtype == F16)
        {
            clip_cpu<uint16_t>(desc, dst, src);
        }
        return STATUS_SUCCESS;
}

