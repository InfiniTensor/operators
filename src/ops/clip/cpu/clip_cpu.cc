#include "clip_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateClipDescriptor(infiniopHandle_t handle,
    ClipCpuDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t y,
    float* min,
    float* max
    ){

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
    bool has_min = min != nullptr;
    bool has_max = max != nullptr;
    float min_ = has_min ? *min : std::numeric_limits<float>::lowest();
    float max_ = has_max ? *max : std::numeric_limits<float>::max();
    *desc_ptr = new ClipCpuDescriptor{
        DevCpu,
        x->dt,
        min_,
        max_,
        has_min,
        has_max,
        element_num
    };
    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t clip_cpu(ClipCpuDescriptor_t desc,
    void const *x,
    void *y){
    auto x_ = reinterpret_cast<Tdata const *>(x);
    auto y_ = reinterpret_cast<Tdata *>(y);
    for (uint64_t i = 0; i < desc->element_num; i++) {
        if constexpr (std::is_same<Tdata, uint16_t>::value){
            float x_f = f16_to_f32(x_[i]);
            x_f = desc->has_min ? std::max(x_f, desc->min) : x_f;
            x_f = desc->has_max ? std::min(x_f, desc->max) : x_f;
            y_[i] = f32_to_f16(x_f);
        }
        else{
            y_[i] = std::min(std::max(x_[i], desc->min), desc->max);
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuClip(ClipCpuDescriptor_t desc,
    void const*x,
    void *y,
    void *stream){
    if (desc->dtype == F16) {
        return clip_cpu<uint16_t>(desc, x, y);
    }
    if (desc->dtype == F32) {
        return clip_cpu<float>(desc, x, y);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
infiniopStatus_t cpuDestroyClipDescriptor(ClipCpuDescriptor_t desc){
    delete desc;
    return STATUS_SUCCESS;
}
