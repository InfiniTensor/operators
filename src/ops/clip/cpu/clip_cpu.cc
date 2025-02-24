#include "clip_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateClipDescriptor(infiniopHandle_t,
                                         ClipCpuDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t y,
                                         infiniopTensorDescriptor_t x,
                                         float *lower_bound,
                                         float *upper_bound) {
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

    *desc_ptr = new ClipCpuDescriptor{
        DevCpu,
        y->dt,
        data_size,
        has_lower_bound,
        lower_bound_,
        has_upper_bound,
        upper_bound_,
    };
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyClipDescriptor(ClipCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t clip_cpu(ClipCpuDescriptor_t desc, void *y, void const *x) {
    auto x_ = reinterpret_cast<Tdata const *>(x);
    auto y_ = reinterpret_cast<Tdata *>(y);
    auto lower_bound_ = desc->lower_bound;
    auto upper_bound_ = desc->upper_bound;
    auto data_size_ = desc->data_size;

    if constexpr (std::is_same<Tdata, uint16_t>::value) {
        if (!desc->has_lower_bound && !desc->has_upper_bound) {
            std::memcpy(y_, x_, data_size_ * sizeof(Tdata));
        } else {
#pragma omp parallel for
            for (uint64_t i = 0; i < data_size_; i++) {
                float x_val = f16_to_f32(x_[i]);
                x_val = (desc->has_lower_bound && x_val < lower_bound_) ? lower_bound_ : x_val;
                x_val = (desc->has_upper_bound && x_val > upper_bound_) ? upper_bound_ : x_val;
                y_[i] = f32_to_f16(x_val);
            }
        }
    } else {
        if (!desc->has_lower_bound && !desc->has_upper_bound) {
            std::memcpy(y_, x_, data_size_ * sizeof(Tdata));
        } else {
#pragma omp parallel for
            for (uint64_t i = 0; i < data_size_; i++) {
                Tdata x_val = x_[i];
                x_val = (desc->has_lower_bound && x_val < lower_bound_) ? lower_bound_ : x_val;
                x_val = (desc->has_upper_bound && x_val > upper_bound_) ? upper_bound_ : x_val;
                y_[i] = x_val;
            }
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuClip(ClipCpuDescriptor_t desc, void *y, void const *x) {
    if (desc->dtype == F16) {
        return clip_cpu<uint16_t>(desc, y, x);
    }
    if (desc->dtype == F32) {
        return clip_cpu<float>(desc, y, x);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}