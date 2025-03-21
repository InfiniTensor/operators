#include "clip_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <limits> 

inline void incrementOne(uint64_t *indices, uint64_t const *shape, uint64_t ndim) {
    for (int64_t i = ndim - 1; i >= 0; --i) {
        if (++indices[i] != shape[i]) {
            return;
        }
        indices[i] = 0;
    }
}

inline uint64_t compactToFlat(uint64_t const *indices, uint64_t const *strides, uint64_t ndim) {
    return std::inner_product(indices, indices + ndim, strides, uint64_t(0));
}

infiniopStatus_t cpuCreateClipDescriptor(infiniopHandle_t,
                                         ClipCpuDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t output,
                                         infiniopTensorDescriptor_t input,
                                         float min_value = std::numeric_limits<float>::lowest(),
                                         float max_value = std::numeric_limits<float>::max()) {
    uint64_t ndim = output->ndim;

    if (!is_contiguous(output) || !is_contiguous(input)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (output->dt != F16 && output->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (output->dt != input->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    uint64_t o_data_size = std::accumulate(
        output->shape, output->shape + output->ndim, 1ULL, std::multiplies<uint64_t>());

    uint64_t *i_strides = new uint64_t[ndim];
    std::copy(input->strides, input->strides + ndim, i_strides);

    uint64_t *o_indices = new uint64_t[ndim];
    std::fill(o_indices, o_indices + ndim, 0);

    uint64_t *o_shape = new uint64_t[ndim];
    std::copy(output->shape, output->shape + ndim, o_shape);

    *desc_ptr = new ClipCpuDescriptor{
        DevCpu,
        output->dt,
        ndim,
        o_data_size,
        o_shape,
        i_strides,
        o_indices,
        min_value,
        max_value,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyClipDescriptor(ClipCpuDescriptor_t desc) {
    delete[] desc->o_shape;
    delete[] desc->i_strides;
    delete[] desc->o_indices;
    delete desc;
    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t clip_cpu(ClipCpuDescriptor_t desc,
                          void *output,
                          void const *input) {
    auto i_ = reinterpret_cast<Tdata const *>(input);
    auto o_ = reinterpret_cast<Tdata *>(output);
    const auto &indices = desc->o_indices;

    float min_value = desc->min_value;
    float max_value = desc->max_value;
    for (uint64_t i = 0; i < desc->o_data_size; ++i, incrementOne(indices, desc->o_shape, desc->ndim)) {
        auto i_index = compactToFlat(indices, desc->i_strides, desc->ndim);
        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            o_[i] = f32_to_f16(std::fmin(
                max_value,
                std::fmax(f16_to_f32(i_[i_index]), min_value)));
        } else {
            o_[i] = std::fmin(
                max_value,
                std::fmax(i_[i_index], min_value));
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuClip(ClipCpuDescriptor_t desc,
                         void *output,
                         void const *input,
                        //  float min_value = std::numeric_limits<float>::lowest(),
                        //  float max_value = std::numeric_limits<float>::max(),
                         void *stream = nullptr) {
    if (desc->dtype == F16) {
        return clip_cpu<uint16_t>(desc, output, input);
    }
    if (desc->dtype == F32) {
        return clip_cpu<float>(desc, output, input);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
