#include "../../devices/cpu/common_cpu.h"
#include "../utils.h"
#include "operators.h"
#include "ops/clip/clip.h"
#include "status.h"
#include "tensor/tensor_descriptor.h"
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>

struct _ClipDescriptor {
    Device device;
    DT dtype;
    uint64_t output_data_size;
    float min_value;
    float max_value;
};

typedef struct _ClipDescriptor *_ClipDescriptor_t;

__C __export infiniopStatus_t infiniopCreateClipDescriptor(infiniopHandle_t handle,
                                                           infiniopClipDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t output,
                                                           infiniopTensorDescriptor_t input,
                                                           float *min,
                                                           float *max) {
    if (!is_contiguous(output) || !is_contiguous(input)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (output->dt != input->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    // type: f16, f32
    if (output->dt != F16 && output->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    uint64_t output_data_size = std::accumulate(output->shape, output->shape + output->ndim, 1ULL, std::multiplies<uint64_t>());

    float min_val = (min != nullptr) ? *min : std::numeric_limits<float>::lowest();
    float max_val = (max != nullptr) ? *max : std::numeric_limits<float>::max();

    *(_ClipDescriptor_t *) desc_ptr = new _ClipDescriptor{
        handle->device,
        output->dt,
        output_data_size,
        min_val,
        max_val};
    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t clip_cpu(_ClipDescriptor_t desc, void *output, void const *input) {
    auto output_data = reinterpret_cast<Tdata *>(output);
    auto input_data = reinterpret_cast<Tdata const *>(input);
    float min_value = desc->min_value;
    float max_value = desc->max_value;

    for (uint64_t i = 0; i < desc->output_data_size; ++i) {
        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            output_data[i] = f32_to_f16(std::min(std::max(f16_to_f32(input_data[i]), min_value), max_value));
        } else {
            output_data[i] = std::min(std::max(input_data[i], min_value), max_value);
        }
    }

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopClip(infiniopClipDescriptor_t desc,
                                           void *output,
                                           void const *input,
                                           void *stream) {
    auto _desc = (_ClipDescriptor_t) desc;
    auto dtype = _desc->dtype;

    if (dtype == F16) {
        return clip_cpu<uint16_t>(_desc, output, input);
    } else if (dtype == F32) {
        return clip_cpu<float>(_desc, output, input);
    }

    return STATUS_BAD_TENSOR_DTYPE;
}

__C __export infiniopStatus_t infiniopDestroyClipDescriptor(infiniopClipDescriptor_t desc) {
    delete (_ClipDescriptor_t) desc;
    return STATUS_SUCCESS;
}
