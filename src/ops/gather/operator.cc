#include "../../devices/cpu/common_cpu.h"
#include "../utils.h"
#include "data_type.h"
#include "operators.h"
#include "ops/gather/gather.h"
#include "status.h"
#include "tensor/tensor_descriptor.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <type_traits>

struct _GatherDescriptor {
    Device device;
    DT dtype;
    DT indices_type;
    uint64_t data_ndim;
    uint64_t indices_ndim;
    uint64_t output_ndim;
    uint64_t output_data_size;
    uint64_t const *output_shape;
    uint64_t const *input_strides;
    uint64_t const *indices_strides;
    uint64_t *output_indices;
    int64_t axis;
};

typedef struct _GatherDescriptor *_GatherDescriptor_t;

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

__C __export infiniopStatus_t infiniopCreateGatherDescriptor(infiniopHandle_t handle,
                                                             infiniopGatherDescriptor_t *desc_ptr,
                                                             infiniopTensorDescriptor_t output,
                                                             infiniopTensorDescriptor_t input,
                                                             infiniopTensorDescriptor_t indices,
                                                             int64_t axis) {
    if (!is_contiguous(input) || !is_contiguous(indices) || !is_contiguous(output)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (indices->dt != I32 && indices->dt != I64) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (output->dt != input->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    uint64_t data_ndim = input->ndim;
    uint64_t indices_ndim = indices->ndim;
    if (axis < -(int64_t(data_ndim)) || axis >= int64_t(data_ndim)) {
        return STATUS_BAD_PARAM;
    }
    if (axis < 0) {
        axis += data_ndim;
    }

    uint64_t output_ndim = data_ndim - 1 + indices_ndim;
    uint64_t *output_shape = new uint64_t[output_ndim];
    for (uint64_t i = 0; i < axis; ++i) {
        output_shape[i] = input->shape[i];
    }
    for (uint64_t i = 0; i < indices_ndim; ++i) {
        output_shape[axis + i] = indices->shape[i];
    }
    for (uint64_t i = axis + 1; i < data_ndim; ++i) {
        output_shape[i + indices_ndim - 1] = input->shape[i];
    }

    for (uint64_t i = 0; i < output_ndim; ++i) {
        if (output_shape[i] != output->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }

    uint64_t output_data_size = std::accumulate(output->shape, output->shape + output->ndim, 1ULL, std::multiplies<uint64_t>());

    uint64_t *output_indices = new uint64_t[output_ndim];
    std::fill(output_indices, output_indices + output_ndim, 0);

    uint64_t *input_strides = new uint64_t[data_ndim];
    for (uint64_t i = 0; i < data_ndim; ++i) {
        input_strides[i] = input->strides[i];
    }
    uint64_t *indices_strides = new uint64_t[indices_ndim];
    for (uint64_t i = 0; i < indices_ndim; ++i) {
        indices_strides[i] = indices->strides[i];
    }

    *(_GatherDescriptor_t *) desc_ptr = new _GatherDescriptor{
        handle->device,
        output->dt,
        indices->dt,
        data_ndim,
        indices_ndim,
        output_ndim,
        output_data_size,
        output_shape,
        input_strides,
        indices_strides,
        output_indices,
        axis};

    return STATUS_SUCCESS;
}

template<typename Tdata, typename Tindices>
infiniopStatus_t gather_cpu(_GatherDescriptor_t desc, void *output, void const *input, void const *indices) {
    auto input_ = reinterpret_cast<Tdata const *>(input);
    auto indices_ = reinterpret_cast<Tindices const *>(indices);
    auto output_ = reinterpret_cast<Tdata *>(output);

    const auto &output_indices = desc->output_indices;
    for (uint64_t i = 0; i < desc->output_data_size; ++i, incrementOne(output_indices, desc->output_shape, desc->output_ndim)) {
        // 下标部分：在 output_indices 中提取 [axis, axis+indices_ndim-1]
        uint64_t flat_indices = compactToFlat(output_indices + desc->axis, desc->indices_strides, desc->indices_ndim);
        int gather_index = indices_[flat_indices];
        // 计算 data 对应多维下标（前+gather_index+后）
        uint64_t *in_multi = new uint64_t[desc->data_ndim];
        for (uint64_t j = 0; j < desc->axis; ++j)
            in_multi[j] = output_indices[j];
        in_multi[desc->axis] = gather_index;
        for (uint64_t j = desc->axis + 1; j < desc->data_ndim; ++j)
            in_multi[j] = output_indices[j + desc->indices_ndim - 1];
        uint64_t in_flat = compactToFlat(in_multi, desc->input_strides, desc->data_ndim);
        delete[] in_multi;
        output_[i] = input_[in_flat];
    }
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopGather(infiniopGatherDescriptor_t desc,
                                             void *output,
                                             void const *input,
                                             void const *indices,
                                             void *stream) {
    auto _desc = (_GatherDescriptor_t) desc;
    auto dtype = _desc->dtype;
    auto indices_type = _desc->indices_type;

    if (dtype == F16) {
        if (indices_type == I32) {
            return gather_cpu<uint16_t, int32_t>(_desc, output, input, indices);
        } else if (indices_type == I64) {
            return gather_cpu<uint16_t, int64_t>(_desc, output, input, indices);
        }
    } else if (dtype == F32) {
        if (indices_type == I32) {
            return gather_cpu<float, int32_t>(_desc, output, input, indices);
        } else if (indices_type == I64) {
            return gather_cpu<float, int64_t>(_desc, output, input, indices);
        }
    }

    return STATUS_BAD_TENSOR_DTYPE;
}

__C __export infiniopStatus_t infiniopDestroyGatherDescriptor(infiniopGatherDescriptor_t desc) {
    auto _desc = (_GatherDescriptor_t) desc;
    delete[] _desc->output_shape;
    delete[] _desc->input_strides;
    delete[] _desc->indices_strides;
    delete[] _desc->output_indices;

    delete (_GatherDescriptor_t) desc;
    return STATUS_SUCCESS;
}
