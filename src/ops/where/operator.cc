#include "../../devices/cpu/common_cpu.h"
#include "../utils.h"
#include "data_type.h"
#include "operators.h"
#include "ops/where/where.h"
#include "status.h"
#include "tensor/tensor_descriptor.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <type_traits>

struct _WhereDescriptor {
    Device device;
    DT dtype;
    uint64_t ndim;
    uint64_t output_data_size;
    uint64_t const *output_shape;
    uint64_t const *condition_strides;
    uint64_t const *x_strides;
    uint64_t const *y_strides;
    uint64_t *output_indices;
};

typedef struct _WhereDescriptor *_WhereDescriptor_t;

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

__C __export infiniopStatus_t infiniopCreateWhereDescriptor(infiniopHandle_t handle,
                                                            infiniopWhereDescriptor_t *desc_ptr,
                                                            infiniopTensorDescriptor_t output,
                                                            infiniopTensorDescriptor_t condition,
                                                            infiniopTensorDescriptor_t x,
                                                            infiniopTensorDescriptor_t y) {
    if (!isValidBroadcastShape(output, condition) ||
        !isValidBroadcastShape(output, x) ||
        !isValidBroadcastShape(output, y)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!is_contiguous(x) || !is_contiguous(y) || !is_contiguous(condition) || !is_contiguous(output)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (condition->dt != U8) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (output->dt != x->dt || output->dt != y->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    uint64_t output_data_size = std::accumulate(output->shape, output->shape + output->ndim, 1ULL, std::multiplies<uint64_t>());
    uint64_t ndim = output->ndim;
    uint64_t *condition_strides = new uint64_t[ndim];
    uint64_t *x_strides = new uint64_t[ndim];
    uint64_t *y_strides = new uint64_t[ndim];
    for (size_t i = 0; i < ndim; i++) {
        condition_strides[i] = (i < ndim - condition->ndim || output->shape[i] != condition->shape[i + condition->ndim - ndim]) ? 0 : condition->strides[i + condition->ndim - ndim];
        x_strides[i] = (i < ndim - x->ndim || output->shape[i] != x->shape[i + x->ndim - ndim]) ? 0 : x->strides[i + x->ndim - ndim];
        y_strides[i] = (i < ndim - y->ndim || output->shape[i] != y->shape[i + y->ndim - ndim]) ? 0 : y->strides[i + y->ndim - ndim];
    }

    uint64_t *output_indices = new uint64_t[ndim];
    std::fill(output_indices, output_indices + ndim, 0);
    uint64_t *output_shape = new uint64_t[ndim];
    std::copy(output->shape, output->shape + ndim, output_shape);

    *(_WhereDescriptor_t *) desc_ptr = new _WhereDescriptor{
        handle->device,
        output->dt,
        ndim,
        output_data_size,
        output_shape,
        condition_strides,
        x_strides,
        y_strides,
        output_indices};

    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t where_cpu(_WhereDescriptor_t desc, void *output, void const *condition, void const *x, void const *y) {
    auto x_ = reinterpret_cast<Tdata const *>(x);
    auto y_ = reinterpret_cast<Tdata const *>(y);
    auto output_ = reinterpret_cast<Tdata *>(output);
    auto condition_ = reinterpret_cast<uint8_t const *>(condition);
    const auto &indices = desc->output_indices;

    for (uint64_t i = 0; i < desc->output_data_size; ++i, incrementOne(indices, desc->output_shape, desc->ndim)) {
        auto x_index = compactToFlat(indices, desc->x_strides, desc->ndim);
        auto y_index = compactToFlat(indices, desc->y_strides, desc->ndim);
        auto condition_index = compactToFlat(indices, desc->condition_strides, desc->ndim);

        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            output_[i] = f32_to_f16(f16_to_f32(condition_[condition_index] ? x_[x_index] : y_[y_index]));
        } else {
            output_[i] = condition_[condition_index] ? x_[x_index] : y_[y_index];
        }
    }

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopWhere(infiniopWhereDescriptor_t desc,
                                            void *output,
                                            void const *condition,
                                            void const *x,
                                            void const *y,
                                            void *stream) {
    auto _desc = (_WhereDescriptor_t) desc;
    auto dtype = _desc->dtype;

    if (dtype == F16) {
        return where_cpu<uint16_t>(_desc, output, condition, x, y);
    } else if (dtype == F32) {
        return where_cpu<float>(_desc, output, condition, x, y);
    }

    return STATUS_BAD_TENSOR_DTYPE;
}

__C __export infiniopStatus_t infiniopDestroyWhereDescriptor(infiniopWhereDescriptor_t desc) {
    delete[] ((_WhereDescriptor_t) desc)->output_shape;
    delete[] ((_WhereDescriptor_t) desc)->condition_strides;
    delete[] ((_WhereDescriptor_t) desc)->x_strides;
    delete[] ((_WhereDescriptor_t) desc)->y_strides;
    delete[] ((_WhereDescriptor_t) desc)->output_indices;

    delete (_WhereDescriptor_t) desc;
    return STATUS_SUCCESS;
}
