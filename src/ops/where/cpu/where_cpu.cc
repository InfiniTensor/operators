#include "where_cpu.h"
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

infiniopStatus_t cpuCreateWhereDescriptor(infiniopHandle_t,
                                          WhereCpuDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t output,
                                          infiniopTensorDescriptor_t condition,
                                          infiniopTensorDescriptor_t x,
                                          infiniopTensorDescriptor_t y) {
    uint64_t ndim = output->ndim;

    if (!isValidBroadcastShape(x, y, output)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!is_contiguous(x) || !is_contiguous(y) || !is_contiguous(output)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (output->dt != F16 && output->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (output->dt != x->dt || output->dt != y->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (condition->dt != U8) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    uint64_t o_data_size = std::accumulate(
        output->shape, output->shape + output->ndim, 1ULL, std::multiplies<uint64_t>());
    
    uint64_t *o_shape = new uint64_t[ndim];
    std::copy(output->shape, output->shape + ndim, o_shape);

    uint64_t *x_strides = new uint64_t[ndim];
    uint64_t *y_strides = new uint64_t[ndim];
    for (size_t i = 0; i < ndim; ++i) {
        x_strides[i] = (i < ndim - x->ndim || output->shape[i] != x->shape[i + x->ndim - ndim]) ? 0 : x->strides[i + x->ndim - ndim];
        y_strides[i] = (i < ndim - y->ndim || output->shape[i] != y->shape[i + y->ndim - ndim]) ? 0 : y->strides[i + y->ndim - ndim];
    }

    uint64_t *o_indices = new uint64_t[ndim];
    std::fill(o_indices, o_indices + ndim, 0);

    *desc_ptr = new WhereCpuDescriptor{
        DevCpu,
        output->dt,
        ndim,
        o_data_size,
        o_shape,
        x_strides,
        y_strides,
        o_indices,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyWhereDescriptor(WhereCpuDescriptor_t desc) {
    delete[] desc->o_shape;
    delete[] desc->x_strides;
    delete[] desc->y_strides;
    delete[] desc->o_indices;
    delete desc;
    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t clip_cpu(WhereCpuDescriptor_t desc,
                          void *output,
                          void const *condition,
                          void const *x,
                          void const *y) {
    auto x_ = reinterpret_cast<Tdata const *>(x);
    auto y_ = reinterpret_cast<Tdata const *>(y);
    auto condition_ = reinterpret_cast<uint8_t const *>(condition);

    auto o_ = reinterpret_cast<Tdata *>(output);
    const auto &o_indices = desc->o_indices;

    for (uint64_t i = 0; i < desc->o_data_size; ++i, incrementOne(o_indices, desc->o_shape, desc->ndim)) {
        auto x_index = compactToFlat(o_indices, desc->x_strides, desc->ndim);
        auto y_index = compactToFlat(o_indices, desc->y_strides, desc->ndim);

        if (condition_[i]) {
            o_[i] = x_[x_index];
        } else {
            o_[i] = y_[y_index];
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuWhere(WhereCpuDescriptor_t desc,
                          void *output,
                          void const *condition,
                          void const *x,
                          void const *y,
                          void *stream) {
    if (desc->dtype == F16) {
        return clip_cpu<uint16_t>(desc, output, condition, x, y);
    }
    if (desc->dtype == F32) {
        return clip_cpu<float>(desc, output, condition, x, y);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
