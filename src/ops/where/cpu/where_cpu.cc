#include "where_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateWhereDescriptor(infiniopHandle_t,
                                          WhereCpuDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t output,
                                          infiniopTensorDescriptor_t x,
                                          infiniopTensorDescriptor_t y,
                                          infiniopTensorDescriptor_t condition) {
    if (!isValidBroadcastShape(output, x) || !isValidBroadcastShape(output, y) || !isValidBroadcastShape(output, condition)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    uint64_t max_rank = std::max(std::max(x->ndim, y->ndim), condition->ndim);
    if (output->ndim != max_rank) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    uint64_t *valid_shape = new uint64_t[max_rank];
    uint64_t *x_padded_shape = new uint64_t[max_rank];
    uint64_t *y_padded_shape = new uint64_t[max_rank];
    uint64_t *condition_padded_shape = new uint64_t[max_rank];
    getBroadcastShape(x->shape, x->ndim, y->shape, y->ndim, condition->shape, condition->ndim, valid_shape, x_padded_shape, y_padded_shape, condition_padded_shape, max_rank);
    for (uint64_t i = 0; i < output->ndim; i++) {
        if (output->shape[i] != valid_shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }

    if (!is_contiguous(y) || !is_contiguous(x)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }

    if (output->dt != x->dt || output->dt != y->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    if (condition->dt != U8) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    uint64_t output_size = std::accumulate(output->shape, output->shape + output->ndim, 1ULL, std::multiplies<uint64_t>());

    uint64_t x_ndim = x->ndim;
    uint64_t y_ndim = y->ndim;
    uint64_t condition_ndim = condition->ndim;

    bool x_broadcast = false;
    for (uint64_t i = 0; i < max_rank; i++) {
        if (x_padded_shape[i] != valid_shape[i]) {
            x_broadcast = true;
            break;
        }
    }

    bool y_broadcast = false;
    for (uint64_t i = 0; i < max_rank; i++) {
        if (y_padded_shape[i] != valid_shape[i]) {
            y_broadcast = true;
            break;
        }
    }

    bool condition_broadcast = false;
    for (uint64_t i = 0; i < max_rank; i++) {
        if (condition_padded_shape[i] != valid_shape[i]) {
            condition_broadcast = true;
            break;
        }
    }

    uint64_t *x_shapes = new uint64_t[x_ndim];
    int64_t *x_strides = new int64_t[x_ndim];
    uint64_t *y_shapes = new uint64_t[y_ndim];
    int64_t *y_strides = new int64_t[y_ndim];
    uint64_t *condition_shapes = new uint64_t[condition_ndim];
    int64_t *condition_strides = new int64_t[condition_ndim];

    if (x_broadcast) {
        std::memcpy(x_shapes, x->shape, x_ndim * sizeof(uint64_t));
        std::memcpy(x_strides, x->strides, x_ndim * sizeof(int64_t));
    }
    if (y_broadcast) {
        std::memcpy(y_shapes, y->shape, y_ndim * sizeof(uint64_t));
        std::memcpy(y_strides, y->strides, y_ndim * sizeof(int64_t));
    }
    if (condition_broadcast) {
        std::memcpy(condition_shapes, condition->shape, condition_ndim * sizeof(uint64_t));
        std::memcpy(condition_strides, condition->strides, condition_ndim * sizeof(int64_t));
    }

    *desc_ptr = new WhereCpuDescriptor{
        DevCpu,
        output->dt,
        output_size,
        x_broadcast,
        x_ndim,
        x_shapes,
        x_strides,
        y_broadcast,
        y_ndim,
        y_shapes,
        y_strides,
        condition_broadcast,
        condition_ndim,
        condition_shapes,
        condition_strides,
    };

    delete[] valid_shape;
    delete[] x_padded_shape;
    delete[] y_padded_shape;
    delete[] condition_padded_shape;
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyWhereDescriptor(WhereCpuDescriptor_t desc) {
    delete desc->x_shapes;
    delete desc->x_strides;
    delete desc->y_shapes;
    delete desc->y_strides;
    delete desc->condition_shapes;
    delete desc->condition_strides;
    delete desc;
    return STATUS_SUCCESS;
}

inline uint64_t broadcast_map(uint64_t idx, uint64_t const *shapes, int64_t const *strides, uint64_t ndim) {
    uint64_t index = 0;
    for (uint64_t i = 0; i < ndim; i++) {
        index += (idx / strides[i]) % shapes[i] * strides[i];
    }
    return index;
}

template<typename Tdata>
infiniopStatus_t where_cpu(WhereCpuDescriptor_t desc, void *output, void const *x, void const *y, void const *condition) {

    auto x_ = reinterpret_cast<Tdata const *>(x);
    auto x_ndim = desc->x_ndim;
    auto x_shapes = desc->x_shapes;
    auto x_strides = desc->x_strides;
    auto y_ = reinterpret_cast<Tdata const *>(y);
    auto y_ndim = desc->y_ndim;
    auto y_shapes = desc->y_shapes;
    auto y_strides = desc->y_strides;
    auto condition_ = reinterpret_cast<uint8_t const *>(condition);
    auto condition_ndim = desc->condition_ndim;
    auto condition_shapes = desc->condition_shapes;
    auto condition_strides = desc->condition_strides;
    auto output_ = reinterpret_cast<Tdata *>(output);
    auto output_size_ = desc->output_size;

    bool x_broadcast = desc->x_broadcast;
    bool y_broadcast = desc->y_broadcast;
    bool condition_broadcast = desc->condition_broadcast;

#pragma omp parallel for
    for (uint64_t i = 0; i < output_size_; i++) {
        if (condition_broadcast) {
            uint64_t condition_idx = broadcast_map(i, condition_shapes, condition_strides, condition_ndim);
            output_[i] = condition_[condition_idx] != 0 ? x_broadcast ? x_[broadcast_map(i, x_shapes, x_strides, x_ndim)] : x_[i] : y_broadcast ? y_[broadcast_map(i, y_shapes, y_strides, y_ndim)]
                                                                                                                                                : y_[i];
        } else {
            output_[i] = condition_[i] != 0 ? x_broadcast ? x_[broadcast_map(i, x_shapes, x_strides, x_ndim)]
                                                          : x_[i]
                                            : y_broadcast ? y_[broadcast_map(i, y_shapes, y_strides, y_ndim)]
                                                          : y_[i];
        }
    }

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuWhere(WhereCpuDescriptor_t desc, void *output, void const *x, void const *y, void const *condition) {
    if (desc->dtype == F16) {
        return where_cpu<uint16_t>(desc, output, x, y, condition);
    }
    if (desc->dtype == F32) {
        return where_cpu<float>(desc, output, x, y, condition);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}