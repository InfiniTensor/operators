#include "reduce_cpu.h"
#include "../../utils.h"
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <unordered_set>

inline uint64_t getTotalSize(const uint64_t *arr, uint64_t ndim) {
    return std::accumulate(arr, arr + ndim, 1ULL, std::multiplies<uint64_t>());
}

infiniopStatus_t cpuCreateReduceDescriptor(infiniopHandle_t handle,
                                           ReduceCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t y,
                                           infiniopTensorDescriptor_t x,
                                           int64_t const *axes,
                                           uint64_t n_axes,
                                           int keep_dims,
                                           int reduce_type) {
    uint64_t x_ndim = x->ndim;
    uint64_t y_ndim = y->ndim;
    if (n_axes <= 0 || n_axes > x_ndim) {
        return STATUS_BAD_PARAM;
    }
    // shape
    if (keep_dims) {
        if (x_ndim != y_ndim) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    } else {
        if (x_ndim - n_axes != y_ndim) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (!is_contiguous(y) || !is_contiguous(x)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (reduce_type < 0 || reduce_type > 2) {
        return STATUS_BAD_PARAM;
    }
    // axes -r ~ r-1
    for (int i = 0; i < n_axes; i++) {
        if (axes[i] < -int64_t(x_ndim) || axes[i] >= int64_t(x_ndim)) {
            return STATUS_BAD_PARAM;
        }
    }

    const auto x_size = getTotalSize(x->shape, x->ndim);
    const auto y_size = getTotalSize(y->shape, y->ndim);

    uint64_t *x_shape = new uint64_t[x_ndim];
    uint64_t *y_shape = new uint64_t[y_ndim];
    memcpy(x_shape, x->shape, x_ndim * sizeof(uint64_t));
    memcpy(y_shape, y->shape, y_ndim * sizeof(uint64_t));

    int64_t *axes_ = new int64_t[n_axes];
    for (int i = 0; i < n_axes; i++) {
        axes_[i] = axes[i] >= 0 ? axes[i] : axes[i] + x_ndim;
    }

    *desc_ptr = new ReduceCpuDescriptor{
        DevCpu,
        y->dt,
        x->ndim,
        y_size,
        x_size,
        x_shape,
        y_shape,
        axes_,
        n_axes,
        keep_dims,
        reduce_type};

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyReduceDescriptor(ReduceCpuDescriptor_t desc) {
    delete[] desc->x_shape;
    delete[] desc->y_shape;
    delete[] desc->axes;
    delete desc;
    return STATUS_SUCCESS;
}

inline bool isOnReduceAxis(int64_t dim, int64_t const *axes, uint64_t n_axes) {
    for (size_t i = 0; i < n_axes; ++i) {
        if (dim == axes[i]) {
            return true;
        }
    }
    return false;
}

uint64_t flattenIndex(uint64_t const *indices, uint64_t const *shape, uint64_t ndim) {
    uint64_t idx = 0;
    uint64_t stride = 1;
    for (int64_t i = ndim - 1; i >= 0; --i) {
        idx += indices[i] * stride;
        stride *= shape[i];
    }
    return idx;
}

template<typename T>
void generateReduceIndices(uint64_t dim, uint64_t *curr_index, uint64_t const *x_shape,
                           int64_t const *axes, uint64_t n_axes, uint64_t ndim,
                           std::vector<uint64_t> &flat_indices) {
    if (dim == ndim) {
        flat_indices.push_back(flattenIndex(curr_index, x_shape, ndim));
        return;
    }

    if (isOnReduceAxis(dim, axes, n_axes)) {
        for (uint64_t i = 0; i < x_shape[dim]; ++i) {
            curr_index[dim] = i;
            generateReduceIndices<T>(dim + 1, curr_index, x_shape, axes, n_axes, ndim, flat_indices);
        }
    } else {
        generateReduceIndices<T>(dim + 1, curr_index, x_shape, axes, n_axes, ndim, flat_indices);
    }
}

template<typename T>
void getReduceIndices(uint64_t y_idx, uint64_t const *y_shape, uint64_t const *x_shape,
                      int64_t const *axes, uint64_t n_axes, uint64_t ndim, uint64_t y_ndim,
                      bool keep_dims, std::vector<uint64_t> &flat_indices) {
    // 将y_idx转换为多维索引
    std::vector<uint64_t> y_indices(y_ndim, 0);
    uint64_t temp = y_idx;
    for (int64_t i = y_ndim - 1; i >= 0; --i) {
        y_indices[i] = temp % y_shape[i];
        temp /= y_shape[i];
    }

    // 将y的多维索引映射到x的多维索引
    std::vector<uint64_t> x_indices(ndim, 0);
    uint64_t y_dim = 0;

    for (uint64_t x_dim = 0; x_dim < ndim; ++x_dim) {
        if (isOnReduceAxis(x_dim, axes, n_axes)) {
            if (keep_dims) {
                y_dim++;
            }
            x_indices[x_dim] = 0;
        } else {
            x_indices[x_dim] = y_indices[y_dim++];
        }
    }

    flat_indices.clear();
    generateReduceIndices<T>(0, x_indices.data(), x_shape, axes, n_axes, ndim, flat_indices);
}

template<typename T>
T performReduce(T const *x, const std::vector<uint64_t> &indices, int reduce_type) {
    if (indices.empty()) {
        return 0;
    }

    T result;
    switch (reduce_type) {
        case 0:// Max
            result = std::numeric_limits<T>::lowest();
            for (uint64_t idx : indices) {
                result = std::max(result, x[idx]);
            }
            break;
        case 1:// Min
            result = std::numeric_limits<T>::max();
            for (uint64_t idx : indices) {
                result = std::min(result, x[idx]);
            }
            break;
        case 2:// Mean
            result = 0;
            for (uint64_t idx : indices) {
                result += x[idx];
            }
            result /= static_cast<T>(indices.size());
            break;
        default:
            result = 0;
    }

    return result;
}

template<>
uint16_t performReduce<uint16_t>(uint16_t const *x, const std::vector<uint64_t> &indices, int reduce_type) {
    if (indices.empty()) {
        return 0;
    }

    float result;
    switch (reduce_type) {
        case 0:// Max
            result = -std::numeric_limits<float>::max();
            for (uint64_t idx : indices) {
                result = std::max(result, f16_to_f32(x[idx]));
            }
            break;
        case 1:// Min
            result = std::numeric_limits<float>::max();
            for (uint64_t idx : indices) {
                result = std::min(result, f16_to_f32(x[idx]));
            }
            break;
        case 2:// Mean
            result = 0;
            for (uint64_t idx : indices) {
                result += f16_to_f32(x[idx]);
            }
            result /= static_cast<float>(indices.size());
            break;
        default:
            result = 0;
    }

    return f32_to_f16(result);
}

template<typename T>
infiniopStatus_t reduce_cpu(ReduceCpuDescriptor_t desc, void *y, void const *x) {
    auto x_ = reinterpret_cast<T const *>(x);
    auto y_ = reinterpret_cast<T *>(y);

    std::vector<uint64_t> reduce_indices;

#pragma omp parallel for private(reduce_indices)
    for (uint64_t i = 0; i < desc->y_size; ++i) {
        // 获取当前输出索引对应的所有输入索引
        getReduceIndices<float>(i, desc->y_shape, desc->x_shape, desc->axes, desc->n_axes,
                            desc->ndim, desc->keep_dims ? desc->ndim : desc->ndim - desc->n_axes,
                            desc->keep_dims, reduce_indices);

        // reduce
        y_[i] = performReduce<T>(x_, reduce_indices, desc->reduce_type);
        if constexpr (std::is_same<T, uint16_t>::value) {
            y_[i] = performReduce<uint16_t>(x_, reduce_indices, desc->reduce_type);
        } else {
            y_[i] = performReduce<T>(x_, reduce_indices, desc->reduce_type);
        }
    }

    return STATUS_SUCCESS;
}


infiniopStatus_t cpuReduce(ReduceCpuDescriptor_t desc, void *y, void const *x, void *stream) {
    if (desc->dt == F16) {
        return reduce_cpu<uint16_t>(desc, y, x);
    } else if (desc->dt == F32) {
        return reduce_cpu<float>(desc, y, x);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
