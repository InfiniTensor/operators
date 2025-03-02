#include "reduce_cpu.h"
#include "../../utils.h"


/**
 * @brief 将 indices 表示的索引按照字典序递增 1，即在当前索引的基础上，转换为下一个索引
 * @param indices 当前的索引
 * @param shape 索引所确定张量的维度
 * @param ndim 张量的阶数
 */
inline void incrementOne(uint64_t *indices, uint64_t const *shape, uint64_t ndim) {
    // 每次优先从最后一维开始递增
    for (int64_t i = ndim - 1; i >= 0; --i) {
        // 如果当前维度递增后没有超过该维度的最大值，则直接递增并返回
        if (++indices[i] != shape[i]) {
            return;
        }
        // 如果递增后等于了最大值，则将该维度的索引置为 0，继续递增前一个维度
        indices[i] = 0;
    }
}

/**
 * @brief 根据给定的索引和步长，计算该索引确定的元素在一维数组中的位置
 * @param indices 给定的索引
 * @param strides 每个维度的偏移步长
 * @param ndim 索引的总维度
 * @return  返回该索引确定的元素在一维数组中的位置
 */
inline uint64_t compactToFlat(uint64_t const *indices, uint64_t const *strides, uint64_t ndim) {
    return std::inner_product(indices, indices + ndim, strides, uint64_t(0));
}


infiniopStatus_t cpuCreateReduceDescriptor(infiniopHandle_t handle,
                                           ReduceCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t reduced,
                                           infiniopTensorDescriptor_t data,
                                           int64_t const *axes,
                                           size_t axes_ndim,
                                           int keepdims,
                                           int noop_with_empty_axes,
                                           int reduce_type) {
    // 1. 检查 data 的类型和形状都正确
    if (data->dt != F16 && data->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (!is_contiguous(data)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    // 2. 检查 reduced 的类型和形状都正确
    if (reduced->dt != data->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    // 3. 如果要保留维度，reduced 应该可以广播到 data
    if (keepdims && !isValidBroadcastShape(data, reduced)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    // 4. 如果不保留维度
    if (keepdims == 0) {
        // 如果 axes 为空，noop_with_empty_axes 为 0，reduced 应该是个标量
        if (axes == nullptr && noop_with_empty_axes == 0 && reduced->ndim != 0) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
        // 如果 axes 非空，且 data 非标量时，reduced 应该是 data 去掉 axes 指定的维度
        if (axes != nullptr && data->ndim != 0 && reduced->ndim != data->ndim - axes_ndim) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    // 计算创建 ReduceCpuDescriptor 所需的参数
    // 1. 计算 data 的总元素个数
    uint64_t data_size =
        std::accumulate(data->shape, data->shape + data->ndim, uint64_t(1), std::multiplies<uint64_t>());
    // 2. 保存 data 的形状
    uint64_t *data_shape = new uint64_t[data->ndim];
    std::copy(data->shape, data->shape + data->ndim, data_shape);
    // 3. 保存 data 的步长
    uint64_t *data_strides = new uint64_t[data->ndim];
    std::copy(data->strides, data->strides + data->ndim, data_strides);
    // 4. 初始化 data_indices
    uint64_t *data_indices = new uint64_t[data->ndim];
    std::fill(data_indices, data_indices + data->ndim, 0);
    // 5. 保存 axes
    int64_t *axes_data = nullptr;
    if (axes != nullptr) {
        axes_data = new int64_t[axes_ndim];
        std::copy(axes, axes + axes_ndim, axes_data);
        // 将 axes_data 中的负数转换为正数
        for (size_t i = 0; i < axes_ndim; ++i) {
            if (axes_data[i] < -static_cast<int64_t>(data->ndim) ||
                axes_data[i] >= static_cast<int64_t>(data->ndim)) {
                return STATUS_BAD_PARAM;
            } else if (axes_data[i] < 0) {
                axes_data[i] += static_cast<int64_t>(data->ndim);
            }
        }
        // 对 axes_data 排序，方便后续遍历
        std::sort(axes_data, axes_data + axes_ndim);
    }
    // 6. 初始化 axes_indices 的索引
    uint64_t *axes_indices = new uint64_t[axes_ndim];
    std::fill(axes_indices, axes_indices + axes_ndim, 0);
    // 7. 初始化 axes_shape (axes 中所指定维度的 shape) 和 axes_size
    uint64_t *axes_shape = new uint64_t[axes_ndim];
    for (size_t i = 0; i < axes_ndim; ++i) {
        axes_shape[i] = data_shape[axes_data[i]];
    }
    u_int64_t axes_size =
        std::accumulate(axes_shape, axes_shape + axes_ndim, uint64_t(1), std::multiplies<uint64_t>());
    // 8. 初始化 reduced 的形状
    uint64_t *reduced_shape = new uint64_t[reduced->ndim];
    if (axes != nullptr && axes_ndim != 0) {
        // axes 非空时，根据是否 keepdims 设置 reduced 的形状
        if (keepdims == 0) {
            // 不保留维度时，reduced 的形状是 data 去掉 axes 指定的维度
            for (size_t i = 0, j = 0; i < data->ndim; ++i) {
                if (i == axes_data[j]) {
                    ++j;
                    continue;
                } else {
                    reduced_shape[i - j] = data_shape[i];
                }
            }
        } else {
            // 保留维度时，reduced 中 axes 指定的维度为 1
            for (size_t i = 0, j = 0; i < data->ndim; ++i) {
                if (i == axes_data[j]) {
                    reduced_shape[i] = 1;
                    ++j;
                } else {
                    reduced_shape[i] = data_shape[i];
                }
            }
        }
    } else {
        // axes 为空时，reduced 的形状根据 noop_with_empty_axes 设置
        if (noop_with_empty_axes == 1) {
            // noop_with_empty_axes 为 1，reduced 与 data 相同
            std::copy(data_shape, data_shape + data->ndim, reduced_shape);
        } else {
            // noop_with_empty_axes 为 0，表示对所有维度进行操作
            if (keepdims == 1) {
                // 保留维度时，reduced 每个维度都为 1
                std::fill(reduced_shape, reduced_shape + reduced->ndim, 1);
            }
            // 不保留维度时，reduced 为标量，其 shape 的形状为 0
        }
    }
    // 9. 初始时根据 reduced 的形状计算总元素个数
    uint64_t reduced_size =
        std::accumulate(reduced->shape, reduced->shape + reduced->ndim, uint64_t(1), std::multiplies<uint64_t>());
    // 10. 初始化 reduced_indices
    uint64_t *reduced_indices = new uint64_t[reduced->ndim];
    std::fill(reduced_indices, reduced_indices + reduced->ndim, 0);

    // 创建 ReduceCpuDescriptor
    *desc_ptr = new ReduceCpuDescriptor{
        DevCpu,
        data->dt,
        data->ndim,
        data_size,
        data_shape,
        data_strides,
        data_indices,
        axes_data,
        axes_indices,
        axes_shape,
        axes_size,
        axes_ndim,
        reduced->ndim,
        reduced_size,
        reduced_shape,
        reduced_indices,
        keepdims,
        noop_with_empty_axes,
        reduce_type,
    };
    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t reduce_cpu(ReduceCpuDescriptor_t desc,
                            void *reduced,
                            void const *data,
                            int64_t const *axes,
                            void *stream) {
    // 1. 先将输入转换为对应类型的数据
    auto reduced_data = reinterpret_cast<Tdata *>(reduced);
    auto data_data = reinterpret_cast<Tdata const *>(data);
    int64_t const *axes_data = desc->axes;

    // 2. 判断输入数据是否为空或标量
    if (desc->data_size == 1) {
        // 如果只有一个元素，表示是一个标量，直接复制返回
        reduced_data[0] = data_data[0];
        return STATUS_SUCCESS;
    } else if (desc->data_size == 0) {
        // 如果元素个数为 0，表示是空集合，应该根据类型，返回无穷大
        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            if (desc->reduce_type == 1) {
                // 返回 float32 的负无穷小
                reduced_data[0] = f32_to_f16(-std::numeric_limits<float>::infinity());
            } else if (desc->reduce_type == 2) {
                // 返回 float32 的正无穷大
                reduced_data[0] = f32_to_f16(std::numeric_limits<float>::infinity());
            } else if (desc->reduce_type == 3) {
                // 返回 float32 的 NaN
                reduced_data[0] = f32_to_f16(std::numeric_limits<float>::quiet_NaN());
            }
        } else {
            if (desc->reduce_type == 1) {
                // 返回 float32 的负无穷小
                reduced_data[0] = -std::numeric_limits<float>::infinity();
            } else if (desc->reduce_type == 2) {
                // 返回 float32 的正无穷大
                reduced_data[0] = std::numeric_limits<float>::infinity();
            } else if (desc->reduce_type == 3) {
                // 返回 float32 的 NaN
                reduced_data[0] = std::numeric_limits<float>::quiet_NaN();
            }
        }
        return STATUS_SUCCESS;
    }

    // 3. 判断是否返回原有数据
    if (axes_data == nullptr && desc->noop_with_empty_axes == 1) {
        // 如果 axes_data 为空且 noop_with_empty_axes 为 true，则将 data 直接复制到 reduced 中后返回
        std::copy(data_data, data_data + desc->data_size, reduced_data);
        return STATUS_SUCCESS;
    }
    
    //4. 判断是否是对所有维度执行操作
    bool is_reduce_all = false;
    if (axes_data == nullptr && desc->noop_with_empty_axes == 0) {
        // axes 为空且 noop_with_empty_axes 为 0 时，表示对所有维度进行操作
        is_reduce_all = true;
    } else if (desc->axes_ndim == desc->data_ndim) {
        // 如果 axes_data 的大小等于 data_ndim，则表示对所有维度执行操作
        is_reduce_all = true;
    }
    // 如果是对所有维度进行操作，直接遍历 data 的每个元素，计算后保存到 reduced_data[0] 中，直接返回
    if (is_reduce_all) {
        float res;
        for (size_t i = 0; i < desc->data_size; ++i) {
            if constexpr (std::is_same<Tdata, uint16_t>::value) {
                if (i == 0) {
                    res = f16_to_f32(data_data[i]);
                    continue;
                }
                if (desc->reduce_type == 1) { // 最大值
                    res = std::max(res, f16_to_f32(data_data[i]));
                } else if (desc->reduce_type == 2) { // 最小值
                    res = std::min(res, f16_to_f32(data_data[i]));
                } else if (desc->reduce_type == 3) { // 平均值
                    res += f16_to_f32(data_data[i]);
                }
            } else {
                if (i == 0) {
                    res = data_data[i];
                    continue;
                }
                if (desc->reduce_type == 1) { // 最大值
                    res = std::max(res, data_data[i]);
                } else if (desc->reduce_type == 2) { // 最小值
                    res = std::min(res, data_data[i]);
                } else if (desc->reduce_type == 3) { // 平均值
                    res += data_data[i];
                }
            }
        }
        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            if (desc->reduce_type == 3) { // 平均值
                res /= desc->data_size;
                reduced_data[0] = f32_to_f16(res);
            } else {
                reduced_data[0] = f32_to_f16(res);
            }
        } else {
            if (desc->reduce_type == 3) { // 平均值
                res /= desc->data_size;
                reduced_data[0] = res;
            } else {
                reduced_data[0] = res;
            }
        }
        return STATUS_SUCCESS;
    }
    
    // 5. 对指定的维度进行操作，遍历 reduced_data 的每个索引，根据索引，从 data_data 中获取对应元素
    const auto &reduced_indices = desc->reduced_indices;
    const auto &data_indices = desc->data_indices;
    for (size_t i = 0; i < desc->reduced_size;
         ++i, incrementOne(reduced_indices, desc->reduced_shape, desc->reduced_ndim)) {
        
        // 先将 reduced_indices 的非 reduce 索引保存到 data_indices 中
        if (desc->keepdims == 0) {
            // 不保存 reduce 的维度时，reduced_indices 是所有不在 axes 中的索引
            for (size_t j = 0, k = 0, l = 0; j < desc->data_ndim; ++j) {
                // l 遍历 axes_data，k 遍历 reduced_indices
                if (l < desc->axes_ndim && j == axes_data[l]) {
                    ++l;
                } else {
                    data_indices[j] = reduced_indices[k++];
                }
            }
        } else {
            // 保留维度时，reduced_indices 是所有维度的索引，可以直接复制
            std::copy(reduced_indices, reduced_indices + desc->reduced_ndim, data_indices);
        }

        // 遍历 axes 中指定的所有维度，与 reduced_indices 组合，构成 data_indices
        float res;
        const auto &axes_indices = desc->axes_indices;
        for (size_t j = 0; j < desc->axes_size;
             ++j, incrementOne(axes_indices, desc->axes_shape, desc->axes_ndim)) {
            // 将 axes_indices 的索引保存到 axes 指定的 data_indices 对应的索引中
            for (size_t k = 0; k < desc->axes_ndim; ++k) {
                data_indices[desc->axes[k]] = axes_indices[k];
            }
            // 根据 data_indices 计算 data_data 的索引
            size_t data_index = compactToFlat(data_indices, desc->data_strides, desc->data_ndim);

            if (j == 0) {
                // 如果是第一个元素，直接赋值给 res
                if constexpr (std::is_same<Tdata, uint16_t>::value) {
                    res = f16_to_f32(data_data[data_index]);
                } else {
                    res = data_data[data_index];
                }
                continue;
            }

            // 根据 reduce_type 计算 res
            if constexpr (std::is_same<Tdata, uint16_t>::value) {
                if (desc->reduce_type == 1) { // 最大值
                    res = std::max(res, f16_to_f32(data_data[data_index]));
                } else if (desc->reduce_type == 2) { // 最小值
                    res = std::min(res, f16_to_f32(data_data[data_index]));
                } else if (desc->reduce_type == 3) { // 平均值
                    res += f16_to_f32(data_data[data_index]);
                }
            } else {
                if (desc->reduce_type == 1) { // 最大值
                    res = std::max(res, data_data[data_index]);
                } else if (desc->reduce_type == 2) { // 最小值
                    res = std::min(res, data_data[data_index]);
                } else if (desc->reduce_type == 3) { // 平均值
                    res += data_data[data_index];
                }
            }
        }
        // 将 res 保存到 reduced_data 中
        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            if (desc->reduce_type == 3) { // 平均值
                res /= desc->axes_size;
                reduced_data[i] = f32_to_f16(res);
            } else {
                reduced_data[i] = f32_to_f16(res);
            }
        } else {
            if (desc->reduce_type == 3) { // 平均值
                res /= desc->axes_size;
                reduced_data[i] = res;
            } else {
                reduced_data[i] = res;
            }
        }
    }

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuReduce(ReduceCpuDescriptor_t desc,
                           void *reduced,
                           void const *data,
                           int64_t const *axes,
                           void *stream) {
    
      if (desc->dtype == F16) {
          return reduce_cpu<uint16_t>(desc, reduced, data, axes, stream);
      }
      if (desc->dtype == F32) {
          return reduce_cpu<float>(desc, reduced, data, axes, stream);
      }
      return STATUS_BAD_TENSOR_DTYPE;
}

infiniopStatus_t cpuDestroyReduceDescriptor(ReduceCpuDescriptor_t desc) {
    delete desc;
    delete[] desc->data_shape;
    delete[] desc->data_strides;
    delete[] desc->data_indices;
    delete[] desc->axes_shape;
    delete[] desc->axes;
    delete[] desc->axes_indices;
    delete[] desc->reduced_shape;
    delete[] desc->reduced_indices;
    return STATUS_SUCCESS;
}