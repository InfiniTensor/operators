#include "gather_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
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

infiniopStatus_t cpuCreateGatherDescriptor(infiniopHandle_t,
                                           GatherCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t output_desc,
                                           infiniopTensorDescriptor_t data_desc,
                                           infiniopTensorDescriptor_t index_desc,
                                           int axis) {

    // 先检查输入数据是否合法
    // 0. 获取每个参数的秩序
    uint64_t output_ndim = output_desc->ndim;
    uint64_t data_ndim = data_desc->ndim;
    uint64_t index_ndim = index_desc->ndim;

    // 1. 检查所有张量的步长都合法
    if (!is_contiguous(output_desc) || !is_contiguous(data_desc) || !is_contiguous(index_desc)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    // 2. 检查 axis 的值是否在 [-ndata_ndim, data_ndim - 1] 范围内
    if (axis < -static_cast<int64_t>(data_ndim) || 
        axis >= static_cast<int64_t>(data_ndim)) {
        return STATUS_BAD_PARAM;
    }
    
    // 3. 检查输入和输出数据类型符合需要
    if (output_desc->dt != F16 && output_desc->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (data_desc->dt != output_desc->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    
    // 4. 检查索引数据类型符合要求
    if (index_desc->dt != I32 && index_desc->dt != I64) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    
    // 5. 检查 data 和 output 的秩合法
    if (data_ndim < 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (output_ndim != index_ndim + data_ndim - 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    // 计算创建 GatherCpuDescriptor 所需的相关数据
    // 1. 将负的 axis 转换为正数
    uint64_t data_axis = axis < 0 ? (axis + data_ndim) : axis;

    // 2. 计算形状
    // 计算 output_shape 并与输入的 output_desc 比较是否相同
    uint64_t *output_shape = new uint64_t[output_ndim];
    for (uint64_t i = 0, j = 0; i < output_ndim;) {
        // j 索引用于遍历 data 的维度
        if (j == data_axis) {
            // 当 j 遍历到 data 的第 axis 索引处时，将 index 的 shape 插入
            for (uint64_t k = 0; k < index_ndim; ++k) {
                output_shape[i++] = index_desc->shape[k];
            }
            ++j;
        } else {
            output_shape[i++] = data_desc->shape[j++];
        }
    }
    // 比较 output_shape 和 output_desc->shape 是否相同
    for (uint64_t i = 0; i < output_ndim; ++i) {
        if (output_shape[i] != output_desc->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }

    // 保存 data_shape
    uint64_t *data_shape = new uint64_t[data_ndim];
    std::copy(data_desc->shape, data_desc->shape + data_ndim, data_shape);

    // 3. 计算所有的元素个数 size
    uint64_t output_size = std::accumulate(output_shape, output_shape + output_ndim, uint64_t(1), std::multiplies<uint64_t>());
    uint64_t indices_size = std::accumulate(index_desc->shape, index_desc->shape + index_ndim, uint64_t(1), std::multiplies<uint64_t>());

    // 4. 保存 data 和 index 的步长
    uint64_t *data_strides = new uint64_t[data_ndim];
    std::copy(data_desc->strides, data_desc->strides + data_ndim, data_strides);
    uint64_t *index_strides = new uint64_t[index_ndim];
    std::copy(index_desc->strides, index_desc->strides + index_ndim, index_strides);

    // 5. 初始化所有的 indices
    uint64_t *output_indices = new uint64_t[output_ndim];
    std::fill(output_indices, output_indices + output_ndim, uint64_t(0));
    uint64_t *data_indices = new uint64_t[data_ndim];
    std::fill(data_indices, data_indices + data_ndim, uint64_t(0));
    uint64_t *index_indices = new uint64_t[index_ndim];
    std::fill(index_indices, index_indices + index_ndim, uint64_t(0));

    // 创建 GatherCpuDescriptor 描述符对象
    *desc_ptr = new GatherCpuDescriptor {
        DevCpu,
        output_desc->dt,
        index_desc->dt,
        output_ndim,
        output_size,
        output_shape,
        output_indices,
        data_ndim,
        data_shape,
        data_strides,
        data_indices,
        index_ndim,
        index_strides,
        index_indices,
        indices_size,
        data_axis
    };

    return STATUS_SUCCESS;
}

template <typename Tdata, typename Tindices>
infiniopStatus_t gather_cpu(GatherCpuDescriptor_t desc, void *output, void *data, void const *indices, void *stream) {
    auto input_data = reinterpret_cast<const Tdata *>(data);
    auto output_data = reinterpret_cast<Tdata *>(output);
    // 将输入的 indices 数据复制到新的数组中，以进行修改
    auto indices_data = reinterpret_cast<const Tindices *>(indices);

    // 取出 desc 中保存的相关信息
    uint64_t axis = desc->axis;
    uint64_t data_ndim = desc->data_ndim;
    auto output_indices = desc->output_indices;
    auto data_indices = desc->data_indices;
    auto index_indices = desc->index_indices;

    // 将对应索引的结果保存到 output 张量中
    for (uint64_t i = 0; i < desc->output_size;
         ++i, incrementOne(output_indices, desc->output_shape, desc->output_ndim)) {
        
        // 根据输出张量的 output_indices 索引确定在 data 中的索引
        for (uint64_t j = 0; j < desc->data_ndim; ++j) {
            if (j < axis) {
                // data 中小于 axis 部分的索引和 output_indices 中的索引相同
                data_indices[j] = output_indices[j];
            } else if (j == axis) {
                // 等于 axis 部分的索引需要从 indices_data 中获取
                for (uint64_t k = 0; k < desc->index_ndim; ++k) {
                    index_indices[k] = output_indices[k + axis];
                }
                // 根据 index_indices 索引获取 indices_data 中的值
                uint64_t cur_index_indices = compactToFlat(index_indices, desc->index_strides, desc->index_ndim);
                // 获取 cur_index_indices 处的值
                Tindices element_index = indices_data[cur_index_indices];
                // 处理负值的情况
                if (element_index < -static_cast<int64_t>(desc->data_shape[desc->axis]) ||
                    element_index >= static_cast<int64_t>(desc->data_shape[desc->axis])) {
                    return STATUS_BAD_PARAM;
                }
                if (element_index < 0) {
                    element_index += desc->data_shape[desc->axis];
                }
                data_indices[j] = element_index;
            } else {
                // 大于 axis 部分的索引为原有的索引，向后偏移 index_ndim - 1 个位置
                data_indices[j] = output_indices[j + desc->index_ndim - 1];
            }
        }

        // 根据 data_indices 索引获取 data 中的值
        uint64_t cur_data_indices = compactToFlat(data_indices, desc->data_strides, desc->data_ndim);
        output_data[i] = input_data[cur_data_indices];
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuGather(GatherCpuDescriptor_t desc, void *output, void *data, void const *indices, void *stream) {
    if (desc->output_dtype == F16) {
        if (desc->index_dtype == I32) {
            return gather_cpu<uint16_t, int32_t>(desc, output, data, indices, stream);
        }
        if (desc->index_dtype == I64) {
            return gather_cpu<uint16_t, int64_t>(desc, output, data, indices, stream);
        }
    }
    if (desc->output_dtype == F32) {
        if (desc->index_dtype == I32) {
            return gather_cpu<float, int32_t>(desc, output, data, indices, stream);
        }
        if (desc->index_dtype == I64) {
            return gather_cpu<float, int64_t>(desc, output, data, indices, stream);
        }
    }
    return STATUS_BAD_TENSOR_DTYPE;
}

infiniopStatus_t cpuDestroyGatherDescriptor(GatherCpuDescriptor_t desc) {
    delete desc;
    delete[] desc->output_shape;
    delete[] desc->output_indices;
    delete[] desc->data_shape;
    delete[] desc->data_strides;
    delete[] desc->data_indices;
    delete[] desc->index_strides;
    delete[] desc->index_indices;
    return STATUS_SUCCESS;
}