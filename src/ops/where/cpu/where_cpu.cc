#include "where_cpu.h"
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

infiniopStatus_t cpuCreateWhereDescriptor(infiniopHandle_t,
                                          WhereCpuDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t output_desc,
                                          infiniopTensorDescriptor_t condition_desc,
                                          infiniopTensorDescriptor_t x_desc,
                                          infiniopTensorDescriptor_t y_desc) {

    // 先执行检查
    // 1. 检查 condition 的类型是否为 bool（按照要求，bool 类型使用 uint8_t 表示）
    if (condition_desc->dt != U8) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    // 2. 确定 output 的张量类型是否为 F16 或 F32
    if (output_desc->dt != F16 && output_desc->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    // 3. 检查 x、y 的类型是否与 output 一致
    if (x_desc->dt != output_desc->dt || y_desc->dt != output_desc->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    // 4. 检查 x、y、output、condition 张量的 strides 是否都合法
    if (!is_contiguous(x_desc) ||
        !is_contiguous(y_desc) ||
        !is_contiguous(output_desc) ||
        !is_contiguous(condition_desc)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    // 5. 检查输出张量能否通过 x 和 y 广播后得到
    if (!isValidBroadcastShape(x_desc, y_desc, output_desc)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    
    // 计算创建 WhereDescriptor 所需的参数
    uint64_t ndim = output_desc->ndim;
    // 1. output_data_size
    uint64_t output_data_size = std::accumulate(output_desc->shape, output_desc->shape + ndim,
                                                1ULL, std::multiplies<uint64_t>());
    // 2. output_shape，应该利用 out_desc 中的 shape 信息创建副本
    uint64_t *output_shape = new uint64_t[ndim];
    std::copy(output_desc->shape, output_desc->shape + ndim, output_shape);
    // 3. x_strides 和 y_strides，使用广播后的形状计算，方便后续根据索引确定元素在实际一维数组中的位置
    uint64_t *x_strides = new uint64_t[ndim];
    uint64_t *y_strides = new uint64_t[ndim];
    for (size_t i = 0; i < ndim; ++i) {
        x_strides[i] =
            (i < ndim - x_desc->ndim || output_desc->shape[i] != x_desc->shape[i + x_desc->ndim - ndim])
                ? 0
                : x_desc->strides[i + x_desc->ndim - ndim];
        y_strides[i] =
            (i < ndim - y_desc->ndim || output_desc->shape[i] != y_desc->shape[i + y_desc->ndim - ndim])
                ? 0
                : y_desc->strides[i + y_desc->ndim - ndim];
    }
    // 4. output_indices，用于索引输出张量的每个元素，初始化为全 0，表示从第一个位置开始遍历计算
    uint64_t *output_indices = new uint64_t[ndim];
    std::fill(output_indices, output_indices + ndim, 0);

    // 分配内存并创建 WhereDescriptor
    *desc_ptr = new WhereCpuDescriptor {
        DevCpu,
        output_desc->dt,
        ndim,
        output_data_size,
        output_shape,
        x_strides,
        y_strides,
        output_indices
    };

    return STATUS_SUCCESS;
}

/**
 * @brief 为了可以根据元素类型直接获取对应类型的指针，因此设置一个模板函数，执行实际的操作
 * @tparam T 输入输出中元素类型
 * @param desc where 操作的描述符
 * @param output 输出张量
 * @param condition 条件张量
 * @param x 条件为真时的输入张量
 * @param y 条件为假时的输入张量
 * @param stream 未使用参数
 * @return 返回操作执行的结果
 */
template <typename T>
infiniopStatus_t where_cpu(WhereCpuDescriptor_t desc, void *output, void *condition, void *x, void *y,
  void *stream) {
    // 执行实际的 where 计算
    // 1. 先将参数转换为对应类型的指针
    auto x_data = reinterpret_cast<T const *>(x);
    auto y_data = reinterpret_cast<T const *>(y);
    auto condition_data = reinterpret_cast<uint8_t const *>(condition);
    auto output_data = reinterpret_cast<T *>(output);
    const auto &indices = desc->output_indices; // 用于遍历输出张量每个元素的索引
    
    // 2. 遍历每个元素，根据 condition 的值选择 x 或 y 的值，将结果写入 output_data
    for (uint64_t i = 0; i < desc->output_data_size; ++i, incrementOne(indices, desc->output_shape, desc->ndim)) {
        // 获取输入张量中索引对应的位置
        auto x_index = compactToFlat(indices, desc->x_strides, desc->ndim);
        auto y_index = compactToFlat(indices, desc->y_strides, desc->ndim);
        
        // for (int j = 0; j < desc->ndim; ++j) {
        //   std::cout << indices[j] << " ";
        // }
        // std::cout << ": " << (condition_data[i] ? "true" : "false") << " " << x_index << " " << y_index << std::endl;
        // 根据条件选择输入张量中的值
        output_data[i] = condition_data[i] ? x_data[x_index] : y_data[y_index];
    }
    return STATUS_SUCCESS;
  }


infiniopStatus_t cpuWhere(WhereCpuDescriptor_t desc, void *output, void *condition, void *x, void *y,
                          void *stream) {
    if (desc->dtype == F16) {
        return where_cpu<uint16_t>(desc, output, condition, x, y, stream);
    }
    if (desc->dtype == F32) {
        return where_cpu<float>(desc, output, condition, x, y, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}

infiniopStatus_t cpuDestroyWhereDescriptor(WhereCpuDescriptor_t desc) {
    delete[] desc->output_shape;
    delete[] desc->x_strides;
    delete[] desc->y_strides;
    delete[] desc->output_indices;
    delete desc;
    return STATUS_SUCCESS;
}