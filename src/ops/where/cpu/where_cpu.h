#ifndef __CPU_WHERE_H__
#define __CPU_WHERE_H__

#include "operators.h"
#include <numeric>
#include <type_traits>

struct WhereCpuDescriptor {
    Device device;
    DT dtype; // 输入和输出张量中元素的类型
    uint64_t ndim; // 输出张量的阶数
    uint64_t output_data_size; // 输出张量的元素数量
    uint64_t const *output_shape; // 结果张量的形状（两个输入广播后的形状）
    uint64_t const *x_strides; // 第一个操作数的偏移量步长
    uint64_t const *y_strides; // 第二个操作数的偏移量步长
    uint64_t *output_indices; // 用于遍历结果张量时的索引数组
};

typedef struct WhereCpuDescriptor *WhereCpuDescriptor_t;

infiniopStatus_t cpuCreateWhereDescriptor(infiniopHandle_t,
                                          WhereCpuDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t output_desc,
                                          infiniopTensorDescriptor_t condition_desc,
                                          infiniopTensorDescriptor_t x_desc,
                                          infiniopTensorDescriptor_t y_desc);

infiniopStatus_t cpuWhere(WhereCpuDescriptor_t desc,
                          void *output, void *condition,
                          void *x, void *y,
                          void *stream);

infiniopStatus_t cpuDestroyWhereDescriptor(WhereCpuDescriptor_t desc);


#endif