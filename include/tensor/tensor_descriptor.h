#ifndef TENSOR_DESCRIPTOR_H
#define TENSOR_DESCRIPTOR_H

#include "../export.h"
#include "../tensor.h"
#include "../status.h"

/**
 * @brief 根据给定的参数，创建表示张量的描述符
 * @param desc_ptr 保存所创建的张量描述符
 * @param ndim 所表示张量的阶
 * @param shape_ 张量的形状
 * @param strides_ 张量的步长
 * @param datatype 张量元素的类型
 * @return 返回表示创建是否成功的状态码
 */
__C __export infiniopStatus_t infiniopCreateTensorDescriptor(infiniopTensorDescriptor_t *desc_ptr, uint64_t ndim, uint64_t const *shape_, int64_t const *strides_, DataLayout datatype);

__C __export infiniopStatus_t infiniopDestroyTensorDescriptor(infiniopTensorDescriptor_t desc);

#endif// TENSOR_DESCRIPTOR_H
