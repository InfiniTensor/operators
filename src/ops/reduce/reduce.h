#ifndef REDUCE_H
#define REDUCE_H

#include "export.h"
#include "operators.h"
#include <stdio.h>

typedef struct ReduceDescriptor {
    Device device;
} ReduceDescriptor;

typedef ReduceDescriptor *infiniopReduceDescriptor_t;

/**
 * @brief 根据传入的参数创建执行 Reduce 操作的描述符
 * @param handle infiniop 句柄
 * @param desc_ptr Reduce 描述符指针
 * @param reduced 结果张量描述符
 * @param data 输入张量描述符
 * @param axes 要进行 Reduce 的维度
 * @param keepdims 是否保留 Reduce 后的维度
 * @param noop_with_empty_axes 在 axes 为空时是否还执行 Reduce 操作
 * @param reduce_type 1: ReduceMax, 2: ReduceMin, 3: ReduceMean
 * @return 描述符是否创建成功
 */
__C infiniopStatus_t infiniopCreateReduceDescriptor(infiniopHandle_t handle,
                                                    infiniopReduceDescriptor_t *desc_ptr,
                                                    infiniopTensorDescriptor_t reduced,
                                                    infiniopTensorDescriptor_t data,
                                                    const int64_t *axes,
                                                    size_t axes_size,
                                                    int keepdims,
                                                    int noop_with_empty_axes,
                                                    int reduce_type);

__C infiniopStatus_t infiniopReduce(infiniopReduceDescriptor_t desc,
                                    void *reduced,
                                    void const *data,
                                    int64_t const *axes,
                                    void *stream);

__C infiniopStatus_t infiniopDestroyReduceDescriptor(infiniopReduceDescriptor_t desc);

#endif