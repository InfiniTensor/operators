#ifndef __CPU_GATHER_H__
#define __CPU_GATHER_H__

#include "operators.h"

/**
 * @brief CPU Gather 操作的描述符
 */
struct GatherCpuDescriptor {
    Device device;  // 设备类型
    DT output_dtype;  // 数据中元素的类型
    DT index_dtype;  // index 中元素的类型

    uint64_t output_ndim;  // 结果张量的维度
    uint64_t output_size;  // 结果张量的元素数量
    uint64_t const *output_shape;  // 结果张量的形状
    uint64_t *output_indices;  // 用于在执行计算时索引 output 的每个元素

    uint64_t data_ndim;  // 输入张量的维度
    uint64_t *data_shape; // 输入张量的形状
    uint64_t const *data_strides;  // 输入张量的偏移量步长（确定元素在张量中的位置）
    uint64_t *data_indices;  // 用于在计算时索引输入张量的每个元素

    uint64_t index_ndim;  // indices 张量的维度
    uint64_t const *index_strides;  // indices 张量的偏移量步长（确定元素在张量中的位置）
    uint64_t *index_indices;  // 用于在计算时索引 indices 张量的每个元素
    uint64_t indices_size;  // indices 张量的元素数量

    uint64_t axis;  // 索引的轴
};

/**
 * @brief CPU Gather 操作描述符的指针类型
 */
typedef struct GatherCpuDescriptor *GatherCpuDescriptor_t;

/**
 * @brief 创建用于对张量执行 Gather 操作的 CPU 描述符
 * @param handle infiniop 句柄
 * @param desc_ptr 指向内部创建的 Gather 描述符的指针
 * @param output_desc 输出张量的描述符
 * @param data_desc 所操作张量的描述符
 * @param index_desc indices 张量描述符
 * @param axis 聚集时的轴
 * @return 返回表示创建是否成功的状态
 */
infiniopStatus_t cpuCreateGatherDescriptor(infiniopHandle_t,
                                           GatherCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t output_desc,
                                           infiniopTensorDescriptor_t data_desc,
                                           infiniopTensorDescriptor_t index_desc,
                                           int axis);

/**
 * @brief 执行 Gather 操作
 * @param desc 指向内部创建的 Gather 描述符的指针
 * @param output 输出张量
 * @param data 所操作张量
 * @param indices indices 张量
 * @param stream 未使用参数
 * @return 返回操作执行的状态
 */
infiniopStatus_t cpuGather(GatherCpuDescriptor_t desc,
                           void *output,
                           void *data,
                           void const *indices,
                           void *stream);

/**
 * @brief 销毁指定的 Gather 操作描述符
 * @param desc 要销毁的 Gather 操作描述符
 * @return 返回表示是否销毁成功的状态
 */
infiniopStatus_t cpuDestroyGatherDescriptor(GatherCpuDescriptor_t desc);

#endif