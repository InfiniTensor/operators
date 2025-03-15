#ifndef __CPU_CLIP_H__
#define __CPU_CLIP_H__

#include "operators.h"

/**
 * @brief CPU 裁剪操作的描述符
 */
struct ClipCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t ndim; // 结果张量的维度
    uint64_t data_size; // 结果张量的元素数量
    bool min_is_null; // 最小值是否为空
    bool max_is_null; // 最大值是否为空
};
typedef struct ClipCpuDescriptor *ClipCpuDescriptor_t;

/**
 * @brief 创建一个用于对张量执行 CPU 裁剪操作的描述符
 * @param desc_ptr CPU 裁剪操作描述符的指针
 * @param output_desc 输出张量描述符
 * @param input_desc 输入张量描述符
 * @param min_desc 每个元素的最小值描述符
 * @param max_desc 每个元素的最大值描述符
 * @return 标识是否创建成功
 */
infiniopStatus_t cpuCreateClipDescriptor(infiniopHandle_t,
                                         ClipCpuDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t output_desc,
                                         infiniopTensorDescriptor_t input_desc,
                                         infiniopTensorDescriptor_t min_desc,
                                         infiniopTensorDescriptor_t max_desc);


/**
 * @brief 执行 CPU 上的裁剪操作
 * @param desc ClipCpuDescriptor_t 类型的指针，指向裁剪操作的描述符
 * @param output 裁剪结果保存的位置
 * @param input 输入张量
 * @param min 每个元素的最小值
 * @param max 每个元素的最大值
 * @param stream 未使用
 * @return 
 */
infiniopStatus_t cpuClip(ClipCpuDescriptor_t desc,
                         void *output, void const *input,
                         void const *min, void const *max,
                         void *stream);

/**
 * @brief 销毁指定的 CPU 裁剪操作描述符
 * @param desc 要销毁的裁剪操作描述符
 * @return 返回表示是否销毁成功的状态
 */
infiniopStatus_t cpuDestroyClipDescriptor(ClipCpuDescriptor_t desc);

#endif