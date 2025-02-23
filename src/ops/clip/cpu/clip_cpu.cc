#include "clip_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateClipDescriptor(infiniopHandle_t, ClipCpuDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t output_desc,
                                         infiniopTensorDescriptor_t input_desc,
                                         infiniopTensorDescriptor_t min_desc,
                                         infiniopTensorDescriptor_t max_desc) {

    // 确定输入张量元素的类型满足要求
    if (!dtype_eq(input_desc->dt, F16) && !dtype_eq(input_desc->dt, F32)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    // 确定输出张量和输入张量的类型是否一致
    if (!dtype_eq(output_desc->dt, input_desc->dt)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    // 确定输入和输出张量的 strides 是否合法
    if (!is_contiguous(output_desc) || !is_contiguous(input_desc)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    // 确定输入张量和输出张量的维度是否一致
    if (output_desc->ndim != input_desc->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    // 确定输入张量和输出张量的形状是否一致
    for (uint64_t i = 0; i < output_desc->ndim; ++i) {
        if (output_desc->shape[i] != input_desc->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    // 确定 min_desc 和 max_desc 是否为空（两者均可以为空），如果非空，判断是否为标量或类型是否满足要求
    bool min_is_null = false;
    bool max_is_null = false;
    if (min_desc == nullptr) {
        min_is_null = true;
    } else {
        if (min_desc->ndim != 0) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
        if (!dtype_eq(min_desc->dt, input_desc->dt)) {
            return STATUS_BAD_TENSOR_DTYPE;
        }
    }
    if (max_desc == nullptr) {
        max_is_null = true;
    } else {
        if (max_desc->ndim != 0) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
        if (!dtype_eq(max_desc->dt, input_desc->dt)) {
            return STATUS_BAD_TENSOR_DTYPE;
        }
    }

    // 计算创建 ClipDescriptor 所需的参数
    uint64_t n_dim = output_desc->ndim;
    uint64_t data_size =
        std::accumulate(output_desc->shape, output_desc->shape + n_dim, 1ULL, std::multiplies<uint64_t>());

    // 创建 ClipDescriptor
    *desc_ptr = new ClipCpuDescriptor{
        DevCpu,
        input_desc->dt,
        n_dim,
        data_size,
        min_is_null,
        max_is_null};

    return STATUS_SUCCESS;
}

/**
 * @brief 实际执行裁剪操作的函数，可以根据模板实参，分别对 F16 和 F32 类型的数据执行裁剪
 * @tparam Tdata 要 clip 张量中的元素数据类型
 * @param desc ClipCpuDescriptor_t 类型的指针，指向裁剪操作的描述符
 * @param output 裁剪结果保存的位置
 * @param input 输入张量
 * @param min 裁剪的最小值
 * @param max 裁剪的最大值
 * @return
 */
template <typename Tdata>
infiniopStatus_t clip_cpu(ClipCpuDescriptor_t desc, void *output, void const *input, void const *min,
                          void const *max) {

    // 先将数据转换为对应类型的指针
    Tdata *output_data = reinterpret_cast<Tdata *>(output);
    Tdata const *input_data = reinterpret_cast<Tdata const *>(input);

    // 如果 min 和 max 都是空，直接复制
    if (desc->min_is_null && desc->max_is_null) {
        std::memcpy(output_data, input_data, desc->data_size * desc->dtype.size);
        return STATUS_SUCCESS;
    }

    // 根据 min 和 max 是否为空决定是否需要转换裁剪的最小值和最大值
    Tdata min_value;
    Tdata max_value;
    if (!desc->min_is_null) {
        min_value = *reinterpret_cast<Tdata const *>(min);
    }
    if (!desc->max_is_null) {
        max_value = *reinterpret_cast<Tdata const *>(max);
    }
    // 遍历输入数据中的每个元素，执行裁剪操作
    for (uint64_t i = 0; i < desc->data_size; ++i) {
        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            // 由于 F16 是用 uint16_t 表示的，所以不能使用 std::numeric_limits<Tdata> 获取 Tdata
            // 类型的最大值和最小值，需要根据不同情况进行比较
            if (!desc->min_is_null && !desc->max_is_null) {
                output_data[i] = f32_to_f16(std::min(
                    std::max(f16_to_f32(input_data[i]), f16_to_f32(min_value)), f16_to_f32(max_value)));
            } else if (desc->min_is_null && !desc->max_is_null) {
                output_data[i] = f32_to_f16(std::min(f16_to_f32(input_data[i]), f16_to_f32(max_value)));
            } else if (!desc->min_is_null && desc->max_is_null) {
                output_data[i] = f32_to_f16(std::max(f16_to_f32(input_data[i]), f16_to_f32(min_value)));
            }
        } else {
            if (!desc->min_is_null && !desc->max_is_null) {
                output_data[i] = std::min(std::max(input_data[i], min_value), max_value);
            } else if (desc->min_is_null && !desc->max_is_null) {
                output_data[i] = std::min(input_data[i], max_value);
            } else if (!desc->min_is_null && desc->max_is_null) {
                output_data[i] = std::max(input_data[i], min_value);
            }
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuClip(ClipCpuDescriptor_t desc, void *output, void const *input, void const *min,
                         void const *max, void *stream) {
    // 根据不同类型的张量类型，给 clip_cpu 函数传递不同的函数实参
    if (desc->dtype == F16) {
        return clip_cpu<uint16_t>(desc, output, input, min, max);
    }
    if (desc->dtype == F32) {
        return clip_cpu<float>(desc, output, input, min, max);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}

infiniopStatus_t cpuDestroyClipDescriptor(ClipCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}