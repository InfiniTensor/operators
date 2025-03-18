#include "gather_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateGatherDescriptor(infiniopHandle_t,
                                           GatherCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t output,
                                           infiniopTensorDescriptor_t input,
                                           infiniopTensorDescriptor_t indices,
                                           uint64_t axis) {
    if (indices->dt != I32 && indices->dt != I64) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (output->dt != input->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    if (output->ndim != input->ndim + indices->ndim - 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    for (int i = 0; i < output->ndim; i++) {
        if (i < axis) {
            if (output->shape[i] != input->shape[i]) {
                return STATUS_BAD_TENSOR_SHAPE;
            }
        } else if (i < axis + indices->ndim) {
            if (output->shape[i] != indices->shape[i - axis]) {
                return STATUS_BAD_TENSOR_SHAPE;
            }
        } else {
            if (output->shape[i] != input->shape[i - indices->ndim + 1]) {
                return STATUS_BAD_TENSOR_SHAPE;
            }
        }
    }

    if (!is_contiguous(output) || !is_contiguous(input) || !is_contiguous(indices)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }

    uint64_t axis_tmp = axis;
    if (axis_tmp < 0) {
        axis_tmp = output->ndim + axis;
    }

    uint64_t pre_size = std::accumulate(input->shape, input->shape + axis_tmp, 1ULL, std::multiplies<uint64_t>());
    uint64_t post_size = std::accumulate(input->shape + axis_tmp + 1, input->shape + input->ndim, 1ULL, std::multiplies<uint64_t>());
    uint64_t indices_size = std::accumulate(indices->shape, indices->shape + indices->ndim, 1ULL, std::multiplies<uint64_t>());
    uint64_t axis_size = input->shape[axis_tmp];

    *desc_ptr = new GatherCpuDescriptor{
        DevCpu,
        output->dt,
        indices->dt,
        pre_size,
        axis_size,
        indices_size,
        post_size,
    };
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyGatherDescriptor(GatherCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

template<typename Tdata, typename Tind>
infiniopStatus_t gather_cpu(GatherCpuDescriptor_t desc, void *output, void const *input, void const *indices) {
    auto input_ = reinterpret_cast<Tdata const *>(input);
    auto output_ = reinterpret_cast<Tdata *>(output);
    auto indices_ = reinterpret_cast<Tind const *>(indices);

    uint64_t pre_size = desc->pre_size;
    uint64_t post_size = desc->post_size;
    uint64_t indices_size = desc->indices_size;
    uint64_t axis_size = desc->axis_size;
    if (post_size == 1) {
#pragma omp parallel for collapse(2)
        for (uint64_t i = 0; i < pre_size; i++) {
            for (uint64_t j = 0; j < indices_size; j++) {
                uint64_t output_offset = i * indices_size * post_size + j * post_size;
                uint64_t input_offset = i * axis_size * post_size + indices_[j] * post_size;
                output_[output_offset] = input_[input_offset];
            }
        }
    } else {
#pragma omp parallel for collapse(2)
        for (uint64_t i = 0; i < pre_size; i++) {
            for (uint64_t j = 0; j < indices_size; j++) {
                uint64_t output_offset = i * indices_size * post_size + j * post_size;
                uint64_t input_offset = i * axis_size * post_size + indices_[j] * post_size;
                std::memcpy(output_ + output_offset, input_ + input_offset, post_size * sizeof(Tdata));
            }
        }
    }

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuGather(GatherCpuDescriptor_t desc, void *output, void const *input, void const *indices, void *stream) {
    if (desc->dtype == F16) {
        if (desc->indices_dtype == I32) {
            return gather_cpu<uint16_t, int32_t>(desc, output, input, indices);
        }
        if (desc->indices_dtype == I64) {
            return gather_cpu<uint16_t, int64_t>(desc, output, input, indices);
        }
    }
    if (desc->dtype == F32) {
        if (desc->indices_dtype == I32) {
            return gather_cpu<float, int32_t>(desc, output, input, indices);
        }
        if (desc->indices_dtype == I64) {
            return gather_cpu<float, int64_t>(desc, output, input, indices);
        }
    }
    return STATUS_BAD_TENSOR_DTYPE;
}