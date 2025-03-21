#include "gather_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

// inline void incrementOne(uint64_t *indices, uint64_t const *shape, uint64_t ndim) {
//     for (int64_t i = ndim - 1; i >= 0; --i) {
//         if (++indices[i] != shape[i]) {
//             return;
//         }
//         indices[i] = 0;
//     }
// }

inline uint64_t compactToFlat(uint64_t const *indices, uint64_t const *strides, uint64_t ndim) {
    return std::inner_product(indices, indices + ndim, strides, uint64_t(0));
}

infiniopStatus_t cpuCreateGatherDescriptor(infiniopHandle_t,
                                           GatherCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t output,
                                           infiniopTensorDescriptor_t data,
                                           infiniopTensorDescriptor_t indices,
                                           int64_t axis = 0) {
    // printf("\tcpuCreateGatherDescriptor\n");
    uint64_t indices_ndim = indices->ndim;// rank q
    uint64_t data_ndim = data->ndim;      // rank r
    uint64_t ndim = output->ndim;         // q+r-1

    if (!is_contiguous(data) || !is_contiguous(indices) || !is_contiguous(output)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (output->dt != F16 && output->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (output->dt != data->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (indices->dt != I32 && indices->dt != I64) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    // q+r-1
    if (ndim != indices_ndim + data_ndim - 1) {
        printf("\tseems wrong shape\n");
        return STATUS_BAD_TENSOR_SHAPE;
    }
    // [-r, r-1]
    // 0 < -2  || 0 > 1
    if (axis < (int64_t)(-data_ndim) || axis > (int64_t)(data_ndim - 1)) {
        // axis:0, data_ndim:2
        printf("\tseems wrong dim, axis:%ld, data_ndim:%ld\n", axis, data_ndim);
        if (axis > data_ndim - 1) {
            printf("\t%ld > %ld\n", axis, data_ndim - 1);
        }
        if (axis < 0-data_ndim) {
            printf("\t%ld < %ld\n", axis, -data_ndim);
        }
        return STATUS_BAD_PARAM;
    }
    axis = (axis + data_ndim) % data_ndim;
    // printf("\tcheck over\n");

    // uint64_t o_data_size = std::accumulate(output->shape, output->shape + output->ndim, 1ULL, std::multiplies<uint64_t>());

    // uint64_t *o_shape = new uint64_t[ndim];
    // std::copy(output->shape, output->shape + ndim, o_shape);

    uint64_t *data_shape = new uint64_t[data->ndim];
    memcpy(data_shape, data->shape, data->ndim * sizeof(uint64_t));
    
    uint64_t *indices_shape = new uint64_t[indices->ndim];
    memcpy(indices_shape, indices->shape, indices->ndim * sizeof(uint64_t));

    uint64_t *data_strides = new uint64_t[ndim];
    memcpy(data_strides, data->strides, data->ndim * sizeof(uint64_t));

    uint64_t *indices_strides = new uint64_t[ndim];
    memcpy(indices_strides, indices->strides, indices->ndim * sizeof(uint64_t));

    // uint64_t *o_indices = new uint64_t[ndim];
    // std::fill(o_indices, o_indices + ndim, 0);

    *desc_ptr = new GatherCpuDescriptor{
        DevCpu,
        output->dt,
        indices->dt,
        data_ndim,
        indices_ndim,
        // o_data_size,
        // o_shape,
        data_shape,
        indices_shape,
        data_strides,
        indices_strides,
        // o_indices,
        axis,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyGatherDescriptor(GatherCpuDescriptor_t desc) {
    // delete[] desc->o_shape;
    delete[] desc->data_shape;
    delete[] desc->indices_shape;
    delete[] desc->data_strides;
    delete[] desc->indices_strides;
    // delete[] desc->o_indices;
    delete desc;
    return STATUS_SUCCESS;
}

template<typename Tdata, typename Tindices>
infiniopStatus_t gather_cpu(GatherCpuDescriptor_t desc,
                            void *output,
                            void const *data,
                            void const *indices) {
    // printf("\tgather_cpu\n");
    auto data_ = reinterpret_cast<Tdata const *>(data);
    auto output_ = reinterpret_cast<Tdata *>(output);
    auto indices_ = reinterpret_cast<Tindices const *>(indices);
    const uint64_t axis = desc->axis;

    uint64_t outer_ = 1;
    for (uint64_t i = 0; i < axis; ++i) {
        outer_ *= desc->data_shape[i];  
    }
    uint64_t indices_element_count = 1;
    for (uint64_t i = 0; i < desc->indices_ndim; ++i) {
        indices_element_count *= desc->indices_shape[i];
    }
    uint64_t cpy_len = 1;
    for (uint64_t i = axis + 1; i < desc->data_ndim; ++i) {
        cpy_len *= desc->data_shape[i];
    }

    for (uint64_t outer = 0; outer < outer_; ++outer) {
        for (uint64_t idx = 0; idx < indices_element_count; ++idx) {
            Tindices index_val = indices_[idx];
            // bounds [-s, s-1]
            index_val = (index_val + desc->data_shape[axis]) % desc->data_shape[axis];

            const uint64_t data_offset = outer * desc->data_shape[axis] * cpy_len +
                                         index_val * cpy_len;
            const uint64_t output_offset = outer * indices_element_count * cpy_len +
                                           idx * cpy_len;
            memcpy(
                output_ + output_offset,
                data_ + data_offset,
                sizeof(Tdata) * cpy_len);
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuGather(GatherCpuDescriptor_t desc,
                           void *output,
                           void const *data,
                           void const *indices,
                           void *stream) {
    if (desc->dtype == F16 && desc->indices_dtype == I32) {
        return gather_cpu<uint16_t, int32_t>(desc, output, data, indices);
    }
    if (desc->dtype == F32 && desc->indices_dtype == I32) {
        return gather_cpu<float, int32_t>(desc, output, data, indices);
    }
    if (desc->dtype == F16 && desc->indices_dtype == I64) {
        return gather_cpu<uint16_t, int64_t>(desc, output, data, indices);
    }
    if (desc->dtype == F32 && desc->indices_dtype == I64) {
        return gather_cpu<float, int64_t>(desc, output, data, indices);
    }
    
    return STATUS_BAD_TENSOR_DTYPE;
}
