#include "gather_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateGatherDescriptor(infiniopHandle_t handle,
    GatherCpuDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t indices,
    int64_t axis
    ){
    if (y->dt != x->dt){
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != F16 && y->dt != F32){
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (!is_contiguous(y) || !is_contiguous(x)){
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (axis < 0 || axis >= x->ndim){
        return STATUS_BAD_PARAM;
    }
    uint64_t *dst_shape = new uint64_t[y->ndim];
    uint64_t *src_shape = new uint64_t[x->ndim];
    uint64_t *indices_shape = new uint64_t[indices->ndim];

    memcpy(dst_shape, y->shape, y->ndim * sizeof(uint64_t));
    memcpy(indices_shape, indices->shape, indices->ndim * sizeof(uint64_t));
    memcpy(src_shape, x->shape, x->ndim * sizeof(uint64_t));

    *desc_ptr = new GatherCpuDescriptor{
        DevCpu,
        y->dt,
        indices->dt,
        dst_shape,
        src_shape,
        indices_shape,
        indices->ndim,
        x->ndim,
        y->ndim,
        axis
    };
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyGatherDescriptor(GatherCpuDescriptor_t desc){
    delete[] desc->dst_shape;
    delete[] desc->src_shape;
    delete[] desc->indices_shape;
    delete desc;
    return STATUS_SUCCESS;
}

template<typename Tdata, typename Tindices>
infiniopStatus_t gather_cpu(GatherCpuDescriptor_t desc,
    void const *x, 
    void const *indices,
    void *y)
{
    auto *src_data = reinterpret_cast<const Tdata*>(x);
    auto *indices_data = reinterpret_cast<const Tindices*>(indices); // [!code ++]
    auto *dst_data = reinterpret_cast<Tdata*>(y);
    uint64_t indices_element_count = 1;
    for (uint64_t i = 0; i < desc->indices_ndim; ++i) {
        indices_element_count *= desc->indices_shape[i];
    }

    const uint64_t axis = desc->axis;
    uint64_t src_outer_dim = 1;
    for (uint64_t i = 0; i < axis; ++i) {
        src_outer_dim *= desc->src_shape[i];  
    }

    uint64_t src_inner_dim = 1;
    for (uint64_t i = axis + 1; i < desc->src_ndim; ++i) {
        src_inner_dim *= desc->src_shape[i];
    }
    for (uint64_t outer = 0; outer < src_outer_dim; ++outer) {
        for (uint64_t idx = 0; idx < indices_element_count; ++idx) {
            const int64_t index_val = indices_data[idx]; // [!code ++]
            const uint64_t src_offset = 
                outer * desc->src_shape[axis] * src_inner_dim + 
                index_val * src_inner_dim;
            const uint64_t dst_offset = 
                outer * indices_element_count * src_inner_dim + 
                idx * src_inner_dim;
            memcpy(
                dst_data + dst_offset,
                src_data + src_offset,
                sizeof(Tdata) * src_inner_dim
            );
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuGather(GatherCpuDescriptor_t desc,
    void const *x, 
    void const *indices,
    void *y,
    void *stream){
    if (desc->dtype == F16){
        if (desc->indices_dtype == I32){
            return gather_cpu<uint16_t, int32_t>(desc, x, indices, y);
        }
        else if (desc->indices_dtype == I64){
            return gather_cpu<uint16_t, int64_t>(desc, x, indices, y);
        }
    }
    if (desc->dtype == F32){
        if (desc->indices_dtype == I32){
            return gather_cpu<float, int32_t>(desc, x, indices, y);
        }
        else if (desc->indices_dtype == I64){
            return gather_cpu<float, int64_t>(desc, x, indices, y);
        }
    }
    return STATUS_SUCCESS;
}

