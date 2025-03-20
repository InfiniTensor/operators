#include "where_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateWhereDescriptor(infiniopHandle_t handle,
    WhereCpuDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t dst,
    infiniopTensorDescriptor_t src1,
    infiniopTensorDescriptor_t src2,
    infiniopTensorDescriptor_t condition
    ){
    if (!isValidBroadcastShape(dst, src1) || !isValidBroadcastShape(dst, src2) || !isValidBroadcastShape(dst, condition)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (dst->dt != F16 && dst->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (src1->dt != F16 && src1->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (src2->dt != F16 && src2->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (condition->dt != U8) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (!is_contiguous(dst) || !is_contiguous(src1) || !is_contiguous(src2) || !is_contiguous(condition)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (dst->dt != src1->dt || dst->dt != src2->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    uint64_t *src1_shape = new uint64_t[dst->ndim];
    uint64_t *src2_shape = new uint64_t[dst->ndim];
    uint64_t *condition_shape = new uint64_t[dst->ndim];
    uint64_t *dst_shape = new uint64_t[dst->ndim];

    int64_t *dst_strides = new int64_t[dst->ndim];
    int64_t *src1_strides = new int64_t[src1->ndim];
    int64_t *src2_strides = new int64_t[src2->ndim];
    int64_t *condition_strides = new int64_t[condition->ndim];
    uint64_t dst_ndim = dst->ndim;
    uint64_t element_num = 1;
    for (uint64_t i = 0; i < dst->ndim; i++) {
        element_num *= dst->shape[i];
    }

    memcpy(src1_shape, src1->shape, src1->ndim * sizeof(uint64_t));
    memcpy(src2_shape, src2->shape, src2->ndim * sizeof(uint64_t));
    memcpy(condition_shape, condition->shape, condition->ndim * sizeof(uint64_t));
    memcpy(dst_shape, dst->shape, dst->ndim * sizeof(uint64_t));

    memcpy(dst_strides, dst->strides, dst->ndim * sizeof(int64_t));
    memcpy(src1_strides, src1->strides, src1->ndim * sizeof(int64_t));
    memcpy(src2_strides, src2->strides, src2->ndim * sizeof(int64_t));
    memcpy(condition_strides, condition->strides, condition->ndim * sizeof(int64_t));
    *desc_ptr = new WhereCpuDescriptor{
        handle->device,
        dst->dt,
        src1_shape,
        src2_shape,
        condition_shape,
        dst_shape,
        dst_strides,
        src1_strides,
        src2_strides,
        condition_strides,
        dst_ndim,
        src1->ndim,
        src2->ndim,
        condition->ndim,
        element_num
    };
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyWhereDescriptor(WhereCpuDescriptor_t desc){
    delete[] desc->src1_shape;
    delete[] desc->src2_shape;
    delete[] desc->condition_shape;
    delete[] desc->dst_shape;
    delete[] desc->dst_strides;
    delete[] desc->src1_strides;
    delete[] desc->src2_strides;
    delete[] desc->condition_strides;
    delete desc;
    return STATUS_SUCCESS;
}
inline uint64_t broadcast_map(
    uint64_t idx, 
    const uint64_t* dst_shape,     
    uint64_t dst_ndim,             
    const uint64_t* input_shape,   
    const int64_t* input_strides,  
    uint64_t input_ndim            
) {
    uint64_t index = 0;
    const int offset = dst_ndim - input_ndim;
    for (int i = dst_ndim - 1; i >= 0; idx /= dst_shape[i--]) {
        const uint64_t coord = idx % dst_shape[i];
        if (i >= offset) { 
            const int dim = i - offset;
            const uint64_t size = input_shape[dim];
            index += (size == 1 ? 0 : coord % size) * input_strides[dim];
        }
    }
    return index;
}


template<typename Tdata>
infiniopStatus_t where_cpu(WhereCpuDescriptor_t desc,
    void *dst, 
    void const *src1,
    void const *src2,
    void const *condition,
    void *stream){
    auto dst_ = reinterpret_cast<Tdata *>(dst);
    auto src1_ = reinterpret_cast<const Tdata *>(src1);
    auto src2_ = reinterpret_cast<const Tdata *>(src2);
    auto condition_ = reinterpret_cast<const uint8_t *>(condition);
    #pragma omp parallel for
    for (uint64_t i = 0; i < desc->element_num; i++) {
        uint64_t condition_index = broadcast_map(i, desc->dst_shape, desc->dst_ndim, desc->condition_shape, desc->condition_strides, desc->condition_ndim);
        uint64_t src1_index = broadcast_map(i, desc->dst_shape, desc->dst_ndim, desc->src1_shape, desc->src1_strides, desc->src1_ndim);
        uint64_t src2_index = broadcast_map(i, desc->dst_shape, desc->dst_ndim, desc->src2_shape, desc->src2_strides, desc->src2_ndim);
        dst_[i] = condition_[condition_index] ? src1_[src1_index] : src2_[src2_index];
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuWhere(WhereCpuDescriptor_t desc,
    void *dst, 
    void const *src1,
    void const *src2,
    void const *condition,
    void *stream){
    if (desc->dtype == F16) {
        return where_cpu<uint16_t>(desc, dst, src1, src2, condition, stream);
    }
    if (desc->dtype == F32) {
        return where_cpu<float>(desc, dst, src1, src2, condition, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}