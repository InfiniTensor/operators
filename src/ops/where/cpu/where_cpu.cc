#include "where_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <algorithm>

infiniopStatus_t cpuCreateWhereDescriptor(infiniopHandle_t handle,
                                          WhereCpuDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t dst,
                                          infiniopTensorDescriptor_t x,
                                          infiniopTensorDescriptor_t y,
                                          infiniopTensorDescriptor_t condition){
    uint64_t ndim = condition->ndim;
    if (ndim != x->ndim || ndim != y->ndim || ndim != dst->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }  
    for (size_t i = 0; i < ndim; ++i) {
        if (condition->shape[i] != x->shape[i] || condition->shape[i] != y->shape[i] || condition->shape[i] != dst->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (!is_contiguous(condition) || !is_contiguous(x) || !is_contiguous(y) || !is_contiguous(dst)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (x->dt != y->dt || x->dt != dst->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (x->dt != F16 && x->dt != F32 && x->dt != U16) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (condition->dt != U8) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    DT dtype = x->dt;
    uint64_t data_size = std::accumulate(condition->shape, condition->shape + ndim, 1ULL, std::multiplies<uint64_t>());
    *desc_ptr = new WhereCpuDescriptor{
        DevCpu,
        dtype,
        data_size,
    };
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyWhereDescriptor(WhereCpuDescriptor_t desc){
    delete desc;
    return STATUS_SUCCESS;    
}

template<typename Tdata>
void where_cpu(WhereCpuDescriptor_t desc, void *dst, void const *x, void const *y, void const *condition){
    auto dstPtr = reinterpret_cast<Tdata*>(dst);
    auto x_Ptr = reinterpret_cast<Tdata const*>(x);
    auto y_Ptr = reinterpret_cast<Tdata const*>(y);
    auto c_Ptr = reinterpret_cast<uint8_t const*>(condition);
    for(uint64_t i = 0; i < desc->data_size; i++){
        dstPtr[i] = (c_Ptr[i]) ? y_Ptr[i] : x_Ptr[i];
    }
}

infiniopStatus_t cpuWhere(WhereCpuDescriptor_t desc,
                          void *dst, 
                          void const *x,
                          void const *y,
                          void const *condition,
                          void *stream){
    if(desc->dtype == F32)
    {
        where_cpu<float>(desc, dst, x, y, condition);
    }
    else if(desc->dtype == F16)
    {
        where_cpu<uint16_t>(desc, dst, x, y, condition);
    }
    return STATUS_SUCCESS;    
}

