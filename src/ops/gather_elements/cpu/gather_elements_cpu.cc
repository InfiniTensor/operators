#include "gather_elements_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"


infiniopStatus_t cpuCreateGatherElementsDescriptor(infiniopHandle_t,
    GatherElementsCpuDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t dst,
    infiniopTensorDescriptor_t data,
    infiniopTensorDescriptor_t indices,
    int axis)
{
    uint64_t data_ndim = data->ndim;
    uint64_t indices_ndim = indices->ndim;
    if (indices_ndim > data_ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if(axis<0){
        axis = axis + dst->ndim;
    }

    for(size_t i = 0; i < indices_ndim; i++){
        if(indices->shape[i] != dst->shape[i]){
            return STATUS_BAD_TENSOR_SHAPE;
        }
        if(i == axis){
            if (data->shape[i] < indices->shape[i]) {
                return STATUS_BAD_TENSOR_SHAPE;
            }            
        }
        else{
            if (data->shape[i] != indices->shape[i]) {
                return STATUS_BAD_TENSOR_SHAPE;
            }                
        }
    }
    if (data->dt != F16 && data->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (!is_contiguous(data)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (!is_contiguous(indices)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (indices->dt != I8 && indices->dt != I16 && indices->dt != I32 && indices->dt != I64) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    DT dtype = data->dt;
    DT indice_type = indices->dt;
    uint64_t indices_size = std::accumulate(indices->shape, indices->shape + indices->ndim, 1ULL, std::multiplies<uint64_t>());
    uint64_t stride_axis = (data->strides)[axis];
    uint64_t shape_axis = (data->shape)[axis];
    uint64_t num_indices = indices->shape[axis];
    *desc_ptr = new GatherElementsCpuDescriptor{
        DevCpu,
        dtype,
        indice_type,
        indices_size,
        stride_axis,
        shape_axis,
        num_indices,
    };
    return STATUS_SUCCESS;
}



infiniopStatus_t cpuDestroyGatherElementsDescriptor(GatherElementsCpuDescriptor_t desc){
    delete desc;
    return STATUS_SUCCESS;
}

template<typename Tdata, typename Idata>
void gather_cpu(GatherElementsCpuDescriptor_t desc,
    void *dst, 
    void const *data,
    void const *indices){
    auto dstPtr = reinterpret_cast<Tdata*>(dst);
    auto dataPtr = reinterpret_cast<Tdata const*>(data);
    auto indicePtr = reinterpret_cast<Idata const*>(indices);
    for(uint64_t i = 0; i < desc->indices_size; i++)
    {
        uint64_t index = indicePtr[i];
        int64_t linearId = desc->shape_axis * desc->stride_axis * (i / (desc->num_indices * desc->stride_axis)) + index * desc->stride_axis + i % desc->stride_axis;
        dstPtr[i] = dataPtr[linearId];
    }
}

infiniopStatus_t cpuGatherElements(GatherElementsCpuDescriptor_t desc,
    void *dst, 
    void const *data,
    void const *indices,
    void *stream){
    if(desc->dtype == F32 && desc->indice_type == I32)
    {
        gather_cpu<float, int32_t>(desc, dst, data, indices);
    }
    else if(desc->dtype == F16 && desc->indice_type == I32)
    {
        gather_cpu<uint16_t, int32_t>(desc, dst, data, indices);
    }
    else if(desc->dtype == F32 && desc->indice_type == I64)
    {
        gather_cpu<float, int64_t>(desc, dst, data, indices);
    }
    else if(desc->dtype == F16 && desc->indice_type == I64)
    {
        gather_cpu<uint16_t, int64_t>(desc, dst, data, indices);
    }

    return STATUS_SUCCESS;
}