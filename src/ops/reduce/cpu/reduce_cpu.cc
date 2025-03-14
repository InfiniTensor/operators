#include "reduce_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateReduceDescriptor(infiniopHandle_t handle,
                                           ReduceCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t dst,
                                           infiniopTensorDescriptor_t src,
                                           int *axis,
                                           const int num_axis,
                                           int keep_dims,
                                           int reduce_type){
    uint64_t ndim = src->ndim;
    int n = num_axis;

    for (int i=0; i<n; i++)
    {
        if(axis[i]<0){
            axis[i] += ndim;
        }
        if(axis[i] >= ndim){
            return STATUS_BAD_PARAM;
        }
        if(keep_dims == 1){
            if(dst->shape[axis[i]] != 1){
                return STATUS_BAD_TENSOR_SHAPE;
            }
        }
    }
    
    if (!is_contiguous(src) || !is_contiguous(dst)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (src->dt != dst->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (src->dt != F16 && src->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (reduce_type > 3) {
        return STATUS_BAD_PARAM;
    }
    DT dtype = src->dt;
    uint64_t dst_size = std::accumulate(dst->shape, dst->shape + dst->ndim, 1ULL, std::multiplies<uint64_t>());
    uint64_t data_size = std::accumulate(src->shape, src->shape + ndim, 1ULL, std::multiplies<uint64_t>());
    uint64_t *shape_axis = new uint64_t[n];
    uint64_t *stride_axis = new uint64_t[n];
    uint64_t add_dim = 1;
    for(int i = 0; i<n; i++){
        shape_axis[i] =  src->shape[axis[i]];
        add_dim *= shape_axis[i];
        stride_axis[i] = src->strides[axis[i]];
    }
    
    *desc_ptr = new ReduceCpuDescriptor{
        DevCpu,
        dtype,
        dst_size,
        data_size,
        stride_axis,
        shape_axis,
        add_dim,
        n,
        reduce_type,
    };
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuGetReduceWorkspaceSize(ReduceCpuDescriptor_t desc, uint64_t *size) {
    
    *size += desc->dst_size * sizeof(float);

    return STATUS_SUCCESS;
}


infiniopStatus_t cpuDestroyReduceDescriptor(ReduceCpuDescriptor_t desc) {
    delete[] desc->shape_axis;
    delete[] desc->stride_axis;
    delete desc;
    return STATUS_SUCCESS;
}

template <typename Tdata>
void reduce(ReduceCpuDescriptor_t desc, float *workspacePtr, Tdata *dstPtr, Tdata const *srcPtr, uint64_t i, uint64_t index){
    switch (desc->reduce_mode)
    {
    case 0:
        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            workspacePtr[index] = std::fmax(workspacePtr[index], f16_to_f32(srcPtr[i]));
            dstPtr[index] = f32_to_f16(workspacePtr[index]);
        } else {
            dstPtr[index] = std::max(dstPtr[index], srcPtr[i]);  
        }        
        break;
    case 1:
        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            workspacePtr[index] = std::fmin(workspacePtr[index], f16_to_f32(srcPtr[i]));
            dstPtr[index] = f32_to_f16(workspacePtr[index]);
        } else {
            dstPtr[index] = std::min(dstPtr[index], srcPtr[i]);  
        }  
        break;
    default:
        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            workspacePtr[index] += f16_to_f32(srcPtr[i]);
        }
        else{
            dstPtr[index] += srcPtr[i];
        }
        break;
    }
}

template <typename Tdata>
void reduce_cpu(ReduceCpuDescriptor_t desc, void *workspace, uint64_t workspace_size, void *dst, void const *src){
    auto dstPtr = reinterpret_cast<Tdata*>(dst);
    auto srcPtr = reinterpret_cast<Tdata const*>(src);
    auto workspacePtr = reinterpret_cast<float *>(workspace);
    
    if constexpr (std::is_same<Tdata, uint16_t>::value){
        switch (desc->reduce_mode)
        {
            case 0:
                std::fill(workspacePtr, workspacePtr + desc->dst_size, -std::numeric_limits<float>::infinity());
                break;
            case 1:
                std::fill(workspacePtr, workspacePtr + desc->dst_size, std::numeric_limits<float>::infinity());
                break;
            default:
                std::fill(workspacePtr, workspacePtr + desc->dst_size, 0);
                break;
        }
    }
    for(uint64_t i = 0; i < desc->data_size; i++){
        uint64_t index = i;
#pragma unroll  
        for(int j = 0; j < desc->axis_num; j++){
            index = index / (desc->stride_axis[j] * desc->shape_axis[j]) * desc->stride_axis[j] + index % desc->stride_axis[j];
        }
        reduce<Tdata>(desc, workspacePtr, dstPtr, srcPtr, i, index);
    }
    if(desc->reduce_mode == 2){
        for(int k = 0; k < desc->dst_size; k++){
            if constexpr (std::is_same<Tdata, uint16_t>::value) {
                dstPtr[k] = f32_to_f16(workspacePtr[k] / desc->add_dim);
            }
            else{
                dstPtr[k] /= desc->add_dim;
            }
        }
    }
    else if(desc->reduce_mode == 3){
        if constexpr (std::is_same<Tdata, uint16_t>::value){
            for(int k = 0; k < desc->dst_size; k++){
                dstPtr[k] = f32_to_f16(workspacePtr[k]);
            }
        }
    }

}

infiniopStatus_t cpuReduce(ReduceCpuDescriptor_t desc,
    void *workspace,
    uint64_t workspace_size,
    void *dst,     
    void const *src,
    void *stream){
    
    if(desc->dtype == F32)
    {
        reduce_cpu<float>(desc, workspace, workspace_size, dst, src);
    }
    else if(desc->dtype == F16)
    {
        reduce_cpu<uint16_t>(desc, workspace, workspace_size, dst, src);
    }    
    return STATUS_SUCCESS;        
}