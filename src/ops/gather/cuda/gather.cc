#include "gather.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateGatherDescriptor(CudaHandle_t handle,
    GatherCudaDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t dst,
    infiniopTensorDescriptor_t data,
    infiniopTensorDescriptor_t indices,
    int axis) {
    uint64_t data_ndim = data->ndim;
    uint64_t indices_ndim = indices->ndim;
    if(axis<0){
        axis += data_ndim;
    }
    if(axis >= data_ndim){
        return STATUS_BAD_PARAM;
    }
    
    int j=0;
    for (int i=0; i<dst->ndim; i++){
        if(i<axis){
            if (data->shape[i] != dst->shape[i]){
                return STATUS_BAD_TENSOR_SHAPE;
            }
        }
        else if(i == axis){
            for(j=0; j<indices_ndim; j++){
                if (indices->shape[j] != dst->shape[i]){
                    return STATUS_BAD_TENSOR_SHAPE;
                }
                i++;
            }
        }
        else{
            if (data->shape[i - j] != dst->shape[i]){
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
    uint64_t dst_size = std::accumulate(dst->shape, dst->shape + dst->ndim, 1ULL, std::multiplies<uint64_t>());

    uint64_t stride_axis = (data->strides)[axis];
    uint64_t shape_axis = (data->shape)[axis];
    uint64_t num_indices = std::accumulate(indices->shape, indices->shape + indices->ndim, 1ULL, std::multiplies<uint64_t>());

    int64_t *stride_axis_ptr, *shape_axis_ptr;
    uint64_t *num_indices_ptr;
    checkCudaErrorWithCode(cudaMalloc(&stride_axis_ptr, sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc(&shape_axis_ptr, sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc(&num_indices_ptr, sizeof(uint64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMemcpy(stride_axis_ptr, &stride_axis, sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMemcpy(shape_axis_ptr, &shape_axis, sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMemcpy(num_indices_ptr, &num_indices, sizeof(uint64_t), cudaMemcpyHostToDevice), STATUS_MEMORY_NOT_ALLOCATED);
 

    *desc_ptr = new GatherCudaDescriptor{
    DevNvGpu,
    dtype,
    indice_type,
    handle->device_id,
    data_ndim,
    dst_size,
    stride_axis_ptr,
    shape_axis_ptr,
    num_indices_ptr,
    static_cast<uint64_t>(handle->prop.maxGridSize[0]),
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyGatherDescriptor(GatherCudaDescriptor_t desc) {
    checkCudaErrorWithCode(cudaFree(desc->stride_axis_ptr), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree(desc->shape_axis_ptr), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree(desc->num_indices_ptr), STATUS_EXECUTION_FAILED);
    delete desc;
    return STATUS_SUCCESS;
}