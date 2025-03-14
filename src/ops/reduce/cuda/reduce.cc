#include "reduce.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateReduceDescriptor(CudaHandle_t handle,
    ReduceCudaDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t dst,
    infiniopTensorDescriptor_t src,
    int *axis,
    int num_axis,
    int keep_dims,
    int reduce_type) {
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

    int64_t *shape_axis = new int64_t[n];
    int64_t *stride_axis = new int64_t[n];
    uint64_t add_dim = 1;
    for(int i = 0; i<n; i++){
        shape_axis[i] =  src->shape[axis[i]];
        add_dim *= shape_axis[i];
        stride_axis[i] = src->strides[axis[i]];
    }

    int64_t *stride_axis_ptr, *shape_axis_ptr;
    checkCudaErrorWithCode(cudaMalloc(&stride_axis_ptr, n * sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc(&shape_axis_ptr, n * sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMemcpy(stride_axis_ptr, stride_axis, n * sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMemcpy(shape_axis_ptr, shape_axis, n * sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_MEMORY_NOT_ALLOCATED);

    delete[] shape_axis;
    delete[] stride_axis;
    *desc_ptr = new ReduceCudaDescriptor{
    DevNvGpu,
    dtype,
    handle->device_id,
    ndim,
    dst_size,
    data_size,
    add_dim,
    n,
    stride_axis_ptr,
    shape_axis_ptr,
    reduce_type,
    static_cast<uint64_t>(handle->prop.maxGridSize[0]),
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaGetReduceWorkspaceSize(ReduceCudaDescriptor_t desc, uint64_t *size){
    *size  = 0;
    return STATUS_SUCCESS;
    
}

infiniopStatus_t cudaDestroyReduceDescriptor(ReduceCudaDescriptor_t desc) {
    checkCudaErrorWithCode(cudaFree(desc->stride_axis_ptr), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree(desc->shape_axis_ptr), STATUS_EXECUTION_FAILED);
    delete desc;
    return STATUS_SUCCESS;
}