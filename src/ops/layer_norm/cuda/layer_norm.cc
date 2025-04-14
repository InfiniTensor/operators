#include "layer_norm.cuh"
#include "../../utils.h"
#include "../../../devices/cuda/common_cuda.h"

infiniopStatus_t cudaCreateLayerNormDescriptor(CudaHandle_t handle,
                                    LayerNormCudaDescriptor_t *desc_ptr,
                                    infiniopTensorDescriptor_t x_desc,
                                    infiniopTensorDescriptor_t w_desc,
                                    infiniopTensorDescriptor_t b_desc,
                                    infiniopTensorDescriptor_t y_desc,
                                    float epsilon) {
    if (w_desc->ndim != b_desc->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    int wDim = w_desc->ndim;
    for(int i = 0; i < wDim; i++){
        if(w_desc->shape[i] != b_desc->shape[i]){
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    int ndim = x_desc->ndim;
    for(int i = 0; i < wDim; i++){
        if(x_desc->shape[i + ndim - wDim] != w_desc->shape[i]){
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (!dtype_eq(x_desc->dt, F16) && !dtype_eq(x_desc->dt, F32)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    int size = 1;
    int behindsize = 1;
    for(int i = 0; i < ndim; i++){
        size *= static_cast<int>(x_desc->shape[i]);
        if(i >= ndim - wDim){
            behindsize *= static_cast<int>(x_desc->shape[i]);
        } 
    }
    *desc_ptr = new LayerNormCudaDescriptor{
        handle->device,
        handle->device_id,
        x_desc->dt,
        size,
        behindsize,
        epsilon};

    return STATUS_SUCCESS;
}
infiniopStatus_t cudaGetLayerNormWorkspaceSize(LayerNormCudaDescriptor_t desc, unsigned long int *size) {
    *size = 0;
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyLayerNormDescriptor(LayerNormCudaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
