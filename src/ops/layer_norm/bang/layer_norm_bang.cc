#include "layer_norm_bang.h"
#include "../../utils.h"
infiniopStatus_t bangCreateLayerNormDescriptor(BangHandle_t handle, LayerNormBangDescriptor_t *desc_ptr,                                            
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
    *desc_ptr = new LayerNormBangDescriptor{
        handle->device,
        handle->device_id,
        x_desc->dt,
        size,
        behindsize,
        epsilon};

    return STATUS_SUCCESS;
}
infiniopStatus_t bangGetLayerNormWorkspaceSize(LayerNormBangDescriptor_t desc, unsigned long int *size) {
    *size = 32 * sizeof(desc->dtype);//taskDim * sizeof(T),taskDim不超过32
    return STATUS_SUCCESS;
}

infiniopStatus_t bangDestroyLayerNormDescriptor(LayerNormBangDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
