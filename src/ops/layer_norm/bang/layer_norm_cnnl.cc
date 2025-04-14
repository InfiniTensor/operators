#include "layer_norm_cnnl.h"
#include "../../utils.h"
infiniopStatus_t cnnlCreateLayerNormDescriptor(BangHandle_t handle, LayerNormCnnlDescriptor_t *desc_ptr,                                            
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
    int axis = ndim - wDim;
    std::vector<int> inDim(ndim);
    std::vector<int> outDim(ndim);
    std::vector<int> filter_biasDim(wDim);
    std::vector<int> mean_rstdDim(axis);
    size_t mean_rstd_size = 1;
    for (int i = 0; i < ndim; i++) {
        inDim[i] = static_cast<int>(x_desc->shape[i]);
        outDim[i] = static_cast<int>(x_desc->shape[i]);
        if(i >= axis){
            filter_biasDim[i - axis] = static_cast<int>(x_desc->shape[i]);           
        }
        else{
            mean_rstdDim[i] = static_cast<int>(x_desc->shape[i]);
            mean_rstd_size *= static_cast<size_t>(x_desc->shape[i]);
        }
    }
    size_t dtype_size = 0;
    cnnlTensorDescriptor_t yDesc, xDesc, filter_bias_desc, mean_rstd_desc;
    cnnlCreateTensorDescriptor(&yDesc);
    cnnlCreateTensorDescriptor(&xDesc);
    cnnlCreateTensorDescriptor(&filter_bias_desc);
    cnnlCreateTensorDescriptor(&mean_rstd_desc);
    
    if(dtype_eq(x_desc->dt, F16)){
        cnnlGetSizeOfDataType(CNNL_DTYPE_HALF, &dtype_size);
        cnnlSetTensorDescriptor(
            xDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
            inDim.size(), inDim.data());
        cnnlSetTensorDescriptor(
            yDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
            outDim.size(), outDim.data());
        cnnlSetTensorDescriptor(
            filter_bias_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
            filter_biasDim.size(), filter_biasDim.data());
        cnnlSetTensorDescriptor(
            mean_rstd_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
            mean_rstdDim.size(), mean_rstdDim.data());
    }
    else if(dtype_eq(x_desc->dt, F32)){
        cnnlGetSizeOfDataType(CNNL_DTYPE_FLOAT, &dtype_size);
        cnnlSetTensorDescriptor(
            xDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
            inDim.size(), inDim.data());
        cnnlSetTensorDescriptor(
            yDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
            outDim.size(), outDim.data());
        cnnlSetTensorDescriptor(
            filter_bias_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
            filter_biasDim.size(), filter_biasDim.data());
        cnnlSetTensorDescriptor(
            mean_rstd_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
            mean_rstdDim.size(), mean_rstdDim.data());
    }
    

    
    size_t size_mean_rstd = mean_rstd_size * dtype_size;
    size_t wsSize;
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    use_cnnl(handle->cnnl_handles, handle->device_id, queue,
             [&](cnnlHandle_t handle) {
                 cnnlGetLayerNormOpWorkspaceSize(handle, axis, xDesc, &wsSize);
             });
    CNRT_CHECK(cnrtQueueDestroy(queue));
    
    *desc_ptr = new LayerNormCnnlDescriptor{
        handle->device,
        handle->device_id,
        x_desc->dt,
        handle->cnnl_handles,
        xDesc,
        yDesc,
        filter_bias_desc,
        mean_rstd_desc,
        axis,
        size_mean_rstd,
        wsSize,
        epsilon};

    return STATUS_SUCCESS;
}
infiniopStatus_t cnnlGetLayerNormWorkspaceSize(LayerNormCnnlDescriptor_t desc, unsigned long int *size) {
    *size = 2 * desc->size_mean_rstd + desc->wsSize;
    return STATUS_SUCCESS;
}
template<typename T>
void layerNorm_cnnl(LayerNormCnnlDescriptor_t desc, void *workspace,
                                          uint64_t workspace_size,                         
                             void const *x, void const *w, void const *b, void *y, 
                             void *stream) {
    
    cnnlTensorDescriptor_t xDesc = desc->xDesc;
    cnnlTensorDescriptor_t yDesc = desc->yDesc;
    cnnlTensorDescriptor_t filter_bias_desc = desc->filter_bias_desc;
    cnnlTensorDescriptor_t mean_rstd_desc = desc->mean_rstd_desc;
    int axis = desc->axis;
    float eps = desc->epsilon;      

    T *mean_dev = reinterpret_cast<T *>(workspace);
    T *rstd_dev = mean_dev + desc->size_mean_rstd;
    
    void *workspace_extra = reinterpret_cast<char *>(workspace) + 2 * desc->size_mean_rstd;
    int wsSize = (int)workspace_size - 2 * desc->size_mean_rstd;
    use_cnnl(desc->cnnl_handles, desc->device_id, (cnrtQueue_t) stream,
             [&](cnnlHandle_t handle) {
                 cnnlLayerNormForward(handle,
                        xDesc,
                        x,
                        axis,
                        filter_bias_desc,
                        w,
                        b,
                        eps,
                        workspace_extra,
                        wsSize,
                        yDesc,
                        y,
                        mean_rstd_desc,
                        mean_dev,
                        rstd_dev);
             });
    
}
infiniopStatus_t cnnlLayerNorm(LayerNormCnnlDescriptor_t desc, void *workspace,
                                          uint64_t workspace_size,                       
                             void const *x, void const *w, void const *b, void *y, 
                             void *stream) {
    if (cnrtSetDevice(desc->device_id) != cnrtSuccess) {
        return STATUS_BAD_DEVICE;
    }

    if (dtype_eq(desc->dtype, F16)) {
        layerNorm_cnnl<uint16_t>(desc, workspace, workspace_size, x, w, b, y, stream);

        return STATUS_SUCCESS;
    }
    if (dtype_eq(desc->dtype, F32)) {
        layerNorm_cnnl<float>(desc, workspace, workspace_size, x, w, b, y, stream);

        return STATUS_SUCCESS;
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
infiniopStatus_t cnnlDestroyLayerNormDescriptor(LayerNormCnnlDescriptor_t desc) {
    desc->cnnl_handles = nullptr;
    cnnlDestroyTensorDescriptor(desc->xDesc);
    cnnlDestroyTensorDescriptor(desc->yDesc);
    cnnlDestroyTensorDescriptor(desc->filter_bias_desc);
    cnnlDestroyTensorDescriptor(desc->mean_rstd_desc);
    delete desc;
    return STATUS_SUCCESS;
}
