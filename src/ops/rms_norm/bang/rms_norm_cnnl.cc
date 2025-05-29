#include "rms_norm_cnnl.h"
#include "../../../devices/bang/bang_handle.h"
#include "../../../devices/bang/common_bang.h"
#include "../../utils.h"

infiniopStatus_t cnnlCreateRMSNormDescriptor(BangHandle_t handle,
                                             RMSNormCnnlDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t y_desc,
                                             infiniopTensorDescriptor_t x_desc,
                                             infiniopTensorDescriptor_t w_desc,
                                             float epsilon) {
    if (y_desc->ndim != 2 || x_desc->ndim != 2 || w_desc->ndim != 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    auto n = y_desc->shape[0],
         d = y_desc->shape[1];

    if (x_desc->shape[0] != n || x_desc->shape[1] != d || w_desc->shape[0] != d) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    // Create tensor descriptors
    cnnlTensorDescriptor_t yDesc, xDesc, wDesc, wDesc_;
    cnnlCreateTensorDescriptor(&yDesc);
    cnnlCreateTensorDescriptor(&xDesc);
    cnnlCreateTensorDescriptor(&wDesc);
    
    // Set tensor descriptors
    setCnnlTensor(yDesc, y_desc);
    setCnnlTensor(xDesc, x_desc);
    setCnnlTensor(wDesc, w_desc);

    if (w_desc->dt == F32) {
        cnnlCreateTensorDescriptor(&wDesc_);
        TensorDescriptor w_ = {F16, w_desc->ndim, w_desc->shape, w_desc->strides};
        setCnnlTensor(wDesc_, &w_);
    }
    // Create and set RMSNorm descriptor
    cnnlFuseNormDescriptor_t opDesc;
    cnnlCreateFuseNormDescriptor(&opDesc);
    cnnlSetFuseNormDescriptor(opDesc, epsilon, 1.0, true,
                              false, false, false, false,
                              cnnlDataTypeConvert(y_desc->dt), 
                              CNNL_TRANSFORMER_RMSNORM);

    *desc_ptr = new RMSNormCnnlDescriptor{
        handle->device,
        handle->device_id,
        handle->cnnl_handles,
        y_desc->dt,
        w_desc->dt,
        yDesc,
        xDesc,
        wDesc,
        wDesc_,
        opDesc,
        n,
        d,
        epsilon};

    return STATUS_SUCCESS;
}

infiniopStatus_t cnnlGetRMSNormWorkspaceSize(RMSNormCnnlDescriptor_t desc, uint64_t *size) {
    size_t wsSize = 0;
    use_cnnl(desc->pool, desc->device_id, nullptr,
             [&](cnnlHandle_t handle) {
                 cnnlGetFuseNormWorkspaceSize(handle, desc->opDesc, desc->xDesc, &wsSize);
             });
    *size = static_cast<uint64_t>(wsSize);
    return STATUS_SUCCESS;
}

infiniopStatus_t cnnlRMSNorm(RMSNormCnnlDescriptor_t desc,
                             void *workspace,
                             uint64_t workspace_size,
                             void *y, void const *x, void const *w,
                             void *stream) {
    if (cnrtSetDevice(desc->device_id) != cnrtSuccess) {
        return STATUS_BAD_DEVICE;
    }

    void *work = nullptr;
    int sum = desc->n * desc->d * 2;
    cnrtMalloc(&work, sum * 2);
    use_cnnl(desc->pool, desc->device_id, (cnrtQueue_t)stream, [&](cnnlHandle_t handle) {
        cnnlCastDataType(handle, desc->wDesc, w, CNNL_CAST_FLOAT_TO_HALF, desc->wDesc_, work);
        });
    // if (dtype_eq(desc->w_datatype, F32)) {
    //     TensorDescriptor w = {F16, desc->wDesc->ndim, desc->wDesc->shape, desc->wDesc->strides};
    //     cnnlCreateTensorDescriptor(&wDesc_);
    //     std::vector<int> dims(&w->ndim);
    //     for (uint64_t i = 0; i < &w->ndim; i++) {
    //         dims[i] = static_cast<int>(&w->shape[i]);
    //         sum *= dims[i];
    //     }
    //     cnnlSetTensorDescriptor(wDesc_, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
    //                         dims.size(), dims.data());
    // }
    use_cnnl(desc->pool, desc->device_id, (cnrtQueue_t)stream,
             [&](cnnlHandle_t handle) {
                 cnnlFuseNorm(handle, 
                             desc->opDesc, 
                             desc->xDesc, x,
                             desc->w_datatype == F32 ? desc->wDesc_ : desc->wDesc, 
                             desc->w_datatype == F32 ? work : w,
                             nullptr, nullptr,
                             nullptr, nullptr,
                             nullptr, nullptr,
                             workspace, workspace_size,
                             desc->yDesc, y,
                             nullptr, nullptr);
             });

    return STATUS_SUCCESS;
}

infiniopStatus_t cnnlDestroyRMSNormDescriptor(RMSNormCnnlDescriptor_t desc) {
    cnnlDestroyTensorDescriptor(desc->yDesc);
    cnnlDestroyTensorDescriptor(desc->xDesc);
    cnnlDestroyTensorDescriptor(desc->wDesc);
    cnnlDestroyFuseNormDescriptor(desc->opDesc);
    delete desc;
    return STATUS_SUCCESS;
}
