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
    cnnlTensorDescriptor_t yDesc, xDesc, wDesc;
    cnnlCreateTensorDescriptor(&yDesc);
    cnnlCreateTensorDescriptor(&xDesc);
    cnnlCreateTensorDescriptor(&wDesc);
    
    // Set tensor descriptors
    setCnnlTensor(yDesc, y_desc);
    setCnnlTensor(xDesc, x_desc);
    setCnnlTensor(wDesc, w_desc);

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
        yDesc,
        xDesc,
        wDesc,
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

    use_cnnl(desc->pool, desc->device_id, (cnrtQueue_t)stream,
             [&](cnnlHandle_t handle) {
                 cnnlFuseNorm_v4(handle,
                                       desc->xDesc, x,        // Input tensor
                                       nullptr, nullptr,         // Input scale (unused)
                                       desc->wDesc, w,        // Norm scale (gamma)
                                       nullptr, nullptr,         // Norm bias (beta, unused)
                                       nullptr, nullptr,         // Residual input (unused)
                                       nullptr, nullptr,         // Bias (unused)
                                       1e-5f,                    // Epsilon
                                       CNNL_QUANTIZE_NONE,       // No quantization
                                       false,                    // Don't store output before norm
                                       false,                    // Don't store output after norm
                                       CNNL_TRANSFORMER_RMSNORM, // Norm type (RMSNorm)
                                       CNNL_DTYPE_FLOAT,         // Compute precision (float)
                                       workspace, workspace_size,
                                       desc->yDesc, y,  // Output
                                       nullptr, nullptr,   // Output before norm (unused)
                                       nullptr, nullptr,   // Output quant scale (unused)
                                       nullptr, nullptr);  // Output after norm (unused)
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
