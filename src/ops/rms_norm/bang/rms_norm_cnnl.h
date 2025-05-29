#ifndef __CNNL_RMS_NORM_H__
#define __CNNL_RMS_NORM_H__

#include "../../../devices/bang/bang_handle.h"
#include "cnnl.h"
#include "cnnl_extra.h"
#include "operators.h"
#include <vector>

struct RMSNormCnnlDescriptor {
    Device device;
    int device_id;
    std::shared_ptr<Pool<cnnlHandle_t>> pool;
    DT dtype;
    DT w_datatype;
    cnnlTensorDescriptor_t yDesc;
    cnnlTensorDescriptor_t xDesc;
    cnnlTensorDescriptor_t wDesc;
    cnnlTensorDescriptor_t wDesc_ = nullptr;
    cnnlFuseNormDescriptor_t opDesc;
    uint64_t n;
    uint64_t d;
    float epsilon;
};

typedef struct RMSNormCnnlDescriptor *RMSNormCnnlDescriptor_t;

infiniopStatus_t cnnlCreateRMSNormDescriptor(BangHandle_t handle,
                                             RMSNormCnnlDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t y_desc,
                                             infiniopTensorDescriptor_t x_desc,
                                             infiniopTensorDescriptor_t w_desc,
                                             float epsilon);

infiniopStatus_t cnnlGetRMSNormWorkspaceSize(RMSNormCnnlDescriptor_t desc, uint64_t *size);

infiniopStatus_t cnnlRMSNorm(RMSNormCnnlDescriptor_t desc,
                             void *workspace,
                             uint64_t workspace_size,
                             void *y, void const *x, void const *w,
                             void *stream);

infiniopStatus_t cnnlDestroyRMSNormDescriptor(RMSNormCnnlDescriptor_t desc);

#endif// __CNNL_RMS_NORM_H__
