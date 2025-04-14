#ifndef __CNNL_LAYER_NORM_H__
#define __CNNL_LAYER_NORM_H__

#include "../../../devices/bang/bang_handle.h"
#include "../../utils.h"
#include "operators.h"

struct LayerNormCnnlDescriptor {
    Device device;
    int device_id;
    DT dtype;
    std::shared_ptr<Pool<cnnlHandle_t>> cnnl_handles;
    cnnlTensorDescriptor_t xDesc;
    cnnlTensorDescriptor_t yDesc;
    cnnlTensorDescriptor_t filter_bias_desc;
    cnnlTensorDescriptor_t mean_rstd_desc;
    int axis;
    size_t size_mean_rstd;
    size_t wsSize;
    float epsilon;
};

typedef struct LayerNormCnnlDescriptor *LayerNormCnnlDescriptor_t;

infiniopStatus_t cnnlCreateLayerNormDescriptor(BangHandle_t handle,
                                             LayerNormCnnlDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t x_desc,
                                             infiniopTensorDescriptor_t w_desc,
                                             infiniopTensorDescriptor_t b_desc,
                                             infiniopTensorDescriptor_t y_desc,
                                             float epsilon);

infiniopStatus_t cnnlGetLayerNormWorkspaceSize(LayerNormCnnlDescriptor_t desc, unsigned long int *size);

infiniopStatus_t cnnlLayerNorm(LayerNormCnnlDescriptor_t desc, void *workspace,
                                          uint64_t workspace_size,                        
                             void const *x, void const *w, void const *b, void *y, 
                             void *stream);

infiniopStatus_t cnnlDestroyLayerNormDescriptor(LayerNormCnnlDescriptor_t desc);

#endif// __CNNL_LAYER_NORM_H__
