#ifndef __BANG_LAYER_NORM_H__
#define __BANG_LAYER_NORM_H__

#include "../../../devices/bang/bang_handle.h"
#include "../../utils.h"
#include "operators.h"

struct LayerNormBangDescriptor {
    Device device;
    int device_id;
    DT dtype;
    int size;
    int behindsize;
    float epsilon;
};

typedef struct LayerNormBangDescriptor *LayerNormBangDescriptor_t;

infiniopStatus_t bangCreateLayerNormDescriptor(BangHandle_t handle,
                                             LayerNormBangDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t x_desc,
                                             infiniopTensorDescriptor_t w_desc,
                                             infiniopTensorDescriptor_t b_desc,
                                             infiniopTensorDescriptor_t y_desc,
                                             float epsilon);

infiniopStatus_t bangGetLayerNormWorkspaceSize(LayerNormBangDescriptor_t desc, unsigned long int *size);

infiniopStatus_t bangLayerNorm(LayerNormBangDescriptor_t desc, void *workspace,
                                          uint64_t workspace_size,                       
                             void const *x, void const *w, void const *b, void *y, 
                             void *stream);

infiniopStatus_t bangDestroyLayerNormDescriptor(LayerNormBangDescriptor_t desc);

#endif// __BANG_LAYER_NORM_H__
