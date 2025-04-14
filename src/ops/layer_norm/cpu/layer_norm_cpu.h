#ifndef __CPU_LAYER_NORM_H__
#define __CPU_LAYER_NORM_H__

#include "operators.h"

struct LayerNormCpuDescriptor {
    Device device;
    DT dtype;
    int size;
    int behindsize;
    float epsilon;
};

typedef struct LayerNormCpuDescriptor *LayerNormCpuDescriptor_t;

infiniopStatus_t cpuCreateLayerNormDescriptor(infiniopHandle_t handle, LayerNormCpuDescriptor_t *desc_ptr,                                           
                                            infiniopTensorDescriptor_t x_desc,
                                            infiniopTensorDescriptor_t w_desc,
                                            infiniopTensorDescriptor_t b_desc,
                                            infiniopTensorDescriptor_t y_desc,
                                            float epsilon);

infiniopStatus_t cpuGetLayerNormWorkspaceSize(LayerNormCpuDescriptor_t desc, unsigned long int *size);

infiniopStatus_t cpuLayerNorm(LayerNormCpuDescriptor_t desc, void *workspace,
                                          uint64_t workspace_size,
                            void const *x, void const *w, void const *b, void *y,
                            void *stream);
infiniopStatus_t cpuDestroyLayerNormDescriptor(LayerNormCpuDescriptor_t desc);

#endif// __CPU_LAYER_NORM_H__
