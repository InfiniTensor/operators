#ifndef CONV_H
#define CONV_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ConvDescriptor {
    Device device;
} ConvDescriptor;

typedef ConvDescriptor *infiniopConvDescriptor_t;

__C __export infiniopStatus_t infiniopCreateConvDescriptor(infiniopHandle_t handle,
                                                           infiniopConvDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y,
                                                           infiniopTensorDescriptor_t x,
                                                           infiniopTensorDescriptor_t w,
                                                           infiniopTensorDescriptor_t b,
                                                           uint64_t const *pads,
                                                           int64_t const *strides,
                                                           uint64_t const *dilations,
                                                           uint64_t n);

__C __export infiniopStatus_t infiniopGetConvWorkspaceSize(infiniopConvDescriptor_t desc, uint64_t *size);

__C __export infiniopStatus_t infiniopConv(infiniopConvDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x, void const *w, void const *b, void *stream);

__C __export infiniopStatus_t infiniopDestroyConvDescriptor(infiniopConvDescriptor_t desc);


#endif
