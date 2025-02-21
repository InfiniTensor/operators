#ifndef CONV_BASE_H
#define CONV_BASE_H

#include "export.h"
#include "operators.h"

typedef struct ConvBaseDescriptor {
    Device device;
} ConvBaseDescriptor;

typedef ConvBaseDescriptor *infiniopConvBaseDescriptor_t;

__C infiniopStatus_t infiniopCreateConvBaseDescriptor(infiniopHandle_t handle,
                                                      infiniopConvBaseDescriptor_t *desc_ptr,
                                                      infiniopTensorDescriptor_t y,
                                                      infiniopTensorDescriptor_t x,
                                                      infiniopTensorDescriptor_t w,
                                                      uint64_t const *pads,
                                                      int64_t const *strides,
                                                      uint64_t const *dilations,
                                                      uint64_t n);

__C infiniopStatus_t infiniopGetConvBaseWorkspaceSize(infiniopConvBaseDescriptor_t desc, uint64_t *size);

__C infiniopStatus_t infiniopConvBase(infiniopConvBaseDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x, void const *w, void *stream);

__C infiniopStatus_t infiniopDestroyConvBaseDescriptor(infiniopConvBaseDescriptor_t desc);


#endif
