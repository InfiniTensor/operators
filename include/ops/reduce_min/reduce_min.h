#ifndef REDUCE_MIN_H
#define REDUCE_MIN_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ReduceMinDescriptor {
    Device device;
} ReduceMinDescriptor;

typedef ReduceMinDescriptor *infiniopReduceMinDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReduceMinDescriptor(infiniopHandle_t handle,
                                                                infiniopReduceMinDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t y,
                                                                infiniopTensorDescriptor_t x,
                                                                int *axes,
                                                                uint64_t axes_ndim);

__C __export infiniopStatus_t infiniopGetReduceMinWorkspaceSize(infiniopReduceMinDescriptor_t desc, uint64_t *size);

__C __export infiniopStatus_t infiniopReduceMin(infiniopReduceMinDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x, void *stream);

__C __export infiniopStatus_t infiniopDestroyReduceMinDescriptor(infiniopReduceMinDescriptor_t desc);


#endif
