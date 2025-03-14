#ifndef REDUCEMAX_H
#define REDUCEMAX_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ReduceMaxDescriptor {
    Device device;
} ReduceMaxDescriptor;
typedef ReduceMaxDescriptor *infiniopReduceMaxDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReduceMaxDescriptor(infiniopHandle_t handle,
                                                                infiniopReduceMaxDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t dst,
                                                                infiniopTensorDescriptor_t src,
                                                                int* axis,
                                                                const int num_axis, 
                                                                int const keepdims);

__C __export infiniopStatus_t infiniopGetReduceMaxWorkspaceSize(infiniopReduceMaxDescriptor_t desc, uint64_t *size); 

__C __export infiniopStatus_t infiniopReduceMax(infiniopReduceMaxDescriptor_t desc, void *workspace, uint64_t workspace_size, void *dst, void const *src, void *stream);

__C __export infiniopStatus_t infiniopDestroyReduceMaxDescriptor(infiniopReduceMaxDescriptor_t desc);
#endif
