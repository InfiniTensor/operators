#ifndef REDUCESUM_H
#define REDUCESUM_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ReduceSumDescriptor {
    Device device;
} ReduceSumDescriptor;
typedef ReduceSumDescriptor *infiniopReduceSumDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReduceSumDescriptor(infiniopHandle_t handle,
                                                                infiniopReduceSumDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t dst,
                                                                infiniopTensorDescriptor_t src,
                                                                int* axis,
                                                                const int num_axis, 
                                                                int const keepdims);

__C __export infiniopStatus_t infiniopGetReduceSumWorkspaceSize(infiniopReduceSumDescriptor_t desc, uint64_t *size); 

__C __export infiniopStatus_t infiniopReduceSum(infiniopReduceSumDescriptor_t desc, void *workspace, uint64_t workspace_size, void *dst, void const *src, void *stream);

__C __export infiniopStatus_t infiniopDestroyReduceSumDescriptor(infiniopReduceSumDescriptor_t desc);
#endif
