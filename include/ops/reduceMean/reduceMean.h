#ifndef REDUCEMEAN_H
#define REDUCEMEAN_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ReduceMeanDescriptor {
    Device device;
} ReduceMeanDescriptor;
typedef ReduceMeanDescriptor *infiniopReduceMeanDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReduceMeanDescriptor(infiniopHandle_t handle,
                                                                infiniopReduceMeanDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t dst,
                                                                infiniopTensorDescriptor_t src,
                                                                int* axis,
                                                                const int num_axis, 
                                                                int const keepdims);
                                                                
__C __export infiniopStatus_t infiniopGetReduceMeanWorkspaceSize(infiniopReduceMeanDescriptor_t desc, uint64_t *size); 

__C __export infiniopStatus_t infiniopReduceMean(infiniopReduceMeanDescriptor_t desc, void *workspace, uint64_t workspace_size, void *dst, void const *src, void *stream);

__C __export infiniopStatus_t infiniopDestroyReduceMeanDescriptor(infiniopReduceMeanDescriptor_t desc);
#endif
