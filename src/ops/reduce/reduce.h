#ifndef REDUCE_H
#define REDUCE_H

#include "export.h"
#include "operators.h"

typedef struct ReduceDescriptor {
    Device device;
} ReduceDescriptor;
typedef ReduceDescriptor *infiniopReduceDescriptor_t;

__C infiniopStatus_t infiniopCreateReduceDescriptor(infiniopHandle_t handle,
                                                    infiniopReduceDescriptor_t *desc_ptr,
                                                    infiniopTensorDescriptor_t dst,
                                                    infiniopTensorDescriptor_t src,
                                                    int *axis,
                                                    const int num_axis,
                                                    int const keepdims,
                                                    int reduce_type);

__C infiniopStatus_t infiniopGetReduceWorkspaceSize(infiniopReduceDescriptor_t desc, uint64_t *size);

__C infiniopStatus_t infiniopReduce(infiniopReduceDescriptor_t desc,
                                    void *workspace, 
                                    uint64_t workspace_size,
                                    void *dst, 
                                    void const*src,
                                    void *stream);

__C infiniopStatus_t infiniopDestroyReduceDescriptor(infiniopReduceDescriptor_t desc);
#endif
