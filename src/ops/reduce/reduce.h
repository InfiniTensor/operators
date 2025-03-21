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
                                                    infiniopTensorDescriptor_t reduced,
                                                    infiniopTensorDescriptor_t data,
                                                    int64_t *axes,
                                                    uint64_t axes_ndim,
                                                    bool keepdims,
                                                    bool noop_with_empty_axes,
                                                    int reduce_type);

__C infiniopStatus_t infiniopGetReduceWorkspaceSize(infiniopReduceDescriptor_t desc, uint64_t *size);

__C infiniopStatus_t infiniopReduce(infiniopReduceDescriptor_t desc,
                                    void *reduced,
                                    void const *data,
                                    void *stream);

__C infiniopStatus_t infiniopDestroyReduceDescriptor(infiniopReduceDescriptor_t desc);
#endif
