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
                                                     infiniopTensorDescriptor_t y,
                                                     infiniopTensorDescriptor_t x,
                                                     int64_t const *axes,
                                                     uint64_t n,
                                                     bool keepdims,
                                                     bool noop_with_empty_axes,
                                                     int reduce_type);

__C infiniopStatus_t infiniopReduce(infiniopReduceDescriptor_t desc, void *y, const void *x, void *dynamic_axes, uint64_t dynamic_axes_size, void *stream);

__C infiniopStatus_t infiniopDestroyReduceDescriptor(infiniopReduceDescriptor_t desc);
#endif