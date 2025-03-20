#ifndef REDUCEMIN_H
#define REDUCEMIN_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ReduceminDescriptor {
    Device device;
} ReduceminDescriptor;
typedef ReduceminDescriptor *infiniopReduceminDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReduceminDescriptor(infiniopHandle_t handle,
                                                                infiniopReduceminDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t dst,
                                                                infiniopTensorDescriptor_t src,
                                                                int64_t const *axes,
                                                                uint64_t n,
                                                                bool keepdims,
                                                                bool noop_with_empty_axes
                                                                );

__C __export infiniopStatus_t infiniopReducemin(infiniopReduceminDescriptor_t desc, void *dst, const void *src, void *dynamic_axes, uint64_t dynamic_axes_size, void *stream);

__C __export infiniopStatus_t infiniopDestroyReduceminDescriptor(infiniopReduceminDescriptor_t desc);
#endif
