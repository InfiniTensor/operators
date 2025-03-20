#ifndef REDUCEMAX_H
#define REDUCEMAX_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ReducemaxDescriptor {
    Device device;
} ReducemaxDescriptor;
typedef ReducemaxDescriptor *infiniopReducemaxDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReducemaxDescriptor(infiniopHandle_t handle,
                                                                infiniopReducemaxDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t y,
                                                                infiniopTensorDescriptor_t x,
                                                                int64_t const *axes,
                                                                uint64_t n,
                                                                bool keepdims,
                                                                bool noop_with_empty_axes
                                                                );

__C __export infiniopStatus_t infiniopReducemax(infiniopReducemaxDescriptor_t desc, void *y, const void *x, void *dynamic_axes, uint64_t dynamic_axes_size, void *stream);

__C __export infiniopStatus_t infiniopDestroyReducemaxDescriptor(infiniopReducemaxDescriptor_t desc);
#endif
