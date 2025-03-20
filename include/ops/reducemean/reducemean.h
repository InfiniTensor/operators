#ifndef REDUCEMEAN_H
#define REDUCEMEAN_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ReducemeanDescriptor {
    Device device;
} ReducemeanDescriptor;
typedef ReducemeanDescriptor *infiniopReducemeanDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReducemeanDescriptor(infiniopHandle_t handle,
                                                                infiniopReducemeanDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t y,
                                                                infiniopTensorDescriptor_t x,
                                                                int64_t const *axes,
                                                                uint64_t n,
                                                                bool keepdims,
                                                                bool noop_with_empty_axes
                                                                );

__C __export infiniopStatus_t infiniopReducemean(infiniopReducemeanDescriptor_t desc, void *dst, const void *src, void *dynamic_axes, uint64_t dynamic_axes_size, void *stream);

__C __export infiniopStatus_t infiniopDestroyReducemeanDescriptor(infiniopReducemeanDescriptor_t desc);
#endif
