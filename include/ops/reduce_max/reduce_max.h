#ifndef REDUCE_MAX_H
#define REDUCE_MAX_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ReduceMaxDescriptor {
    Device device;
} ReduceMaxDescriptor;
typedef ReduceMaxDescriptor *infiniopReduceMaxDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReduceMaxDescriptor(infiniopHandle_t handle,
                                                                infiniopReduceMaxDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t y,
                                                                infiniopTensorDescriptor_t x,
                                                                int64_t const *axes,
                                                                uint64_t n_axes,
                                                                int keep_dims);

__C __export infiniopStatus_t infiniopReduceMax(infiniopReduceMaxDescriptor_t desc, void *y, void const *x, void *stream);

__C __export infiniopStatus_t infiniopDestroyReduceMaxDescriptor(infiniopReduceMaxDescriptor_t desc);

#endif
