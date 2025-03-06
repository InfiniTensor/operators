#ifndef REDUCE_MEAN_H
#define REDUCE_MEAN_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ReduceMeanDescriptor {
    Device device;
} ReduceMeanDescriptor;
typedef ReduceMeanDescriptor *infiniopReduceMeanDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReduceMeanDescriptor(infiniopHandle_t handle,
                                                                 infiniopReduceMeanDescriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t y,
                                                                 infiniopTensorDescriptor_t x,
                                                                 int64_t const *axes,
                                                                 uint64_t n_axes,
                                                                 int keep_dims);

__C __export infiniopStatus_t infiniopReduceMean(infiniopReduceMeanDescriptor_t desc, void *y, void const *x, void *stream);

__C __export infiniopStatus_t infiniopDestroyReduceMeanDescriptor(infiniopReduceMeanDescriptor_t desc);

#endif
