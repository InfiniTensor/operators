#ifndef __REDUCE_MEAN_H__
#define __REDUCE_MEAN_H__

#include "../../export.h"
#include "../../operators.h"

typedef struct ReduceMeanDescriptor {
    Device device;
} ReduceMeanDescriptor;

typedef ReduceMeanDescriptor *infiniopReduceMeanDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReduceMeanDescriptor(infiniopHandle_t handle,
                                                                infiniopReduceMeanDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t reduced,
                                                                infiniopTensorDescriptor_t data,
                                                                int64_t const *axes,
                                                                size_t axes_size,
                                                                int keepdims,
                                                                int noop_with_empty_axes);

__C __export infiniopStatus_t infiniopReduceMean(infiniopReduceMeanDescriptor_t desc,
                                                void *reduced,
                                                void const *data,
                                                int64_t const *axes,
                                                void *stream);

__C __export infiniopStatus_t infiniopDestroyReduceMeanDescriptor(infiniopReduceMeanDescriptor_t desc);

#endif