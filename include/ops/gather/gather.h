#ifndef GAHTER_H
#define GAHTER_H

#include "../../export.h"
#include "../../operators.h"

typedef struct GatherDescriptor {
    Device device;
} GatherDescriptor;
typedef GatherDescriptor *infiniopGatherDescriptor_t;

__C __export infiniopStatus_t infiniopCreateGatherDescriptor(infiniopHandle_t handle,
                                                                infiniopGatherDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t y,
                                                                infiniopTensorDescriptor_t x,
                                                                infiniopTensorDescriptor_t indices,
                                                                int64_t axis
                                                                );

__C __export infiniopStatus_t infiniopGather(infiniopGatherDescriptor_t desc, void const *x, void const *indices, void *y, void *stream);

__C __export infiniopStatus_t infiniopDestroyGatherDescriptor(infiniopGatherDescriptor_t desc);

#endif