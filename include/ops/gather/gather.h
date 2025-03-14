#ifndef GATHER_H
#define GATHER_H

#include "../../export.h"
#include "../../operators.h"

typedef struct{
    Device device;
}GatherDescriptor;

typedef GatherDescriptor *infiniopGatherDescriptor_t;

__C __export infiniopStatus_t infiniopCreateGatherDescriptor(infiniopHandle_t handle,
                                                             infiniopGatherDescriptor_t *desc_ptr,
                                                             infiniopTensorDescriptor_t dst,
                                                             infiniopTensorDescriptor_t data,
                                                             infiniopTensorDescriptor_t indices,
                                                             int axis);

__C __export infiniopStatus_t infiniopGather(infiniopGatherDescriptor_t desc,
                                             void *dst, 
                                             void const *data,
                                             void const *indices,
                                             void *stream);

__C __export infiniopStatus_t infiniopDestroyGatherDescriptor(infiniopGatherDescriptor_t desc);

#endif