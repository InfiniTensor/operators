#ifndef GATHER_ELEMENTS_H
#define GATHER_ELEMENTS_H

#include "../../export.h"
#include "../../operators.h"

typedef struct{
    Device device;
}GatherElementsDescriptor;

typedef GatherElementsDescriptor *infiniopGatherElementsDescriptor_t;

__C __export infiniopStatus_t infiniopCreateGatherElementsDescriptor(infiniopHandle_t handle,
                                                             infiniopGatherElementsDescriptor_t *desc_ptr,
                                                             infiniopTensorDescriptor_t dst,
                                                             infiniopTensorDescriptor_t data,
                                                             infiniopTensorDescriptor_t indices,
                                                             int axis);

__C __export infiniopStatus_t infiniopGatherElements(infiniopGatherElementsDescriptor_t desc,
                                             void *dst, 
                                             void const *data,
                                             void const *indices,
                                             void *stream);

__C __export infiniopStatus_t infiniopDestroyGatherElementsDescriptor(infiniopGatherElementsDescriptor_t desc);

#endif