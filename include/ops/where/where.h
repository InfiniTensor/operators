#ifndef WHERE_H
#define WHERE_H

#include "../../export.h"
#include "../../operators.h"

typedef struct WhereDescriptor{
    Device device;
} WhereDescriptor;

typedef WhereDescriptor *infiniopWhereDescriptor_t;

__C __export infiniopStatus_t infiniopCreateWhereDescriptor(infiniopHandle_t handle,
                                                          infiniopWhereDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t dst,
                                                          infiniopTensorDescriptor_t x,
                                                          infiniopTensorDescriptor_t y,
                                                          infiniopTensorDescriptor_t condition);

__C __export infiniopStatus_t infiniopWhere(infiniopWhereDescriptor_t desc,
                                          void *dst,
                                          void const *x,
                                          void const *y,
                                          void const *condition,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroyWhereDescriptor(infiniopWhereDescriptor_t desc);

#endif
