#ifndef WHERE_H
#define WHERE_H

#include "../../export.h"
#include "../../operators.h"

typedef struct WhereDescriptor {
    Device device;
} WhereDescriptor;
typedef WhereDescriptor *infiniopWhereDescriptor_t;

__C __export infiniopStatus_t infiniopCreateWhereDescriptor(infiniopHandle_t handle,
                                                            infiniopWhereDescriptor_t *desc_ptr,
                                                            infiniopTensorDescriptor_t dst,
                                                            infiniopTensorDescriptor_t src1,
                                                            infiniopTensorDescriptor_t src2,
                                                            infiniopTensorDescriptor_t condition
                                                            );

__C __export infiniopStatus_t infiniopWhere(infiniopWhereDescriptor_t desc, void *dst, void const *src1, void const *src2, void const *condition, void *stream);

__C __export infiniopStatus_t infiniopDestroyWhereDescriptor(infiniopWhereDescriptor_t desc);

#endif