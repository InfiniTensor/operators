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
                                                            infiniopTensorDescriptor_t output_desc,
                                                            infiniopTensorDescriptor_t condition_desc,
                                                            infiniopTensorDescriptor_t x_desc,
                                                            infiniopTensorDescriptor_t y_desc);

__C __export infiniopStatus_t infiniopWhere(infiniopWhereDescriptor_t desc,
                                            void *output,
                                            void *condition,
                                            void *x,
                                            void *y,
                                            void *stream);

__C __export infiniopStatus_t infiniopDestroyWhereDescriptor(infiniopWhereDescriptor_t desc);

#endif