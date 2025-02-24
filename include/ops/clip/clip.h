#ifndef CLIP_H
#define CLIP_H

#include "../../export.h"
#include "../../operators.h"
#include <optional>

typedef struct ClipDescriptor {
    Device device;
} ClipDescriptor;

typedef ClipDescriptor *infiniopClipDescriptor_t;

__C __export infiniopStatus_t infiniopCreateClipDescriptor(infiniopHandle_t handle,
                                                           infiniopClipDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y,
                                                           infiniopTensorDescriptor_t x,
                                                           float* lower_bound,
                                                           float* upper_bound);

__C __export infiniopStatus_t infiniopClip(infiniopClipDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream);

__C __export infiniopStatus_t infiniopDestroyClipDescriptor(infiniopClipDescriptor_t desc);

#endif
