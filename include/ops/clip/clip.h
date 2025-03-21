#ifndef CLIP_H
#define CLIP_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ClipDescriptor {
    Device device;
} ClipDescriptor;

typedef ClipDescriptor *infiniopClipDescriptor_t;

__C __export infiniopStatus_t infiniopCreateClipDescriptor(infiniopHandle_t handle,
                                                           infiniopClipDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t output,
                                                           infiniopTensorDescriptor_t input,
                                                           float min_value,
                                                           float max_value);

__C __export infiniopStatus_t infiniopClip(infiniopClipDescriptor_t desc,
                                           void *output,
                                           void const *input,
                                           void *stream);

__C __export infiniopStatus_t infiniopDestroyClipDescriptor(infiniopClipDescriptor_t desc);

#endif
