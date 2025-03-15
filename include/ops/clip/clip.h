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
                                                           infiniopTensorDescriptor_t output_desc,
                                                           infiniopTensorDescriptor_t intput_desc,
                                                           infiniopTensorDescriptor_t min_desc,
                                                           infiniopTensorDescriptor_t max_desc);

__C __export infiniopStatus_t infiniopClip(infiniopClipDescriptor_t desc,
                                           void *output,
                                           void *input,
                                           void const *min,
                                           void const *max,
                                           void *stream);

__C __export infiniopStatus_t infiniopDestroyClipDescriptor(infiniopClipDescriptor_t desc);

#endif