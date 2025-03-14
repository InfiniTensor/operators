#ifndef CLIP_H
#define CLIP_H

#include "../../export.h"
#include "../../operators.h"
#include <limits>

typedef struct{
    Device device;
}ClipDescriptor;

typedef ClipDescriptor *infiniopClipDescriptor_t;

__C __export infiniopStatus_t infiniopCreateClipDescriptor(infiniopHandle_t handle,
                                                    infiniopClipDescriptor_t *desc_ptr,
                                                    infiniopTensorDescriptor_t dst,
                                                    infiniopTensorDescriptor_t src,
                                                    float max,
                                                    float min);

__C __export infiniopStatus_t infiniopClip(infiniopClipDescriptor_t desc,
                                    void *dst, 
                                    void const *src,
                                    void *stream);

__C __export infiniopStatus_t infiniopDestroyClipDescriptor(infiniopClipDescriptor_t desc);

#endif