#ifndef __CPU_CLIP_H__
#define __CPU_CLIP_H__
#include "operators.h"

struct ClipCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t data_size;
    float max;
    float min;
};

typedef ClipCpuDescriptor *ClipCpuDescriptor_t;

infiniopStatus_t cpuCreateClipDescriptor(infiniopHandle_t,
                                        ClipCpuDescriptor_t *,
                                        infiniopTensorDescriptor_t dst,
                                        infiniopTensorDescriptor_t src,
                                        float max,
                                        float min);

infiniopStatus_t cpuClip(ClipCpuDescriptor_t desc,
                       void *dst, void const *src,
                       void *stream);

infiniopStatus_t cpuDestroyClipDescriptor(ClipCpuDescriptor_t desc);
#endif