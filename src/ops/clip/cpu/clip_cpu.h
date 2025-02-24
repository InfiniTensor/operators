#ifndef __CPU_CLIP_H__
#define __CPU_CLIP_H__

#include "operators.h"
#include <cstring>
#include <numeric>
#include <optional>

struct ClipCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t data_size;
    bool has_lower_bound;
    float lower_bound;
    bool has_upper_bound;
    float upper_bound;
};

typedef struct ClipCpuDescriptor *ClipCpuDescriptor_t;

infiniopStatus_t cpuCreateClipDescriptor(infiniopHandle_t,
                                         ClipCpuDescriptor_t *,
                                         infiniopTensorDescriptor_t y,
                                         infiniopTensorDescriptor_t x,
                                         float* lower_bound,
                                         float* upper_bound);

infiniopStatus_t cpuClip(ClipCpuDescriptor_t desc,
                         void *y,
                         void const *x);

infiniopStatus_t cpuDestroyClipDescriptor(ClipCpuDescriptor_t desc);

#endif
