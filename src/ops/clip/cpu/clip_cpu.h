#ifndef __CPU_CLIP_H__
#define __CPU_CLIP_H__

#include "operators.h"
struct ClipCpuDescriptor {
    Device device;
    DT dtype;
    float min;
    float max;
    bool has_min;
    bool has_max;
    uint64_t element_num;
};

typedef struct ClipCpuDescriptor *ClipCpuDescriptor_t;

infiniopStatus_t cpuCreateClipDescriptor(infiniopHandle_t handle,
                                        ClipCpuDescriptor_t *desc_ptr,
                                        infiniopTensorDescriptor_t x,
                                        infiniopTensorDescriptor_t y,
                                        float* min,
                                        float* max
                                        );

infiniopStatus_t cpuClip(ClipCpuDescriptor_t desc,
                                  void const *x, 
                                  void *y,
                                  void *stream);

infiniopStatus_t cpuDestroyClipDescriptor(ClipCpuDescriptor_t desc);

#endif
