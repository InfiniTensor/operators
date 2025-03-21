#ifndef __CPU_CLIP_H__
#define __CPU_CLIP_H__

#include "operators.h"
#include <numeric>
#include <type_traits>

struct ClipCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t ndim;
    uint64_t o_data_size;
    uint64_t const *o_shape;
    uint64_t const *i_strides;
    uint64_t *o_indices;
    float min_value;
    float max_value;
};

typedef struct ClipCpuDescriptor *ClipCpuDescriptor_t;

infiniopStatus_t cpuCreateClipDescriptor(infiniopHandle_t,
                                         ClipCpuDescriptor_t *,
                                         infiniopTensorDescriptor_t output,
                                         infiniopTensorDescriptor_t input,
                                         float min_value,
                                         float max_value);

infiniopStatus_t cpuClip(ClipCpuDescriptor_t desc,
                         void *output,
                         void const *input,
                        //  float min_value,
                        //  float max_value,
                         void *stream);

infiniopStatus_t cpuDestroyClipDescriptor(ClipCpuDescriptor_t desc);

#endif
