#ifndef __CPU_GATHER_H__
#define __CPU_GATHER_H__

#include "operators.h"
#include <cstring>
#include <numeric>

struct GatherCpuDescriptor {
    Device device;
    DT dtype;
    DT indices_dtype;
    uint64_t pre_size;
    uint64_t axis_size;
    uint64_t indices_size;
    uint64_t post_size;
};

typedef struct GatherCpuDescriptor *GatherCpuDescriptor_t;

infiniopStatus_t cpuCreateGatherDescriptor(infiniopHandle_t,
                                           GatherCpuDescriptor_t *,
                                           infiniopTensorDescriptor_t output,
                                           infiniopTensorDescriptor_t input,
                                           infiniopTensorDescriptor_t indices,
                                           uint64_t axis);

infiniopStatus_t cpuGather(GatherCpuDescriptor_t desc,
                           void *output,
                           void const *input,
                           void const *indices,
                           void *stream);

infiniopStatus_t cpuDestroyGatherDescriptor(GatherCpuDescriptor_t desc);

#endif
