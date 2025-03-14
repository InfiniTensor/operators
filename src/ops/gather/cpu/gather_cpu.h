#ifndef __CPU_GATHER_H__
#define __CPU_GATHER_H__
#include "operators.h"

struct GatherCpuDescriptor {
    Device device;
    DT dtype;
    DT indice_type;
    uint64_t dst_size;
    uint64_t stride_axis;
    uint64_t shape_axis;
    uint64_t num_indices;
};

typedef GatherCpuDescriptor *GatherCpuDescriptor_t;

infiniopStatus_t cpuCreateGatherDescriptor(infiniopHandle_t,
                                        GatherCpuDescriptor_t *,
                                        infiniopTensorDescriptor_t dst,
                                        infiniopTensorDescriptor_t data,
                                        infiniopTensorDescriptor_t indices,
                                        int const axis);

infiniopStatus_t cpuGather(GatherCpuDescriptor_t desc,
                       void *dst,     
                       void const *data,
                       void const *indices,
                       void *stream);

infiniopStatus_t cpuDestroyGatherDescriptor(GatherCpuDescriptor_t desc);
#endif