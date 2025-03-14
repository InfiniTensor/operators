#ifndef __CPU_GATHER_ELEMENTS_H__
#define __CPU_GATHER_ELEMENTS_H__
#include "operators.h"

struct GatherElementsCpuDescriptor {
    Device device;
    DT dtype;
    DT indice_type;
    uint64_t indices_size;
    uint64_t stride_axis;
    uint64_t shape_axis;
    uint64_t num_indices;
};

typedef GatherElementsCpuDescriptor *GatherElementsCpuDescriptor_t;

infiniopStatus_t cpuCreateGatherElementsDescriptor(infiniopHandle_t,
                                        GatherElementsCpuDescriptor_t *,
                                        infiniopTensorDescriptor_t dst,
                                        infiniopTensorDescriptor_t data,
                                        infiniopTensorDescriptor_t indices,
                                        int const axis);

infiniopStatus_t cpuGatherElements(GatherElementsCpuDescriptor_t desc,
                       void *dst,     
                       void const *data,
                       void const *indices,
                       void *stream);

infiniopStatus_t cpuDestroyGatherElementsDescriptor(GatherElementsCpuDescriptor_t desc);
#endif