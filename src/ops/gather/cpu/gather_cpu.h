#ifndef __CPU_GATHER_H__
#define __CPU_GATHER_H__

#include "operators.h"
#include <numeric>
#include <type_traits>

struct GatherCpuDescriptor {
    Device device;
    DT dtype;
    DT indices_dtype;
    uint64_t data_ndim;
    uint64_t indices_ndim;
    // uint64_t o_data_size;
    // uint64_t const *o_shape;
    uint64_t const *data_shape;
    uint64_t const *indices_shape;
    uint64_t const *data_strides;
    uint64_t const *indices_strides;
    // uint64_t *o_indices;
    int64_t axis;
};

typedef struct GatherCpuDescriptor *GatherCpuDescriptor_t;

infiniopStatus_t cpuCreateGatherDescriptor(infiniopHandle_t,
                                           GatherCpuDescriptor_t *,
                                           infiniopTensorDescriptor_t output,
                                           infiniopTensorDescriptor_t data,
                                           infiniopTensorDescriptor_t indices,
                                           int64_t axis);

infiniopStatus_t cpuGather(GatherCpuDescriptor_t desc,
                           void *output,
                           void const *data,
                           void const *indices,
                           void *stream);

infiniopStatus_t cpuDestroyGatherDescriptor(GatherCpuDescriptor_t desc);

#endif
