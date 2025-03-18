#ifndef __CPU_REDUCE_H__
#define __CPU_REDUCE_H__

#include "operators.h"
#include <cstring>
#include <vector>
#include <numeric>

struct ReduceCpuDescriptor {
    Device device;
    DT dtype;
    bool use_1Dreduce;
    std::vector<int> axes;
    std::vector<int> out_of_axes;
    uint64_t ndim;
    uint64_t axes_ndim;
    uint64_t out_ndim;
    uint64_t axes_size;
    uint64_t out_size;
    int64_t *strides;
    std::vector<int> axes_strides;
    std::vector<int> out_strides;
    int reduce_op;
};

typedef struct ReduceCpuDescriptor *ReduceCpuDescriptor_t;

infiniopStatus_t cpuCreateReduceDescriptor(infiniopHandle_t,
                                           ReduceCpuDescriptor_t *,
                                           infiniopTensorDescriptor_t y,
                                           infiniopTensorDescriptor_t x,
                                           int *axes,
                                           uint64_t axes_ndim,
                                           int reduce_op);

infiniopStatus_t cpuReduce(ReduceCpuDescriptor_t desc,
                           void *y,
                           void const *x,
                           void *stream);

infiniopStatus_t cpuDestroyReduceDescriptor(ReduceCpuDescriptor_t desc);

#endif
