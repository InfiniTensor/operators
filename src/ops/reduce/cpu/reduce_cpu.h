#ifndef __CPU_REDUCE_H__
#define __CPU_REDUCE_H__

#include "../../../devices/cpu/common_cpu.h"
#include "operators.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

struct ReduceCpuDescriptor {
    Device device;
    DataLayout dt;
    uint64_t ndim;
    uint64_t y_size;
    uint64_t x_size;
    uint64_t *x_shape;
    uint64_t *y_shape;
    int64_t *axes;
    uint64_t n_axes;
    int keep_dims;
    int reduce_type;  // 0: max, 1: min, 2: mean
};

typedef struct ReduceCpuDescriptor *ReduceCpuDescriptor_t;

infiniopStatus_t cpuCreateReduceDescriptor(infiniopHandle_t handle,
                                           ReduceCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t y,
                                           infiniopTensorDescriptor_t x,
                                           int64_t const *axes,
                                           uint64_t n_axes,
                                           int keep_dims,
                                           int reduce_type);

infiniopStatus_t cpuReduce(ReduceCpuDescriptor_t desc,
                           void *y,
                           void const *x,
                           void *stream);

infiniopStatus_t cpuDestroyReduceDescriptor(ReduceCpuDescriptor_t desc);

#endif
