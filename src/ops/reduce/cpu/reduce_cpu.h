#ifndef __CPU_REDUCE_H__
#define __CPU_REDUCE_H__

#include "../../../devices/cpu/common_cpu.h"
#include "operators.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <unordered_set>

struct ReduceCpuDescriptor {
    Device device;
    DataLayout dt;
    uint64_t ndim;
    uint64_t x_ndim;
    uint64_t const *x_shape;
    int64_t *axes;
    int64_t const *x_strides;
    uint64_t *reduce_axes_stride;
    int64_t const *y_strides;
    uint64_t y_size;
    int reduce_mode;
    int reduce_element_num;
    uint64_t axes_num;
    bool is_axes_static;
    bool noop_with_empty_axes;
    bool owns_axes_memory;
    bool keepdims;
};

typedef struct ReduceCpuDescriptor *ReduceCpuDescriptor_t;

infiniopStatus_t cpuCreateReduceDescriptor(infiniopHandle_t handle,
                                            ReduceCpuDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t y,
                                            infiniopTensorDescriptor_t x,
                                            int64_t const *axes,
                                            uint64_t n,
                                            int reduce_type,
                                            bool noop_with_empty_axes,
                                            bool keepdims);


infiniopStatus_t cpuReduce(ReduceCpuDescriptor_t desc,
                            void *y,
                            void const *x,
                            void *dynamic_axes,
                            uint64_t dynamic_axes_size,
                            void *stream);

infiniopStatus_t cpuDestroyReduceDescriptor(ReduceCpuDescriptor_t desc);

#endif
