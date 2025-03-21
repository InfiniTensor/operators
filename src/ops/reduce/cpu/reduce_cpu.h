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

    uint64_t reduced_ndim;
    uint64_t data_ndim;
    uint64_t axes_ndim;

    uint64_t reduced_size;
    uint64_t data_size;
    uint64_t axes_size;

    int64_t const *reduced_axes; //在 data 中需要保留的维度
    int64_t const *axes;
    
    int64_t const *data_strides;
    int64_t const *reduced_strides;
    int64_t const *anti_out_strides;

    // int reduce_element_num;
    bool keepdims;
    int reduce_mode;
};

typedef struct ReduceCpuDescriptor *ReduceCpuDescriptor_t;

infiniopStatus_t cpuCreateReduceDescriptor(infiniopHandle_t handle,
                                           ReduceCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t reduced,
                                           infiniopTensorDescriptor_t data,
                                           int64_t *axes,
                                           uint64_t axes_ndim,
                                           bool keepdims,
                                           bool noop_with_empty_axes,
                                           int reduce_type);

infiniopStatus_t cpuReduce(ReduceCpuDescriptor_t desc,
                            void *reduced,
                            void const *data,
                            void *stream);

infiniopStatus_t cpuDestroyReduceDescriptor(ReduceCpuDescriptor_t desc);

#endif
