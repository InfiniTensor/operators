#ifndef __CPU_REDUCE_H__
#define __CPU_REDUCE_H__

#include "../../../devices/cpu/common_cpu.h"
#include "operators.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

struct ReduceCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t dst_size;
    uint64_t data_size;
    uint64_t* stride_axis;
    uint64_t* shape_axis;
    uint64_t add_dim;
    int axis_num;
    int reduce_mode;
};

typedef struct ReduceCpuDescriptor *ReduceCpuDescriptor_t;

infiniopStatus_t cpuCreateReduceDescriptor(infiniopHandle_t handle,
                                           ReduceCpuDescriptor_t *,
                                           infiniopTensorDescriptor_t dst,
                                           infiniopTensorDescriptor_t src,
                                           int *axis,
                                           const int num_axis,
                                           int const keepdims,
                                           int reduce_type);

infiniopStatus_t cpuGetReduceWorkspaceSize(ReduceCpuDescriptor_t desc, uint64_t *size);

infiniopStatus_t cpuReduce(ReduceCpuDescriptor_t desc,
                            void *workspace,
                            uint64_t workspace_size,
                            void *dst,     
                            void const *src,
                            void *stream);

infiniopStatus_t cpuDestroyReduceDescriptor(ReduceCpuDescriptor_t desc);

#endif
