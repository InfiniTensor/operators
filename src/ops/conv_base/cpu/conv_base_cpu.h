#ifndef __CPU_CONV_BASE_H__
#define __CPU_CONV_BASE_H__

#include "../../../devices/cpu/common_cpu.h"
#include "operators.h"
#include <algorithm>
#include <cstring>
#include <numeric>

struct ConvBaseCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t ndim;
    uint64_t y_size;
    uint64_t padded_x_size;
    uint64_t const *padded_shape;
    uint64_t const *x_shape;
    uint64_t const *w_shape;
    uint64_t const *y_shape;
    uint64_t const *pads;
    int64_t const *strides;
    uint64_t const *dilations;
};

typedef struct ConvBaseCpuDescriptor *ConvBaseCpuDescriptor_t;

infiniopStatus_t cpuCreateConvBaseDescriptor(infiniopHandle_t,
                                             ConvBaseCpuDescriptor_t *,
                                             infiniopTensorDescriptor_t y,
                                             infiniopTensorDescriptor_t x,
                                             infiniopTensorDescriptor_t w,
                                             uint64_t const *pads,
                                             int64_t const *strides,
                                             uint64_t const *dilations,
                                             uint64_t n);

infiniopStatus_t cpuGetConvBaseWorkspaceSize(ConvBaseCpuDescriptor_t desc, uint64_t *size);

infiniopStatus_t cpuConvBase(ConvBaseCpuDescriptor_t desc,
                             void *workspace, uint64_t workspace_size,
                             void *y, void const *x, void const *w,
                             void *stream);

infiniopStatus_t cpuDestroyConvBaseDescriptor(ConvBaseCpuDescriptor_t desc);

#endif
