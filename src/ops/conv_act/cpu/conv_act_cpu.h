#ifndef __CPU_CONV_ACT_H__
#define __CPU_CONV_ACT_H__

#include "../../../devices/cpu/common_cpu.h"
#include "operators.h"
#include "ops/conv_act/conv_act.h"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <functional>
#include <numeric>

struct ConvActCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t ndim;
    uint64_t y_size;
    uint64_t padded_x_size;
    uint64_t const *padded_shape;
    uint64_t const *x_shape;
    uint64_t const *w_shape;
    uint64_t const *b_shape;
    uint64_t const *y_shape;
    uint64_t const *pads;
    int64_t const *strides;
    uint64_t const *dilations;
    InfiniActivationMode_t mode;
    bool bias_is_optional;
    ConvActParam_t act_params;
};

typedef struct ConvActCpuDescriptor *ConvActCpuDescriptor_t;

infiniopStatus_t cpuCreateConvActDescriptor(infiniopHandle_t,
                                            ConvActCpuDescriptor_t *,
                                            infiniopTensorDescriptor_t y,
                                            infiniopTensorDescriptor_t x,
                                            infiniopTensorDescriptor_t w,
                                            infiniopTensorDescriptor_t b,
                                            uint64_t const *pads,
                                            int64_t const *strides,
                                            uint64_t const *dilations,
                                            uint64_t n,
                                            InfiniActivationMode_t activation_mode,
                                            ConvActParam_t act_params);

infiniopStatus_t cpuGetConvActWorkspaceSize(ConvActCpuDescriptor_t desc, uint64_t *size);

infiniopStatus_t cpuConvAct(ConvActCpuDescriptor_t desc,
                            void *workspace, uint64_t workspace_size,
                            void *y, void const *x, void const *w,
                            void const *b, void *stream);

infiniopStatus_t cpuDestroyConvActDescriptor(ConvActCpuDescriptor_t desc);

#endif
