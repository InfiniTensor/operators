#ifndef CONV_ACT_H
#define CONV_ACT_H

#include "../../export.h"
#include "../../operators.h"
#include "../activations.h"
#include <cstddef>

typedef struct ConvActParam {
    /**
     * Used by:
     *  - INFINI_ACTIVATION_CLIPPED_RELU: as its clipping ceiling
     */
    double clip_coef;
    /**
     * Used by:
     *  - INFINI_ACTIVATION_LEAKY_RELU: as its slope for x < 0
     *  - INFINI_ACTIVATION_ELU: alpha * (exp(x) - 1.) for x < 0
     */
    double alpha;
    /**
     * Used by:
     *  - INFINI_ACTIVATION_GELU: as its approximation switch
     */
    const char *approximate;

} ConvActParam_t;

typedef struct ConvActDescriptor {
    Device device;
} ConvActDescriptor;

typedef ConvActDescriptor *infiniopConvActDescriptor_t;

__C __export infiniopStatus_t infiniopCreateConvActDescriptor(infiniopHandle_t handle,
                                                              infiniopConvActDescriptor_t *desc_ptr,
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

__C __export infiniopStatus_t infiniopGetConvActWorkspaceSize(infiniopConvActDescriptor_t desc, uint64_t *size);

__C __export infiniopStatus_t infiniopConvAct(infiniopConvActDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x, void const *w, void const *b, void *stream);

__C __export infiniopStatus_t infiniopDestroyConvActDescriptor(infiniopConvActDescriptor_t desc);


#endif
