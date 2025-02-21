#include "../conv_base/conv_base.h"
#include "../utils.h"
#include "ops/conv/conv.h"
#include "ops/conv_act/conv_act.h"

struct _ConvDescriptor {
    Device device;
    infiniopConvBaseDescriptor_t conv_base_desc;
    infiniopConvActDescriptor_t conv_act_desc;
};

typedef struct _ConvDescriptor *_ConvDescriptor_t;

__C infiniopStatus_t infiniopCreateConvDescriptor(infiniopHandle_t handle,
                                                  infiniopConvDescriptor_t *desc_ptr,
                                                  infiniopTensorDescriptor_t y,
                                                  infiniopTensorDescriptor_t x,
                                                  infiniopTensorDescriptor_t w,
                                                  infiniopTensorDescriptor_t b,
                                                  uint64_t const *pads,
                                                  int64_t const *strides,
                                                  uint64_t const *dilations,
                                                  uint64_t n) {
    infiniopConvBaseDescriptor_t conv_base_desc = nullptr;
    infiniopConvActDescriptor_t conv_act_desc = nullptr;
    if (!b) {
        CHECK_STATUS(infiniopCreateConvBaseDescriptor(handle, &conv_base_desc, y, x, w, pads, strides, dilations, n), STATUS_SUCCESS);
    } else {
        ConvActParam_t act_params;
        CHECK_STATUS(infiniopCreateConvActDescriptor(handle, &conv_act_desc, y, x, w, b, pads, strides, dilations, n, INFINI_ACTIVATION_IDENTITY, act_params), STATUS_SUCCESS);
    }

    // create descriptor
    *(_ConvDescriptor_t *) desc_ptr = new _ConvDescriptor{
        handle->device,
        conv_base_desc,
        conv_act_desc,
    };

    return STATUS_SUCCESS;
}

__C infiniopStatus_t infiniopGetConvWorkspaceSize(infiniopConvDescriptor_t desc, uint64_t *size) {
    _ConvDescriptor_t _conv_desc = (_ConvDescriptor_t) desc;
    if (_conv_desc->conv_base_desc) {
        CHECK_STATUS(infiniopGetConvBaseWorkspaceSize(_conv_desc->conv_base_desc, size), STATUS_SUCCESS);
    } else {
        CHECK_STATUS(infiniopGetConvActWorkspaceSize(_conv_desc->conv_act_desc, size), STATUS_SUCCESS);
    }
    return STATUS_SUCCESS;
}

__C infiniopStatus_t infiniopConv(infiniopConvDescriptor_t desc,
                                  void *workspace,
                                  uint64_t workspace_size,
                                  void *y,
                                  void const *x,
                                  void const *w,
                                  void const *b,
                                  void *stream) {
    _ConvDescriptor_t _conv_desc = (_ConvDescriptor_t) desc;
    if (_conv_desc->conv_base_desc) {
        CHECK_STATUS(infiniopConvBase(_conv_desc->conv_base_desc, workspace, workspace_size, y, x, w, stream), STATUS_SUCCESS);
    } else {
        if (!b) {
            WARN("The bias descriptor has been initialized, but no bias data is provided. The computation will proceed as if there is no bias and continue as far as possible.");
        }
        CHECK_STATUS(infiniopConvAct(_conv_desc->conv_act_desc, workspace, workspace_size, y, x, w, b, stream), STATUS_SUCCESS);
    }
    return STATUS_SUCCESS;
}

__C infiniopStatus_t infiniopDestroyConvDescriptor(infiniopConvDescriptor_t desc) {
    _ConvDescriptor_t _conv_desc = (_ConvDescriptor_t) desc;
    if (_conv_desc->conv_base_desc) {
        CHECK_STATUS(infiniopDestroyConvBaseDescriptor(_conv_desc->conv_base_desc), STATUS_SUCCESS);
    } else {
        CHECK_STATUS(infiniopDestroyConvActDescriptor(_conv_desc->conv_act_desc), STATUS_SUCCESS);
    }
    delete desc;
    return STATUS_SUCCESS;
}
