#include "../utils.h"
#include "operators.h"
#include "ops/conv_act/conv_act.h"

#ifdef ENABLE_CPU
#include "cpu/conv_act_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/conv_act.cuh"
#endif

__C infiniopStatus_t infiniopCreateConvActDescriptor(
    infiniopHandle_t handle,
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
    ConvActParam_t act_params) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateConvActDescriptor(handle, (ConvActCpuDescriptor_t *) desc_ptr, y, x, w, b, pads, strides, dilations, n, activation_mode, act_params);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateConvActDescriptor((CudaHandle_t) handle, (ConvActCudaDescriptor_t *) desc_ptr, y, x, w, b, pads, strides, dilations, n, activation_mode, act_params);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopGetConvActWorkspaceSize(infiniopConvActDescriptor_t desc, uint64_t *size) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuGetConvActWorkspaceSize((ConvActCpuDescriptor_t) desc, size);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaGetConvActWorkspaceSize((ConvActCudaDescriptor_t) desc, size);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopConvAct(infiniopConvActDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x, void const *w, void const *b, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuConvAct((ConvActCpuDescriptor_t) desc, workspace, workspace_size, y, x, w, b, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaConvAct((ConvActCudaDescriptor_t) desc, workspace, workspace_size, y, x, w, b, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyConvActDescriptor(infiniopConvActDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyConvActDescriptor((ConvActCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyConvActDescriptor((ConvActCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}
