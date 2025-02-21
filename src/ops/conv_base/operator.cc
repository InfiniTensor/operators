#include "../utils.h"
#include "conv_base.h"
#include "operators.h"

#ifdef ENABLE_CPU
#include "cpu/conv_base_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/conv_base.cuh"
#endif

__C infiniopStatus_t infiniopCreateConvBaseDescriptor(
    infiniopHandle_t handle,
    infiniopConvBaseDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t w,
    uint64_t const *pads,
    int64_t const *strides,
    uint64_t const *dilations,
    uint64_t n) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateConvBaseDescriptor(handle, (ConvBaseCpuDescriptor_t *) desc_ptr, y, x, w, pads, strides, dilations, n);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateConvBaseDescriptor((CudaHandle_t) handle, (ConvBaseCudaDescriptor_t *) desc_ptr, y, x, w, pads, strides, dilations, n);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopGetConvBaseWorkspaceSize(infiniopConvBaseDescriptor_t desc, uint64_t *size) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuGetConvBaseWorkspaceSize((ConvBaseCpuDescriptor_t) desc, size);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaGetConvBaseWorkspaceSize((ConvBaseCudaDescriptor_t) desc, size);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopConvBase(infiniopConvBaseDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x, void const *w, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuConvBase((ConvBaseCpuDescriptor_t) desc, workspace, workspace_size, y, x, w, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaConvBase((ConvBaseCudaDescriptor_t) desc, workspace, workspace_size, y, x, w, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyConvBaseDescriptor(infiniopConvBaseDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyConvBaseDescriptor((ConvBaseCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyConvBaseDescriptor((ConvBaseCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}
