#include "../utils.h"
#include "operators.h"
#include "ops/clip/clip.h"

#ifdef ENABLE_CPU
#include "cpu/clip_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/clip.cuh"
#endif

__C infiniopStatus_t infiniopCreateClipDescriptor(
    infiniopHandle_t handle,
    infiniopClipDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    float* lower_bound,
    float* upper_bound) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateClipDescriptor(handle, (ClipCpuDescriptor_t *) desc_ptr, y, x, lower_bound, upper_bound);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateClipDescriptor((CudaHandle_t) handle, (ClipCudaDescriptor_t *) desc_ptr, y, x, lower_bound, upper_bound);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
        default:
            return STATUS_BAD_DEVICE;
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopClip(
    infiniopClipDescriptor_t desc,
    void *y,
    void const *x,
    void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuClip((ClipCpuDescriptor_t) desc, y, x);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaClip((ClipCudaDescriptor_t) desc, y, x, stream);
        }

#endif
        default:
            return STATUS_BAD_DEVICE;
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyClipDescriptor(infiniopClipDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyClipDescriptor((ClipCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyClipDescriptor((ClipCudaDescriptor_t) desc);
        }

#endif
        default:
            return STATUS_BAD_DEVICE;
    }
    return STATUS_BAD_DEVICE;
}
