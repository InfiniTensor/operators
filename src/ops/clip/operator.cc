#include "../utils.h"
#include "operators.h"
#include "ops/clip/clip.h"

#ifdef ENABLE_CPU
#include "cpu/clip_cpu.h"
#endif

#ifdef ENABLE_NV_GPU
#include "cuda/clip_cuda.h"
#endif


__C infiniopStatus_t infiniopCreateClipDescriptor(
    infiniopHandle_t handle,
    infiniopClipDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t y,
    float* min,
    float* max
    ) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateClipDescriptor(handle, (ClipCpuDescriptor_t *) desc_ptr, x, y, min, max);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateClipDescriptor((CudaHandle_t) handle, (ClipCudaDescriptor_t *) desc_ptr, x, y, min, max);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}


__C infiniopStatus_t infiniopClip(infiniopClipDescriptor_t desc, void const *x, void *y, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuClip((ClipCpuDescriptor_t) desc, x, y, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return cudaClip((ClipCudaDescriptor_t) desc, x, y, stream);
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyClipDescriptor(infiniopClipDescriptor_t desc){
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyClipDescriptor((ClipCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return cudaDestroyClipDescriptor((ClipCudaDescriptor_t) desc);
#endif
    }
    return STATUS_BAD_DEVICE;
}