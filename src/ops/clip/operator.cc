#include "../utils.h"
#include "operators.h"
#include "ops/clip/clip.h"

#ifdef ENABLE_CPU
#include "cpu/clip_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
// #include "../../devices/cuda/cuda_handle.h"
// #include "cuda/clip.cuh"
#endif
#ifdef ENABLE_MTHREADS_GPU
// #include "musa/clip_musa.h"
#endif

__C infiniopStatus_t infiniopCreateClipDescriptor(infiniopHandle_t handle,
                                                  infiniopClipDescriptor_t *desc_ptr,
                                                  infiniopTensorDescriptor_t output,
                                                  infiniopTensorDescriptor_t input,
                                                  float min,
                                                  float max) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateClipDescriptor(handle, (ClipCpuDescriptor_t *) desc_ptr, output, input, min, max);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
#ifdef ENABLE_MTHREADS_GPU
        case DevMthreadsGpu: {
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopClip(infiniopClipDescriptor_t desc,
                                  void *output,
                                  void const *input,
                                  void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuClip((ClipCpuDescriptor_t) desc, output, input, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
#ifdef ENABLE_MTHREADS_GPU
        case DevMthreadsGpu: {
        }
#endif
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
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
#ifdef ENABLE_MTHREADS_GPU
        case DevMthreadsGpu: {
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}
