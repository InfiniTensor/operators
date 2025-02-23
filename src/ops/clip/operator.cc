#include "../utils.h"
#include "operators.h"
#include "ops/clip/clip.h"

#ifdef ENABLE_CPU
#include "cpu/clip_cpu.h"
#endif

__C infiniopStatus_t infiniopCreateClipDescriptor(infiniopHandle_t handle, infiniopClipDescriptor_t *desc_ptr,
                                                  infiniopTensorDescriptor_t output_desc,
                                                  infiniopTensorDescriptor_t intput_desc,
                                                  infiniopTensorDescriptor_t min_desc,
                                                  infiniopTensorDescriptor_t max_desc) {
    switch (handle->device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuCreateClipDescriptor(handle, (ClipCpuDescriptor_t *)desc_ptr, output_desc, intput_desc,
                                       min_desc, max_desc);
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopClip(infiniopClipDescriptor_t desc, void *output, void *input, void const *min,
                                  void const *max, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuClip((ClipCpuDescriptor_t)desc, output, input, min, max, stream);
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyClipDescriptor(infiniopClipDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuDestroyClipDescriptor((ClipCpuDescriptor_t)desc);
#endif
    }
    return STATUS_BAD_DEVICE;
}