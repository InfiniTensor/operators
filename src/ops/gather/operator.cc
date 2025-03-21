#include "../utils.h"
#include "operators.h"
#include "ops/gather/gather.h"

#ifdef ENABLE_CPU
#include "cpu/gather_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#endif
#ifdef ENABLE_MTHREADS_GPU
#endif

__C infiniopStatus_t infiniopCreateGatherDescriptor(
    infiniopHandle_t handle,
    infiniopGatherDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t data,
    infiniopTensorDescriptor_t indices,
    int64_t axis) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateGatherDescriptor(handle, (GatherCpuDescriptor_t *) desc_ptr, output, data, indices, axis);
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

__C infiniopStatus_t infiniopGather(infiniopGatherDescriptor_t desc, void *output, void const *data, void const *indices, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuGather((GatherCpuDescriptor_t) desc, output, data, indices, stream);
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

__C infiniopStatus_t infiniopDestroyGatherDescriptor(infiniopGatherDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyGatherDescriptor((GatherCpuDescriptor_t) desc);
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
