#include "../utils.h"
#include "operators.h"
#include "ops/where/where.h"

#ifdef ENABLE_CPU
#include "cpu/where_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#endif

__C infiniopStatus_t infiniopCreateWhereDescriptor(infiniopHandle_t handle,
                                                   infiniopWhereDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t output,
                                                   infiniopTensorDescriptor_t condition,
                                                   infiniopTensorDescriptor_t x,
                                                   infiniopTensorDescriptor_t y) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateWhereDescriptor(handle, (WhereCpuDescriptor_t *) desc_ptr, output, condition, x, y);
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

__C infiniopStatus_t infiniopWhere(infiniopWhereDescriptor_t desc,
                                   void *output,
                                   void const *condition,
                                   void const *x,
                                   void const *y,
                                   void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuWhere((WhereCpuDescriptor_t) desc, output, condition, x, y, stream);
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

__C infiniopStatus_t infiniopDestroyWhereDescriptor(infiniopWhereDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyWhereDescriptor((WhereCpuDescriptor_t) desc);
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
