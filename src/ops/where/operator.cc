#include "../utils.h"
#include "operators.h"
#include "ops/where/where.h"

#ifdef ENABLE_CPU
#include "cpu/where_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/where.cuh"
#endif

__C infiniopStatus_t infiniopCreateWhereDescriptor(
    infiniopHandle_t handle,
    infiniopWhereDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t condition) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateWhereDescriptor(handle, (WhereCpuDescriptor_t *) desc_ptr, output, x, y, condition);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateWhereDescriptor((CudaHandle_t) handle, (WhereCudaDescriptor_t *) desc_ptr, output, x, y, condition);
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

__C infiniopStatus_t infiniopWhere(
    infiniopWhereDescriptor_t desc,
    void *output,
    void const *x,
    void const *y,
    void const *condition,
    void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuWhere((WhereCpuDescriptor_t) desc, output, x, y, condition);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaWhere((WhereCudaDescriptor_t) desc, output, x, y, condition, stream);
        }

#endif
        default:
            return STATUS_BAD_DEVICE;
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
            return cudaDestroyWhereDescriptor((WhereCudaDescriptor_t) desc);
        }

#endif
        default:
            return STATUS_BAD_DEVICE;
    }
    return STATUS_BAD_DEVICE;
}
