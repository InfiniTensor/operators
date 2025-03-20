#include "../utils.h"
#include "operators.h"
#include "ops/where/where.h"

#ifdef ENABLE_CPU
#include "cpu/where_cpu.h"
#endif

#ifdef ENABLE_NV_GPU
#include "cuda/where_cuda.h"
#endif


__C infiniopStatus_t infiniopCreateWhereDescriptor(
    infiniopHandle_t handle,
    infiniopWhereDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t dst,
    infiniopTensorDescriptor_t src1,
    infiniopTensorDescriptor_t src2,
    infiniopTensorDescriptor_t condition
    ) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateWhereDescriptor(handle, (WhereCpuDescriptor_t *) desc_ptr, dst, src1, src2, condition);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateWhereDescriptor((CudaHandle_t) handle, (WhereCudaDescriptor_t *) desc_ptr, dst, src1, src2, condition);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}


__C infiniopStatus_t infiniopWhere(infiniopWhereDescriptor_t desc, void *dst, void const *src1, void const *src2, void const *condition, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuWhere((WhereCpuDescriptor_t) desc, dst, src1, src2, condition, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return cudaWhere((WhereCudaDescriptor_t) desc, dst, src1, src2, condition, stream);
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyWhereDescriptor(infiniopWhereDescriptor_t desc){
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyWhereDescriptor((WhereCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return cudaDestroyWhereDescriptor((WhereCudaDescriptor_t) desc);
#endif
    }
    return STATUS_BAD_DEVICE;
}