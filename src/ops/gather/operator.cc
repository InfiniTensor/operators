#include "../utils.h"
#include "operators.h"
#include "ops/gather/gather.h"

#ifdef ENABLE_CPU
#include "cpu/gather_cpu.h"
#endif

#ifdef ENABLE_NV_GPU
#include "cuda/gather_cuda.h"
#endif

__C infiniopStatus_t infiniopCreateGatherDescriptor(
    infiniopHandle_t handle,
    infiniopGatherDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t indices,
    int64_t axis
    ) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateGatherDescriptor(handle, (GatherCpuDescriptor_t *) desc_ptr, y, x, indices, axis);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateGatherDescriptor((CudaHandle_t) handle, (GatherCudaDescriptor_t *) desc_ptr, y, x, indices, axis);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}
__C infiniopStatus_t infiniopGather(infiniopGatherDescriptor_t desc, void const *x, void const *indices, void *y, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuGather((GatherCpuDescriptor_t) desc, x, indices, y, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return cudaGather((GatherCudaDescriptor_t) desc, x, indices, y, stream);
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyGatherDescriptor(infiniopGatherDescriptor_t desc){
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyGatherDescriptor((GatherCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return cudaDestroyGatherDescriptor((GatherCudaDescriptor_t) desc);
#endif
    }
    return STATUS_BAD_DEVICE;
}