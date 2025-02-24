#include "../utils.h"
#include "operators.h"
#include "ops/gather/gather.h"

#ifdef ENABLE_CPU
#include "cpu/gather_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/gather.cuh"
#endif

__C infiniopStatus_t infiniopCreateGatherDescriptor(
    infiniopHandle_t handle,
    infiniopGatherDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input,
    infiniopTensorDescriptor_t indices,
    uint64_t axis) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateGatherDescriptor(handle, (GatherCpuDescriptor_t *) desc_ptr, output, input, indices, axis);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateGatherDescriptor((CudaHandle_t) handle, (GatherCudaDescriptor_t *) desc_ptr, output, input, indices, axis);
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

__C infiniopStatus_t infiniopGather(
    infiniopGatherDescriptor_t desc,
    void *output,
    void const *input,
    void const *indices,
    void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuGather((GatherCpuDescriptor_t) desc, output, input, indices, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaGather((GatherCudaDescriptor_t) desc, output, input, indices, stream);
        }

#endif
        default:
            return STATUS_BAD_DEVICE;
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
            return cudaDestroyGatherDescriptor((GatherCudaDescriptor_t) desc);
        }

#endif
        default:
            return STATUS_BAD_DEVICE;
    }
    return STATUS_BAD_DEVICE;
}
