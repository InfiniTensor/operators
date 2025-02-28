#include "../utils.h"
#include "operators.h"
#include "reduce.h"

#ifdef ENABLE_CPU
#include "cpu/reduce_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/reduce.cuh"
#endif

__C infiniopStatus_t infiniopCreateReduceDescriptor(
    infiniopHandle_t handle,
    infiniopReduceDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    int *axes,
    uint64_t axes_ndim,
    int reduce_op) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateReduceDescriptor(handle, (ReduceCpuDescriptor_t *) desc_ptr, y, x, axes, axes_ndim, reduce_op);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateReduceDescriptor((CudaHandle_t) handle, (ReduceCudaDescriptor_t *) desc_ptr, y, x, axes, axes_ndim, reduce_op);
        }

#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopGetReduceWorkspaceSize(infiniopReduceDescriptor_t desc, uint64_t *size) {
    // std::cout << desc->device << std::endl;
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu: {
            return STATUS_SUCCESS;
        }
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaGetReduceWorkspaceSize((ReduceCudaDescriptor_t) desc, size);
        }

#endif

        default:
            return STATUS_BAD_DEVICE;
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopReduce(infiniopReduceDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuReduce((ReduceCpuDescriptor_t) desc, y, x, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: 
            return cudaReduce((ReduceCudaDescriptor_t) desc, workspace, workspace_size, y, x, stream);

#endif
        default:
            return STATUS_BAD_DEVICE;
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyReduceDescriptor(infiniopReduceDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyReduceDescriptor((ReduceCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyReduceDescriptor((ReduceCudaDescriptor_t) desc);
        }

#endif
    }
    return STATUS_BAD_DEVICE;
}
