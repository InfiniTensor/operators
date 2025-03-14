#include "../utils.h"
#include "operators.h"
#include "reduce.h"

#ifdef ENABLE_CPU
#include "cpu/reduce_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/common_cuda.h"
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/reduce.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
// TODO
#endif

__C infiniopStatus_t infiniopCreateReduceDescriptor(
    infiniopHandle_t handle,
    infiniopReduceDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t dst,
    infiniopTensorDescriptor_t src,
    int *axis,
    const int num_axis,
    int const keepdims,
    int reduce_type) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateReduceDescriptor(handle, (ReduceCpuDescriptor_t *)desc_ptr, dst, src, axis, num_axis, keepdims, reduce_type);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateReduceDescriptor((CudaHandle_t) handle, (ReduceCudaDescriptor_t *) desc_ptr, dst, src, axis, num_axis, keepdims, reduce_type);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopReduce(infiniopReduceDescriptor_t desc,
                                    void *workspace, 
                                    uint64_t workspace_size,    
                                    void *dst, 
                                    void const *src,
                                    void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuReduce((ReduceCpuDescriptor_t)desc, workspace, workspace_size, dst, src, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaReduce((ReduceCudaDescriptor_t)desc, workspace, workspace_size, dst, src, stream);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopGetReduceWorkspaceSize(infiniopReduceDescriptor_t desc, uint64_t *size) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuGetReduceWorkspaceSize((ReduceCpuDescriptor_t) desc, size);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaGetReduceWorkspaceSize((ReduceCudaDescriptor_t) desc, size);
        }
#endif
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
            return cudaDestroyReduceDescriptor((ReduceCudaDescriptor_t)desc);
        }

#endif
    }
    return STATUS_BAD_DEVICE;
}
