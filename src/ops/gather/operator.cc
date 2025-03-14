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
    infiniopTensorDescriptor_t dst,
    infiniopTensorDescriptor_t data,
    infiniopTensorDescriptor_t indices,
    int axis){ 
    switch (handle->device)
    {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuCreateGatherDescriptor(handle, (GatherCpuDescriptor_t *)desc_ptr, dst, data, indices, axis);
#endif        
#ifdef ENABLE_NV_GPU
    case DevNvGpu:
        return cudaCreateGatherDescriptor((CudaHandle_t)handle, (GatherCudaDescriptor_t *)desc_ptr, dst, data, indices, axis);
#endif        
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopGather(infiniopGatherDescriptor_t desc,
    void *dst, 
    void const *data,
    void const *indices,
    void *stream){
    switch (desc->device)
    {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuGather((GatherCpuDescriptor_t)desc, dst, data, indices, stream);
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu:
        return cudaGather((GatherCudaDescriptor_t)desc, dst, data, indices, stream);
#endif          
    }
   return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyGatherDescriptor(infiniopGatherDescriptor_t desc){
    switch (desc->device)
    {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuDestroyGatherDescriptor((GatherCpuDescriptor_t)desc);
#endif        
#ifdef ENABLE_NV_GPU
    case DevNvGpu:
        return cudaDestroyGatherDescriptor((GatherCudaDescriptor_t)desc);
#endif        
    }
   return STATUS_BAD_DEVICE;
}