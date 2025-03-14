#include "../utils.h"
#include "operators.h"

#include "ops/gather_elements/gather_elements.h"

#ifdef ENABLE_CPU
#include "cpu/gather_elements_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/gather_elements.cuh"
#endif

__C infiniopStatus_t infiniopCreateGatherElementsDescriptor(
    infiniopHandle_t handle,
    infiniopGatherElementsDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t dst,
    infiniopTensorDescriptor_t data,
    infiniopTensorDescriptor_t indices,
    int axis){ 
    switch (handle->device)
    {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuCreateGatherElementsDescriptor(handle, (GatherElementsCpuDescriptor_t *)desc_ptr, dst, data, indices, axis);
#endif        
#ifdef ENABLE_NV_GPU
    case DevNvGpu:
        return cudaCreateGatherElementsDescriptor((CudaHandle_t)handle, (GatherElementsCudaDescriptor_t *)desc_ptr, dst, data, indices, axis);
#endif        
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopGatherElements(infiniopGatherElementsDescriptor_t desc,
    void *dst, 
    void const *data,
    void const *indices,
    void *stream){
    switch (desc->device)
    {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuGatherElements((GatherElementsCpuDescriptor_t)desc, dst, data, indices, stream);
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu:
        return cudaGatherElements((GatherElementsCudaDescriptor_t)desc, dst, data, indices, stream);
#endif          
    }
   return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyGatherElementsDescriptor(infiniopGatherElementsDescriptor_t desc){
    switch (desc->device)
    {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuDestroyGatherElementsDescriptor((GatherElementsCpuDescriptor_t)desc);
#endif        
#ifdef ENABLE_NV_GPU
    case DevNvGpu:
        return cudaDestroyGatherElementsDescriptor((GatherElementsCudaDescriptor_t)desc);
#endif        
    }
   return STATUS_BAD_DEVICE;
}