#include "../utils.h"
#include "operators.h"

#include "ops/clip/clip.h"

#ifdef ENABLE_CPU
#include "cpu/clip_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/clip.cuh"
#endif

__C infiniopStatus_t infiniopCreateClipDescriptor(
    infiniopHandle_t handle,
    infiniopClipDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t dst,
    infiniopTensorDescriptor_t src,
    float max, 
    float min){ 
    switch (handle->device)
    {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuCreateClipDescriptor(handle, (ClipCpuDescriptor_t *)desc_ptr, dst, src, max, min);
#endif        
#ifdef ENABLE_NV_GPU
    case DevNvGpu: 
        return cudaCreateClipDescriptor((CudaHandle_t) handle, (ClipCudaDescriptor_t *) desc_ptr, dst, src, max, min);
        
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopClip(infiniopClipDescriptor_t desc,
    void *dst, 
    void const *src,
    void *stream){
    switch (desc->device)
    {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuClip((ClipCpuDescriptor_t)desc, dst, src, stream);
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu: 
        return cudaClip((ClipCudaDescriptor_t) desc, dst, src, stream);
        

#endif        
    }
   return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyClipDescriptor(infiniopClipDescriptor_t desc){
    switch (desc->device)
    {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuDestroyClipDescriptor((ClipCpuDescriptor_t)desc);
#endif        
#ifdef ENABLE_NV_GPU
    case DevNvGpu: 
        return cudaDestroyClipDescriptor((ClipCudaDescriptor_t) desc);
    

#endif
    }
   return STATUS_BAD_DEVICE;
}