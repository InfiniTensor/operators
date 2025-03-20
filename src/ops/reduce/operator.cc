#include "../utils.h"
#include "operators.h"
#include "reduce.h"

#ifdef ENABLE_CPU
#include "cpu/reduce_cpu.h"
#endif

#ifdef ENABLE_NV_GPU
#include "cuda/reduce_cuda.h"
#endif


__C infiniopStatus_t infiniopCreateReduceDescriptor(
    infiniopHandle_t handle,
    infiniopReduceDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    int64_t const *axes,
    uint64_t n,
    bool keepdims,
    bool noop_with_empty_axes,
    int reduce_type) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateReduceDescriptor(handle, (ReduceCpuDescriptor_t *) desc_ptr, y, x, axes, n, reduce_type, noop_with_empty_axes, keepdims);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return cudaCreateReduceDescriptor((CudaHandle_t)handle, (ReduceCudaDescriptor_t *) desc_ptr, y, x, axes, n, reduce_type, keepdims);
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopReduce(infiniopReduceDescriptor_t desc, void *y, const void *x, void *dynamic_axes, uint64_t dynamic_axes_size, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuReduce((ReduceCpuDescriptor_t) desc, y, x, dynamic_axes, dynamic_axes_size, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return cudaReduce((ReduceCudaDescriptor_t) desc, y, x, stream);
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyReduceDescriptor(infiniopReduceDescriptor_t desc){
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyReduceDescriptor((ReduceCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return cudaDestroyReduceDescriptor((ReduceCudaDescriptor_t) desc);
#endif
    }
    return STATUS_BAD_DEVICE;

}