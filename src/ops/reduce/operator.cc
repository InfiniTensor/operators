#include "../utils.h"
#include "operators.h"
#include "reduce.h"

#ifdef ENABLE_CPU
#include "cpu/reduce_cpu.h"
#endif

__C infiniopStatus_t infiniopCreateReduceDescriptor(
    infiniopHandle_t handle,
    infiniopReduceDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    int64_t const *axes,
    uint64_t n_axes,
    int keep_dims,
    int reduce_type) {
    if (handle->device == DevCpu) {
        return cpuCreateReduceDescriptor(handle, (ReduceCpuDescriptor_t *) desc_ptr, y, x, axes, n_axes, keep_dims, reduce_type);
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopReduce(infiniopReduceDescriptor_t desc, void *y, void const *x, void *stream) {
    if (desc->device == DevCpu) {
        return cpuReduce((ReduceCpuDescriptor_t) desc, y, x, stream);
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyReduceDescriptor(infiniopReduceDescriptor_t desc) {
    if (desc->device == DevCpu) {
        return cpuDestroyReduceDescriptor((ReduceCpuDescriptor_t) desc);
    }
    return STATUS_BAD_DEVICE;
}
