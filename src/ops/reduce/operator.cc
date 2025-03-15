#include "../utils.h"
#include "operators.h"
#include "reduce.h"

#ifdef ENABLE_CPU
#include "cpu/reduce_cpu.h"
#endif

__C infiniopStatus_t infiniopCreateReduceDescriptor(
    infiniopHandle_t handle,
    infiniopReduceDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t reduced,
    infiniopTensorDescriptor_t data,
    const int64_t *axes,
    size_t axes_size,
    int keepdims,
    int noop_with_empty_axes,
    int reduce_type) {

    switch (handle->device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuCreateReduceDescriptor(handle,
                                         (ReduceCpuDescriptor_t *)desc_ptr,
                                         reduced,
                                         data,
                                         axes,
                                         axes_size,
                                         keepdims,
                                         noop_with_empty_axes,
                                         reduce_type);
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopReduce(
    infiniopReduceDescriptor_t desc,
    void *reduced,
    void const *data,
    int64_t const *axes,
    void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuReduce((ReduceCpuDescriptor_t)desc, reduced, data, axes, stream);
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyReduceDescriptor(infiniopReduceDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuDestroyReduceDescriptor((ReduceCpuDescriptor_t)desc);
#endif
    }
    return STATUS_BAD_DEVICE;
}