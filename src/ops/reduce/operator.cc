#include "../utils.h"
#include "operators.h"
#include "reduce.h"

#ifdef ENABLE_CPU
#include "cpu/reduce_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#endif
#ifdef ENABLE_MTHREADS_GPU
#endif

__C infiniopStatus_t infiniopCreateReduceDescriptor(infiniopHandle_t handle,
                                                    infiniopReduceDescriptor_t *desc_ptr,
                                                    infiniopTensorDescriptor_t reduced,
                                                    infiniopTensorDescriptor_t data,
                                                    int64_t *axes,
                                                    uint64_t axes_ndim,
                                                    bool keepdims,
                                                    bool noop_with_empty_axes,
                                                    int reduce_type) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateReduceDescriptor(handle, (ReduceCpuDescriptor_t *) desc_ptr,
                                             reduced, data, axes, axes_ndim, keepdims, noop_with_empty_axes, reduce_type);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
#ifdef ENABLE_MTHREADS_GPU
        case DevMthreadsGpu: {
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
        }
#endif
        default:
            return STATUS_BAD_DEVICE;
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopReduce(infiniopReduceDescriptor_t desc,
                                    void *reduced,
                                    void const *data,
                                    void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuReduce((ReduceCpuDescriptor_t) desc, reduced, data, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
#ifdef ENABLE_MTHREADS_GPU
        case DevMthreadsGpu: {
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
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
#ifdef ENABLE_MTHREADS_GPU
        case DevMthreadsGpu: {
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}
