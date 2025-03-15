#include "../utils.h"
#include "operators.h"
#include "ops/gather/gather.h"

#ifdef ENABLE_CPU
#include "cpu/gather_cpu.h"
#endif

__C infiniopStatus_t infiniopCreateGatherDescriptor(
    infiniopHandle_t handle,
    infiniopGatherDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t data_desc,
    infiniopTensorDescriptor_t index_desc,
    int axis) {

    switch (handle->device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuCreateGatherDescriptor(handle, (GatherCpuDescriptor_t *)desc_ptr, output_desc, data_desc, index_desc, axis);
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopGather(infiniopGatherDescriptor_t desc, void *output, void *data,
                                    void const *indices, void *stream) {

    switch (desc->device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuGather((GatherCpuDescriptor_t)desc, output, data, indices, stream);
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyGatherDescriptor(infiniopGatherDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuDestroyGatherDescriptor((GatherCpuDescriptor_t)desc);
#endif
    }
    return STATUS_BAD_DEVICE;
}