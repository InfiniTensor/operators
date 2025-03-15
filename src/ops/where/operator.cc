#include "../utils.h"
#include "operators.h"
#include "ops/where/where.h"

#ifdef ENABLE_CPU
#include "cpu/where_cpu.h"
#endif

__C infiniopStatus_t infiniopCreateWhereDescriptor(infiniopHandle_t handle,
    infiniopWhereDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t condition_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc) {
    switch (handle->device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuCreateWhereDescriptor(handle, (WhereCpuDescriptor_t *)desc_ptr, output_desc, condition_desc,
                                        x_desc, y_desc);
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C __export infiniopStatus_t infiniopWhere(infiniopWhereDescriptor_t desc, void *output, void *condition,
                                            void *x, void *y, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuWhere((WhereCpuDescriptor_t)desc, output, condition, x, y, stream);
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C __export infiniopStatus_t infiniopDestroyWhereDescriptor(infiniopWhereDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuDestroyWhereDescriptor((WhereCpuDescriptor_t)desc);
#endif
    }
    return STATUS_BAD_DEVICE;
}
