#ifndef __CPU_WHERE_H__
#define __CPU_WHERE_H__
#include "operators.h"

struct WhereCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t data_size;
};

typedef WhereCpuDescriptor *WhereCpuDescriptor_t;

infiniopStatus_t cpuCreateWhereDescriptor(infiniopHandle_t,
                                        WhereCpuDescriptor_t *,
                                        infiniopTensorDescriptor_t dst,
                                        infiniopTensorDescriptor_t x,
                                        infiniopTensorDescriptor_t y,
                                        infiniopTensorDescriptor_t condition);

infiniopStatus_t cpuWhere(WhereCpuDescriptor_t desc,
                       void *dst, 
                       void const *x,
                       void const *y,
                       void const *condition,
                       void *stream);

infiniopStatus_t cpuDestroyWhereDescriptor(WhereCpuDescriptor_t desc);
#endif