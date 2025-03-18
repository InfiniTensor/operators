#ifndef __CPU_WHERE_H__
#define __CPU_WHERE_H__

#include "operators.h"
#include <cstring>
#include <numeric>

struct WhereCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t output_size;
    bool x_broadcast;
    uint64_t x_ndim;
    uint64_t *x_shapes;
    int64_t *x_strides;
    bool y_broadcast;
    uint64_t y_ndim;
    uint64_t *y_shapes;
    int64_t *y_strides;
    bool condition_broadcast;
    uint64_t condition_ndim;
    uint64_t *condition_shapes;
    int64_t *condition_strides;
};

typedef struct WhereCpuDescriptor *WhereCpuDescriptor_t;

infiniopStatus_t cpuCreateWhereDescriptor(infiniopHandle_t,
                                          WhereCpuDescriptor_t *,
                                          infiniopTensorDescriptor_t output,
                                          infiniopTensorDescriptor_t x,
                                          infiniopTensorDescriptor_t y,
                                          infiniopTensorDescriptor_t condition);

infiniopStatus_t cpuWhere(WhereCpuDescriptor_t desc,
                          void *output,
                          void const *x,
                          void const *y,
                          void const *condition);

infiniopStatus_t cpuDestroyWhereDescriptor(WhereCpuDescriptor_t desc);

#endif
