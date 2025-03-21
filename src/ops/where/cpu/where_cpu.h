#ifndef __CPU_WHERE_H__
#define __CPU_WHERE_H__

#include "operators.h"
#include <numeric>
#include <type_traits>

struct WhereCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t ndim;
    uint64_t o_data_size;
    uint64_t const *o_shape;
    uint64_t const *x_strides;
    uint64_t const *y_strides;
    uint64_t *o_indices;
};

typedef struct WhereCpuDescriptor *WhereCpuDescriptor_t;

infiniopStatus_t cpuCreateWhereDescriptor(infiniopHandle_t,
                                          WhereCpuDescriptor_t *,
                                          infiniopTensorDescriptor_t output,
                                          infiniopTensorDescriptor_t condition,
                                          infiniopTensorDescriptor_t x,
                                          infiniopTensorDescriptor_t y);

infiniopStatus_t cpuWhere(WhereCpuDescriptor_t desc,
                          void *output,
                          void const *condition,
                          void const *x,
                          void const *y,
                          void *stream);

infiniopStatus_t cpuDestroyWhereDescriptor(WhereCpuDescriptor_t desc);

#endif
