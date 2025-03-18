#ifndef __CUDA_WHERE_H__
#define __CUDA_WHERE_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <cuda_fp16.h>
#include <numeric>

struct WhereCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    uint64_t output_size;
    bool x_broadcast;
    uint64_t x_ndim;
    uint64_t const *x_shapes;
    int64_t const *x_strides;
    bool y_broadcast;
    uint64_t y_ndim;
    uint64_t const *y_shapes;
    int64_t const *y_strides;
    bool condition_broadcast;
    uint64_t condition_ndim;
    uint64_t const *condition_shapes;
    int64_t const *condition_strides;
    uint64_t max_grid_size;
};

typedef struct WhereCudaDescriptor *WhereCudaDescriptor_t;

infiniopStatus_t cudaCreateWhereDescriptor(CudaHandle_t,
                                           WhereCudaDescriptor_t *,
                                           infiniopTensorDescriptor_t output,
                                           infiniopTensorDescriptor_t x,
                                           infiniopTensorDescriptor_t y,
                                           infiniopTensorDescriptor_t condition);

infiniopStatus_t cudaWhere(WhereCudaDescriptor_t desc,
                           void *output,
                           void const *x, void const *y, void const *condition,
                           void *stream);

infiniopStatus_t cudaDestroyWhereDescriptor(WhereCudaDescriptor_t desc);

#endif
