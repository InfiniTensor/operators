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
    uint64_t ndim;
    uint64_t data_size;
    uint64_t max_grid_size;
};

typedef struct WhereCudaDescriptor *WhereCudaDescriptor_t;

infiniopStatus_t cudaCreateWhereDescriptor(CudaHandle_t,
                                         WhereCudaDescriptor_t *,
                                         infiniopTensorDescriptor_t dst,
                                         infiniopTensorDescriptor_t x,
                                         infiniopTensorDescriptor_t y,
                                         infiniopTensorDescriptor_t condition);

infiniopStatus_t cudaWhere(WhereCudaDescriptor_t desc,
                            void *dst, 
                            void const *x,
                            void const *y,
                            void const *condition,
                            void *stream);

infiniopStatus_t cudaDestroyWhereDescriptor(WhereCudaDescriptor_t desc);

#endif
