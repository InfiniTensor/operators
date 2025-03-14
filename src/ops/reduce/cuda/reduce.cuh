#ifndef __CUDA_REDUCE_H__
#define __CUDA_REDUCE_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <cuda_fp16.h>
#include <numeric>

struct ReduceCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    uint64_t ndim;
    uint64_t dst_size;
    uint64_t data_size;
    uint64_t add_dim;
    int axis_num;
    int64_t* stride_axis_ptr;
    int64_t* shape_axis_ptr;
    int reduce_mode;
    uint64_t max_grid_size;
};

typedef struct ReduceCudaDescriptor *ReduceCudaDescriptor_t;

infiniopStatus_t cudaCreateReduceDescriptor(CudaHandle_t,
                                         ReduceCudaDescriptor_t *,
                                         infiniopTensorDescriptor_t dst,
                                         infiniopTensorDescriptor_t src,
                                         int *axis,
                                         int num_axis,
                                         int keepdims,
                                         int reduce_type);

infiniopStatus_t cudaGetReduceWorkspaceSize(ReduceCudaDescriptor_t desc, uint64_t *size);                                         

infiniopStatus_t cudaReduce(ReduceCudaDescriptor_t desc,
                            void *workspace, 
                            uint64_t workspace_size,
                            void *dst,     
                            void const *src,
                            void *stream);

infiniopStatus_t cudaDestroyReduceDescriptor(ReduceCudaDescriptor_t desc);

#endif
