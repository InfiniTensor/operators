#ifndef __CUDA_REDUCE_H__
#define __CUDA_REDUCE_H__

#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <memory>

typedef struct ReduceCudaDescriptor {
    Device device;
    DT dtype;
    uint64_t input_ndim;
    uint64_t output_ndim;
    int64_t *non_reduce_axes;
    int64_t *input_strides;
    int64_t *output_strides;
    uint64_t *input_shape;
    uint64_t *output_shape;
    int64_t *reduce_axes;
    uint64_t *reduce_axes_stride;
    uint64_t reduce_size;
    uint64_t element_num;
    uint64_t output_size;
    int reduce_op_type;
    int reduce_mode; // output_size * reduce_size = element_num
    uint64_t axes_size;
    bool keepdims;
    int64_t start_axis;
    int64_t end_axis;
    int prefix_size;
    int suffix_size;
} ReduceCudaDescriptor;

typedef struct ReduceCudaDescriptor *ReduceCudaDescriptor_t;


enum ReduceOpType {
    REDUCE_SUM = 0,
    REDUCE_MIN = 1,
    REDUCE_MAX = 2,
    REDUCE_MEAN = 3
};

infiniopStatus_t cudaCreateReduceDescriptor(CudaHandle_t handle,
                                            ReduceCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t x,
                                            infiniopTensorDescriptor_t y,
                                            int64_t const *axes,
                                            uint64_t axes_size,
                                            int reduce_op_type,
                                            bool keepdims
                                            );


infiniopStatus_t cudaReduce(ReduceCudaDescriptor_t desc,
    void *y,
    void const *x,
    void *stream);

infiniopStatus_t cudaDestroyReduceDescriptor(ReduceCudaDescriptor_t desc);

#endif// __CUDA_MATMUL_H__
