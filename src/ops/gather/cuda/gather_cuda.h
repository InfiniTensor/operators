#ifndef __CUDA_GATHER_H__
#define __CUDA_GATHER_H__

#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <memory>

typedef struct GatherCudaDescriptor {
    Device device;
    DT dtype;
    DT indices_dtype;
    int64_t axis;
    int otherDims;
    uint64_t input_ndim;
    uint64_t output_ndim;
    int dim_size;
    int indices_size;
    int stride;

} GatherCudaDescriptor;

typedef struct GatherCudaDescriptor *GatherCudaDescriptor_t;

infiniopStatus_t cudaCreateGatherDescriptor(CudaHandle_t handle,
                                            GatherCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t y,
                                            infiniopTensorDescriptor_t x,
                                            infiniopTensorDescriptor_t indices,
                                            int64_t axis
                                            );


infiniopStatus_t cudaGather(GatherCudaDescriptor_t desc,
                            void const *x,
                            void const *indices,
                            void *y,
                            void *stream);

infiniopStatus_t cudaDestroyGatherDescriptor(GatherCudaDescriptor_t desc);

#endif// __CUDA_MATMUL_H__
