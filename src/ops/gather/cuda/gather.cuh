#ifndef __CUDA_GATHER_H__
#define __CUDA_GATHER_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <cuda_fp16.h>
#include <numeric>

struct GatherCudaDescriptor {
    Device device;
    DT dtype;
    DT indices_dtype;
    int device_id;
    uint64_t output_size;
    uint64_t pre_size;
    uint64_t axis_size;
    uint64_t indices_size;
    uint64_t post_size;
    int kernel_type;
    uint64_t max_grid_size;
};

typedef struct GatherCudaDescriptor *GatherCudaDescriptor_t;

infiniopStatus_t cudaCreateGatherDescriptor(CudaHandle_t,
                                            GatherCudaDescriptor_t *,
                                            infiniopTensorDescriptor_t output,
                                            infiniopTensorDescriptor_t input,
                                            infiniopTensorDescriptor_t indices,
                                            uint64_t axis);

infiniopStatus_t cudaGather(GatherCudaDescriptor_t desc,
                            void *output, void const *input,
                            void const *indices, void *stream);

infiniopStatus_t cudaDestroyGatherDescriptor(GatherCudaDescriptor_t desc);

#endif
