#ifndef __CUDA_CLIP_H__
#define __CUDA_CLIP_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <cuda_fp16.h>
#include <numeric>
#include <optional>

struct ClipCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    uint64_t data_size;
    bool has_lower_bound;
    float lower_bound;
    bool has_upper_bound;
    float upper_bound;
    uint64_t max_grid_size;
};

typedef struct ClipCudaDescriptor *ClipCudaDescriptor_t;

infiniopStatus_t cudaCreateClipDescriptor(CudaHandle_t,
                                          ClipCudaDescriptor_t *,
                                          infiniopTensorDescriptor_t y,
                                          infiniopTensorDescriptor_t x,
                                          float* lower_bound,
                                          float* upper_bound);

infiniopStatus_t cudaClip(ClipCudaDescriptor_t desc,
                          void *y, void const *x,
                          void *stream);

infiniopStatus_t cudaDestroyClipDescriptor(ClipCudaDescriptor_t desc);

#endif
