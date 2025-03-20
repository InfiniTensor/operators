#ifndef __CUDA_CLIP_H__
#define __CUDA_CLIP_H__

#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <memory>

typedef struct ClipCudaDescriptor {
    Device device;
    DT dtype;
    uint64_t ndim;
    uint64_t element_num;
    float min;
    float max;
    bool has_min;
    bool hax_max;
} ClipCudaDescriptor;

typedef struct ClipCudaDescriptor *ClipCudaDescriptor_t;

infiniopStatus_t cudaCreateClipDescriptor(CudaHandle_t handle,
                                            ClipCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t x,
                                            infiniopTensorDescriptor_t y,
                                            float* min,
                                            float* max
                                            );


infiniopStatus_t cudaClip(ClipCudaDescriptor_t desc,
                            void const *x,
                            void *y,
                            void *stream);

infiniopStatus_t cudaDestroyClipDescriptor(ClipCudaDescriptor_t desc);

#endif// __CUDA_MATMUL_H__
