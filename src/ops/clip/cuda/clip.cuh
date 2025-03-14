#ifndef __CUDA_CLIP_H__
#define __CUDA_CLIP_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <cuda_fp16.h>
#include <numeric>

struct ClipCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    uint64_t ndim;
    uint64_t data_size;
    uint64_t max_grid_size;
    void *max_ptr;
    void *min_ptr;
};

typedef struct ClipCudaDescriptor *ClipCudaDescriptor_t;

infiniopStatus_t cudaCreateClipDescriptor(CudaHandle_t,
                                         ClipCudaDescriptor_t *,
                                         infiniopTensorDescriptor_t dst,
                                         infiniopTensorDescriptor_t src,
                                         float max,
                                         float min);

infiniopStatus_t cudaClip(ClipCudaDescriptor_t desc,
                         void *dst, void const *src,
                         void *stream);

infiniopStatus_t cudaDestroyClipDescriptor(ClipCudaDescriptor_t desc);

#endif
