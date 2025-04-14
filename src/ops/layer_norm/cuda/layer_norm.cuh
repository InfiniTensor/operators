#ifndef __NV_GPU_LAYER_NORM_H__
#define __NV_GPU_LAYER_NORM_H__

#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"

struct LayerNormCudaDescriptor {
    Device device;
    int device_id;
    DT dtype;
    int size;
    int behindsize;
    float epsilon;
};

typedef struct LayerNormCudaDescriptor *LayerNormCudaDescriptor_t;

infiniopStatus_t cudaCreateLayerNormDescriptor(CudaHandle_t handle,
                                            LayerNormCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t x_desc,
                                            infiniopTensorDescriptor_t w_desc,
                                            infiniopTensorDescriptor_t b_desc,
                                            infiniopTensorDescriptor_t y_desc,
                                             float epsilon);
                                             
infiniopStatus_t cudaGetLayerNormWorkspaceSize(LayerNormCudaDescriptor_t desc, unsigned long int *size);

infiniopStatus_t cudaLayerNorm(LayerNormCudaDescriptor_t desc, void *workspace,
                                          uint64_t workspace_size,
                            void const *x, void const *w, void const *b, void *y,
                             void *stream);

infiniopStatus_t cudaDestroyLayerNormDescriptor(LayerNormCudaDescriptor_t desc);

#endif// __NV_GPU_LAYER_NORM_H__
