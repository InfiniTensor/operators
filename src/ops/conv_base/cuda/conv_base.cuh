#ifndef __CUDA_CONV_H__
#define __CUDA_CONV_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <cudnn.h>

struct ConvBaseCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    std::shared_ptr<Pool<cudnnHandle_t>> cudnn_handles_t;
    cudnnTensorDescriptor_t const x_desc;
    cudnnFilterDescriptor_t const w_desc;
    cudnnTensorDescriptor_t const y_desc;
    cudnnConvolutionDescriptor_t const op_desc;
    cudnnConvolutionFwdAlgo_t algo;
    const float alpha;
    const float beta;
    uint64_t workspace_size;
};

typedef struct ConvBaseCudaDescriptor *ConvBaseCudaDescriptor_t;

infiniopStatus_t cudaCreateConvBaseDescriptor(CudaHandle_t,
                                              ConvBaseCudaDescriptor_t *,
                                              infiniopTensorDescriptor_t y,
                                              infiniopTensorDescriptor_t x,
                                              infiniopTensorDescriptor_t w,
                                              uint64_t const *pads,
                                              int64_t const *strides,
                                              uint64_t const *dilations,
                                              uint64_t n);

infiniopStatus_t cudaGetConvBaseWorkspaceSize(ConvBaseCudaDescriptor_t desc, uint64_t *size);

infiniopStatus_t cudaConvBase(ConvBaseCudaDescriptor_t desc,
                              void *workspace, uint64_t workspace_size,
                              void *y, void const *x, void const *w,
                              void *stream);

infiniopStatus_t cudaDestroyConvBaseDescriptor(ConvBaseCudaDescriptor_t desc);

#endif
