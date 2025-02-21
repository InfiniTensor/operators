#ifndef __CUDA_CONV_ACT_H__
#define __CUDA_CONV_ACT_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include "ops/conv_act/conv_act.h"
#include <cstddef>
#include <cudnn.h>

struct ConvActCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    std::shared_ptr<Pool<cudnnHandle_t>> cudnn_handles_t;
    cudnnTensorDescriptor_t const x_desc;
    cudnnFilterDescriptor_t const w_desc;
    cudnnTensorDescriptor_t const b_desc;
    cudnnTensorDescriptor_t const y_desc;
    cudnnConvolutionDescriptor_t const op_desc;
    cudnnActivationDescriptor_t const act_desc;
    cudnnConvolutionFwdAlgo_t algo;
    const float alpha;
    const float beta;
    uint64_t workspace_size;
    uint64_t bias_size;
};

typedef struct ConvActCudaDescriptor *ConvActCudaDescriptor_t;

infiniopStatus_t cudaCreateConvActDescriptor(CudaHandle_t,
                                             ConvActCudaDescriptor_t *,
                                             infiniopTensorDescriptor_t y,
                                             infiniopTensorDescriptor_t x,
                                             infiniopTensorDescriptor_t w,
                                             infiniopTensorDescriptor_t b,
                                             uint64_t const *pads,
                                             int64_t const *strides,
                                             uint64_t const *dilations,
                                             uint64_t n,
                                             InfiniActivationMode_t activation_mode,
                                             ConvActParam_t act_params);

infiniopStatus_t cudaGetConvActWorkspaceSize(ConvActCudaDescriptor_t desc, uint64_t *size);

infiniopStatus_t cudaConvAct(ConvActCudaDescriptor_t desc,
                             void *workspace, uint64_t workspace_size,
                             void *y, void const *x, void const *w,
                             void const *b, void *stream);

infiniopStatus_t cudaDestroyConvActDescriptor(ConvActCudaDescriptor_t desc);

#endif
