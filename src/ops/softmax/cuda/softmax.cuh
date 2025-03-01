#ifndef __CUDA_SOFTMAX_H__
#define __CUDA_SOFTMAX_H__

#include "../../../devices/cuda/cuda_handle.h"
#include "../../utils.h"
#include "operators.h"

struct SoftmaxCudaDescriptor {
    Device device;
    int device_id;
    DT dtype;
    int dimsize;
    int stride;
    int othersize;
};

typedef struct SoftmaxCudaDescriptor *SoftmaxCudaDescriptor_t;

infiniopStatus_t cudaCreateSoftmaxDescriptor(CudaHandle_t handle,
                                             SoftmaxCudaDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t input_desc, int axis, infiniopTensorDescriptor_t output_desc);

infiniopStatus_t cudaGetSoftmaxWorkspaceSize(SoftmaxCudaDescriptor_t desc, unsigned long int *size);
infiniopStatus_t cudaSoftmax(SoftmaxCudaDescriptor_t desc, void *workspace,
                                          uint64_t workspace_size,
                             void const *input,
                             void *output,
                             void *stream);

infiniopStatus_t cudaDestroySoftmaxDescriptor(SoftmaxCudaDescriptor_t desc);


#endif
