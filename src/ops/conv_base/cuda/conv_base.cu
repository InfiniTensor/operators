#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "conv_base.cuh"

infiniopStatus_t conv_nv_gpu(ConvBaseCudaDescriptor_t desc, void *workspace, uint64_t workspace_size,
                             void *y, void const *x, void const *w, void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    checkCudnnError(use_cudnn(desc->cudnn_handles_t, desc->device_id, (cudaStream_t) stream,
                              [&](cudnnHandle_t handle) { return cudnnConvolutionForward(handle, &desc->alpha,
                                                                                         desc->x_desc, x, desc->w_desc, w, desc->op_desc, desc->algo, workspace, workspace_size,
                                                                                         &desc->beta, desc->y_desc, y); }));
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaConvBase(ConvBaseCudaDescriptor_t desc,
                              void *workspace, uint64_t workspace_size,
                              void *y, void const *x, void const *w,
                              void *stream) {
    if (desc->dtype == F16 || desc->dtype == F32) {
        return conv_nv_gpu(desc, workspace, workspace_size, y, x, w, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
