#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "conv_act.cuh"

infiniopStatus_t conv_bias_act_nv_gpu(ConvActCudaDescriptor_t desc, void *workspace, uint64_t workspace_size,
                                      void *y, void const *x, void const *w, void const *b, void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    void const *b_ = b;
    if (!b || desc->bias_size != 0) {
        b_ = reinterpret_cast<void *>(reinterpret_cast<char *>(workspace) + desc->workspace_size - desc->bias_size);
        checkCudaErrorWithCode(cudaMemset((void *) b_, 0, desc->bias_size), STATUS_EXECUTION_FAILED);
    }
    void *workspace_ = (desc->bias_size == 0 || desc->workspace_size > desc->bias_size) ? workspace : nullptr;
    checkCudnnError(use_cudnn(desc->cudnn_handles_t, desc->device_id, (cudaStream_t) stream,
                              [&](cudnnHandle_t handle) { return cudnnConvolutionBiasActivationForward(handle, &desc->alpha,
                                                                                                       desc->x_desc, x, desc->w_desc, w, desc->op_desc, desc->algo, workspace_, workspace_size - desc->bias_size,
                                                                                                       &desc->beta, desc->y_desc, y, desc->b_desc, b_, desc->act_desc, desc->y_desc, y); }));
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaConvAct(ConvActCudaDescriptor_t desc,
                             void *workspace, uint64_t workspace_size,
                             void *y, void const *x, void const *w,
                             void const *b, void *stream) {
    if (workspace_size < desc->workspace_size) {
        return STATUS_INSUFFICIENT_WORKSPACE;
    }
    if (desc->dtype == F16 || desc->dtype == F32) {
        return conv_bias_act_nv_gpu(desc, workspace, workspace_size, y, x, w, b, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
