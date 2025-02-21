#include "conv_act.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateConvActDescriptor(CudaHandle_t handle,
                                             ConvActCudaDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t y,
                                             infiniopTensorDescriptor_t x,
                                             infiniopTensorDescriptor_t w,
                                             infiniopTensorDescriptor_t b,
                                             uint64_t const *pads,
                                             int64_t const *strides,
                                             uint64_t const *dilations,
                                             uint64_t n,
                                             InfiniActivationMode_t activation_mode,
                                             ConvActParam_t act_params) {
    uint64_t ndim = y->ndim;
    if (ndim < 3 || ndim != x->ndim || ndim != w->ndim || n != ndim - 2) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (x->shape[0] != y->shape[0] || w->shape[0] != y->shape[1] || x->shape[1] != w->shape[1]) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != x->dt || y->dt != w->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (b) {
        if (b->ndim != 1 || b->shape[0] != w->shape[0]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
        if (y->dt != b->dt) {
            return STATUS_BAD_TENSOR_DTYPE;
        }
    }
    // check if the activation_mode is valid
    if (activation_mode < 0 || activation_mode >= INFINI_ACTIVATION_COUNT) {
        return STATUS_BAD_PARAM;
    }
    cudnnActivationMode_t act_mode = [&] {
        switch (activation_mode) {
            case INFINI_ACTIVATION_IDENTITY:
                return CUDNN_ACTIVATION_IDENTITY;
            case INFINI_ACTIVATION_RELU:
                return CUDNN_ACTIVATION_RELU;
            default:
                return CUDNN_ACTIVATION_SIGMOID;
        }
    }();
    // cudnnConvolutionBiasActivationForward() currently only supports identity and relu activations
    if (act_mode != CUDNN_ACTIVATION_IDENTITY && act_mode != CUDNN_ACTIVATION_RELU) {
        return STATUS_BAD_PARAM;
    }

    const auto new_ndim = std::max(4UL, ndim);
    const auto new_n = std::max(2UL, n);
    // convert pads, strides, dilations into int32[]
    int32_t pad[new_n];
    int32_t stride[new_n];
    int32_t dilation[new_n];
    int32_t x_shape[new_ndim];
    int32_t w_shape[new_ndim];
    int32_t b_shape[new_ndim];
    int32_t y_shape[new_ndim];
#pragma unroll
    for (size_t i = 0; i < new_n; ++i) {
        pad[i] = i < n ? static_cast<int32_t>(pads[i]) : 0;
        stride[i] = i < n ? static_cast<int32_t>(strides[i]) : 1;
        dilation[i] = i < n ? static_cast<int32_t>(dilations[i]) : 1;
    }
#pragma unroll
    for (size_t i = 0; i < new_ndim; ++i) {
        x_shape[i] = i < ndim ? static_cast<int32_t>(x->shape[i]) : 1;
        w_shape[i] = i < ndim ? static_cast<int32_t>(w->shape[i]) : 1;
        b_shape[i] = i == 1 ? static_cast<int32_t>(w->shape[0]) : 1;
        y_shape[i] = i < ndim ? static_cast<int32_t>(y->shape[i]) : 1;
    }

    // get the data types of the tensors and the conv operator
    CREATE_CHECK_ERROR(auto tensor_dt = dataTypeMap[x->dt], tensor_dt, -1, STATUS_BAD_PARAM);
    cudnnDataType_t conv_op_dt = [&] {
        switch (tensor_dt) {
            case CUDNN_DATA_HALF:
            case CUDNN_DATA_BFLOAT16:
            case CUDNN_DATA_FLOAT:
                return CUDNN_DATA_FLOAT;
            case CUDNN_DATA_DOUBLE:
                return CUDNN_DATA_DOUBLE;
            default:
                return CUDNN_DATA_INT32;
        }
    }();

    // create and set tensor descriptors for x
    cudnnTensorDescriptor_t x_desc;
    checkCudnnError(cudnnCreateTensorDescriptor(&x_desc));
    checkCudnnError(cudnnSetTensorNdDescriptorEx(x_desc, CUDNN_TENSOR_NCHW, static_cast<cudnnDataType_t>(tensor_dt), new_ndim, x_shape));

    // create and set tensor descriptors for w
    cudnnFilterDescriptor_t w_desc;
    checkCudnnError(cudnnCreateFilterDescriptor(&w_desc));
    checkCudnnError(cudnnSetFilterNdDescriptor(w_desc, static_cast<cudnnDataType_t>(tensor_dt), CUDNN_TENSOR_NCHW, new_ndim, w_shape));

    // create and set conv operator descriptor
    cudnnConvolutionDescriptor_t op_desc;
    checkCudnnError(cudnnCreateConvolutionDescriptor(&op_desc));
    checkCudnnError(cudnnSetConvolutionNdDescriptor(
        op_desc, new_ndim - 2, pad, stride, dilation, CUDNN_CROSS_CORRELATION,
        conv_op_dt));

    cudnnSetConvolutionMathType(op_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);

    // create and set tensor descriptors for y
    cudnnTensorDescriptor_t y_desc;
    int outDim[new_ndim];
    checkCudnnError(cudnnGetConvolutionNdForwardOutputDim(op_desc, x_desc, w_desc, new_ndim, outDim));
    checkCudnnError(cudnnCreateTensorDescriptor(&y_desc));
    checkCudnnError(cudnnSetTensorNdDescriptorEx(y_desc, CUDNN_TENSOR_NCHW, static_cast<cudnnDataType_t>(tensor_dt), new_ndim, y_shape));

    // create the activation descriptor
    cudnnActivationDescriptor_t act_desc;
    checkCudnnError(cudnnCreateActivationDescriptor(&act_desc));
    checkCudnnError(cudnnSetActivationDescriptor(act_desc, act_mode, CUDNN_NOT_PROPAGATE_NAN, act_params.clip_coef));

    // create the bias descriptor
    cudnnTensorDescriptor_t b_desc;
    checkCudnnError(cudnnCreateTensorDescriptor(&b_desc));
    checkCudnnError(cudnnSetTensorNdDescriptorEx(b_desc, CUDNN_TENSOR_NCHW, static_cast<cudnnDataType_t>(tensor_dt), new_ndim, b_shape));

    // get the best algorithm and the required workspace
    cudnnConvolutionFwdAlgo_t algo;
    size_t workspace_size = 0;

    if (act_mode == CUDNN_ACTIVATION_IDENTITY) {
        algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        checkCudnnError(use_cudnn(handle->cudnn_handles_t, handle->device_id, nullptr,
                                  [&](cudnnHandle_t handle) { return cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, op_desc, y_desc, algo, &workspace_size); }));
    } else {// tuning
        int requestedAlgoCount = 1;
        checkCudnnError(use_cudnn(handle->cudnn_handles_t, handle->device_id, nullptr,
                                  [&](cudnnHandle_t handle) { return cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &requestedAlgoCount); }));
        int algoCounts = 0;
        int chosenAlgoIndex = 0;
        bool chosen = false;

        cudnnConvolutionFwdAlgoPerf_t perf_results[requestedAlgoCount];
        checkCudnnError(use_cudnn(handle->cudnn_handles_t, handle->device_id, nullptr,
                                  [&](cudnnHandle_t handle) { return cudnnFindConvolutionForwardAlgorithm(handle, x_desc, w_desc, op_desc, y_desc, requestedAlgoCount, &algoCounts, perf_results); }));
        if (algoCounts < 1) {
            return STATUS_EXECUTION_FAILED;
        }
        for (int i = 0; i < algoCounts; ++i) {
            if (use_cudnn(handle->cudnn_handles_t, handle->device_id, nullptr,
                          [&](cudnnHandle_t handle) { return cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, op_desc, y_desc, perf_results[i].algo, &workspace_size); }) == CUDNN_STATUS_SUCCESS) {
                chosenAlgoIndex = i;
                chosen = true;
                break;
            }
        }
        if (!chosen) {
            return STATUS_EXECUTION_FAILED;
        }
        algo = perf_results[chosenAlgoIndex].algo;
    }

    // if bias is not given, add the workspace size needed by the optional bias
    uint64_t bias_size = 0;
    if (!b) {
        bias_size = w->shape[0] * w->dt.size;
        workspace_size += bias_size;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    *desc_ptr = new ConvActCudaDescriptor{
        DevNvGpu,
        y->dt,
        handle->device_id,
        handle->cudnn_handles_t,
        x_desc,
        w_desc,
        b_desc,
        y_desc,
        op_desc,
        act_desc,
        algo,
        alpha,
        beta,
        workspace_size,
        bias_size,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaGetConvActWorkspaceSize(ConvActCudaDescriptor_t desc, uint64_t *size) {
    *size = desc->workspace_size;
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyConvActDescriptor(ConvActCudaDescriptor_t desc) {
    checkCudnnError(cudnnDestroyActivationDescriptor(desc->act_desc));
    checkCudnnError(cudnnDestroyConvolutionDescriptor(desc->op_desc));
    checkCudnnError(cudnnDestroyTensorDescriptor(desc->y_desc));
    checkCudnnError(cudnnDestroyTensorDescriptor(desc->b_desc));
    checkCudnnError(cudnnDestroyFilterDescriptor(desc->w_desc));
    checkCudnnError(cudnnDestroyTensorDescriptor(desc->x_desc));
    desc->cudnn_handles_t = nullptr;
    delete desc;
    return STATUS_SUCCESS;
}
