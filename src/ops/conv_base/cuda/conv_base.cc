#include "conv_base.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateConvBaseDescriptor(CudaHandle_t handle,
                                              ConvBaseCudaDescriptor_t *desc_ptr,
                                              infiniopTensorDescriptor_t y,
                                              infiniopTensorDescriptor_t x,
                                              infiniopTensorDescriptor_t w,
                                              uint64_t const *pads,
                                              int64_t const *strides,
                                              uint64_t const *dilations,
                                              uint64_t n) {
    uint64_t ndim = y->ndim;
    if (ndim < 3 || ndim != x->ndim || ndim != w->ndim) {
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

    const uint64_t new_ndim = std::max(ndim, (uint64_t) 4);
    // convert pads, strides, dilations into int32[]
    int32_t *pad = new int32_t[new_ndim];
    int32_t *stride = new int32_t[new_ndim];
    int32_t *dilation = new int32_t[new_ndim];
    int32_t *x_shape = new int32_t[new_ndim];
    int32_t *w_shape = new int32_t[new_ndim];
    int32_t *y_shape = new int32_t[new_ndim];
    for (size_t i = 0; i < new_ndim; ++i) {
        pad[i] = i < ndim - 2 ? static_cast<int32_t>(pads[i]) : 0;
        stride[i] = i < ndim - 2 ? static_cast<int32_t>(strides[i]) : 1;
        dilation[i] = i < ndim - 2 ? static_cast<int32_t>(dilations[i]) : 1;
        x_shape[i] = i < ndim ? static_cast<int32_t>(x->shape[i]) : 1;
        w_shape[i] = i < ndim ? static_cast<int32_t>(w->shape[i]) : 1;
        y_shape[i] = i < ndim ? static_cast<int32_t>(y->shape[i]) : 1;
    }

    // get the data types of the tensors and the conv operator
    CREATE_CHECK_ERROR(auto tensor_dt = dataTypeMap[x->dt], tensor_dt, -1, STATUS_BAD_PARAM);
    cudnnDataType_t conv_op_dt = [&] {
        switch (tensor_dt) {
            case CUDNN_DATA_HALF:
                if (ndim >= 5) {
                    return CUDNN_DATA_FLOAT;
                }
                if (handle->compute_capability_major > 5 || (handle->compute_capability_major == 5 && handle->compute_capability_minor >= 3)) {
                    return CUDNN_DATA_HALF;
                }
                return CUDNN_DATA_FLOAT;
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

    // create and set tensor descriptors for y
    cudnnTensorDescriptor_t y_desc;
    std::vector<int> outDim_(new_ndim);
    auto outDim = outDim_.data();
    checkCudnnError(cudnnGetConvolutionNdForwardOutputDim(op_desc, x_desc, w_desc, new_ndim, outDim));
    checkCudnnError(cudnnCreateTensorDescriptor(&y_desc));
    checkCudnnError(cudnnSetTensorNdDescriptorEx(y_desc, CUDNN_TENSOR_NCHW, static_cast<cudnnDataType_t>(tensor_dt), new_ndim, y_shape));

    cudnnSetConvolutionMathType(op_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);

    // tuning: get the best algorithm
    int requestedAlgoCount = 1;
    checkCudnnError(use_cudnn(handle->cudnn_handles_t, handle->device_id, nullptr,
                              [&](cudnnHandle_t handle) { return cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &requestedAlgoCount); }));
    int algoCounts = 0;
    int chosenAlgoIndex = 0;
    bool chosen = false;
    size_t workspace_size = 0;
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results_(requestedAlgoCount);
    auto perf_results = perf_results_.data();
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

    const float alpha = 1.0f;
    const float beta = 0.0f;

    *desc_ptr = new ConvBaseCudaDescriptor{
        DevNvGpu,
        y->dt,
        handle->device_id,
        handle->cudnn_handles_t,
        x_desc,
        w_desc,
        y_desc,
        op_desc,
        perf_results[chosenAlgoIndex].algo,
        alpha,
        beta,
        workspace_size};

    delete[] pad;
    delete[] stride;
    delete[] dilation;
    delete[] x_shape;
    delete[] w_shape;
    delete[] y_shape;

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaGetConvBaseWorkspaceSize(ConvBaseCudaDescriptor_t desc, uint64_t *size) {
    *size = desc->workspace_size;
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyConvBaseDescriptor(ConvBaseCudaDescriptor_t desc) {
    checkCudnnError(cudnnDestroyConvolutionDescriptor(desc->op_desc));
    checkCudnnError(cudnnDestroyTensorDescriptor(desc->y_desc));
    checkCudnnError(cudnnDestroyFilterDescriptor(desc->w_desc));
    checkCudnnError(cudnnDestroyTensorDescriptor(desc->x_desc));
    desc->cudnn_handles_t = nullptr;
    delete desc;
    return STATUS_SUCCESS;
}
