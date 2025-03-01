#include "softmax_cnnl.h"
#include "../../../devices/bang/bang_handle.h"
#include "../../../devices/bang/common_bang.h"
#include "../../utils.h"
#include "cnrt.h"
infiniopStatus_t cnnlCreateSoftmaxDescriptor(BangHandle_t handle,
                                             SoftmaxCnnlDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t input_desc,
                                             int axis,
                                             infiniopTensorDescriptor_t output_desc) {
    if (input_desc->ndim != output_desc->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!dtype_eq(input_desc->dt, F16) && !dtype_eq(input_desc->dt, F32)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    int ndim = input_desc->ndim;

    for (int i = 0; i < ndim; i++) {
        if (input_desc->shape[i] != output_desc->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }

    int *shape = new int[ndim];
    for (int i = 0; i < ndim; i++) {
        shape[i] = static_cast<int>(input_desc->shape[i]);
    }
    cnnlSoftmaxMode_t mode;
    std::vector<int> inDim = {1, 1, 1};
    std::vector<int> outDim = inDim;

    if (ndim >= 3) {
        if (axis == 0) {
            mode = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
            inDim[0] = shape[0];
            inDim[1] = shape[1];
            for (int i = 2; i < ndim; ++i) {
                inDim[2] *= shape[i];
            }
            outDim = inDim;
        } else if (axis == ndim - 1) {
            mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
            inDim[0] = shape[0];
            for (int i = 1; i < axis; ++i) {
                inDim[1] *= shape[i];
            }
            inDim[2] = shape[axis];
            outDim = inDim;
        } else {
            mode = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
            for (int i = 0; i < axis; ++i) {
                inDim[0] *= shape[i];
            }
            inDim[1] = shape[axis];
            for (int i = axis + 1; i < ndim; ++i) {
                inDim[2] *= shape[i];
            }
            outDim = inDim;
        }
    } else if (ndim == 2) {
        if (axis == 0) {
            mode = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
            inDim[0] = shape[0];
            inDim[1] = shape[1];

            outDim = inDim;
        } else {
            mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
            inDim[1] = shape[0];
            inDim[2] = shape[1];

            outDim = inDim;
        }
    } else {
        mode = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
        inDim[0] = shape[0];

        outDim = inDim;
    }
    cnnlTensorDescriptor_t aDesc, cDesc;
    cnnlCreateTensorDescriptor(&aDesc);
    cnnlCreateTensorDescriptor(&cDesc);
    if (dtype_eq(input_desc->dt, F16)) {
        cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
            inDim.size(), inDim.data());
        cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
            outDim.size(), outDim.data());
    } else if (dtype_eq(input_desc->dt, F32)) {
        cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
            inDim.size(), inDim.data());
        cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
            outDim.size(), outDim.data());
    }

    float alpha = 1.0;
    float beta = 0.0;
    *desc_ptr = new SoftmaxCnnlDescriptor{
        handle->device,
        handle->device_id,
        input_desc->dt,
        handle->cnnl_handles,
        mode,
        aDesc,
        cDesc,
        alpha,
        beta};
    return STATUS_SUCCESS;
}
infiniopStatus_t cnnlGetSoftmaxWorkspaceSize(SoftmaxCnnlDescriptor_t desc, unsigned long int *size) {
    *size = 0;
    return STATUS_SUCCESS;
}

infiniopStatus_t cnnlDestroySoftmaxDescriptor(SoftmaxCnnlDescriptor_t desc) {
    desc->cnnl_handles = nullptr;
    cnnlDestroyTensorDescriptor(desc->aDesc);
    cnnlDestroyTensorDescriptor(desc->cDesc);
    delete desc;
    return STATUS_SUCCESS;
}

void softmax_cnnl(SoftmaxCnnlDescriptor_t desc, void const *input, void *output, void *stream) {
    float alpha = desc->alpha;
    float beta = desc->beta;
    cnnlSoftmaxMode_t mode = desc->mode;
    cnnlTensorDescriptor_t aDesc = desc->aDesc;
    cnnlTensorDescriptor_t cDesc = desc->cDesc;

    use_cnnl(desc->cnnl_handles, desc->device_id, (cnrtQueue_t) stream,
             [&](cnnlHandle_t handle) {
                 cnnlSoftmaxForward_v2(handle, CNNL_SOFTMAX_ACCURATE,
                                       mode, CNNL_COMPUTATION_ULTRAHIGH_PRECISION,
                                       &alpha, aDesc, input, &beta, cDesc, output);
             });
}
infiniopStatus_t cnnlSoftmax(SoftmaxCnnlDescriptor_t desc, void *workspace,
                                          uint64_t workspace_size, void const *input, void *output, void *stream) {
    if (cnrtSetDevice(desc->device_id) != cnrtSuccess) {
        return STATUS_BAD_DEVICE;
    }

    if (dtype_eq(desc->dtype, F16) || dtype_eq(desc->dtype, F32)) {
        softmax_cnnl(desc, input, output, stream);

        return STATUS_SUCCESS;
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
