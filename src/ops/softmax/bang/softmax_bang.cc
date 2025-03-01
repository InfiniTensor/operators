#include "softmax_bang.h"
#include "../../utils.h"

infiniopStatus_t bangCreateSoftmaxDescriptor(BangHandle_t handle,
                                             SoftmaxBangDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t input_desc, int axis, infiniopTensorDescriptor_t output_desc) {

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

    int stride = 1;
    int dimsize = static_cast<int>(input_desc->shape[axis]);
    int othersize = 1;
    int frontsize = 1;

    for (int s = ndim - 1; s >= 0; s--) {
        if (s > axis) {
            stride *= static_cast<int>(input_desc->shape[s]);
        }
        if (s < axis) {
            frontsize *= static_cast<int>(input_desc->shape[s]);
        }
        if (s != axis) {
            othersize *= static_cast<int>(input_desc->shape[s]);
        }
    }
    *desc_ptr = new SoftmaxBangDescriptor{
        handle->device,
        handle->device_id,
        input_desc->dt,
        ndim,
        axis,
        dimsize,
        stride,
        othersize,
        frontsize};

    return STATUS_SUCCESS;
}
infiniopStatus_t bangGetSoftmaxWorkspaceSize(SoftmaxBangDescriptor_t desc, unsigned long int *size) {
    *size = 32 * desc->othersize * sizeof(desc->dtype);//taskDim * othersize * sizeof(T),taskDim不超过32
    return STATUS_SUCCESS;
}

infiniopStatus_t bangDestroySoftmaxDescriptor(SoftmaxBangDescriptor_t desc) {

    delete desc;
    return STATUS_SUCCESS;
}
