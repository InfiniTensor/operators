#include "gather.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateGatherDescriptor(CudaHandle_t handle,
                                            GatherCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t output,
                                            infiniopTensorDescriptor_t input,
                                            infiniopTensorDescriptor_t indices,
                                            uint64_t axis) {
    if (indices->dt != I32 && indices->dt != I64) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (output->dt != input->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    if (output->ndim != input->ndim + indices->ndim - 1) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    for (int i = 0; i < output->ndim; i++) {
        if (i < axis) {
            if (output->shape[i] != input->shape[i]) {
                return STATUS_BAD_TENSOR_SHAPE;
            }
        } else if (i < axis + indices->ndim) {
            if (output->shape[i] != indices->shape[i - axis]) {
                return STATUS_BAD_TENSOR_SHAPE;
            }
        } else {
            if (output->shape[i] != input->shape[i - indices->ndim + 1]) {
                return STATUS_BAD_TENSOR_SHAPE;
            }
        }
    }

    if (!is_contiguous(output) || !is_contiguous(input) || !is_contiguous(indices)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }

    uint64_t axis_tmp = axis;
    if (axis_tmp < 0) {
        axis_tmp = output->ndim + axis;
    }

    uint64_t output_size = std::accumulate(output->shape, output->shape + output->ndim, 1ULL, std::multiplies<uint64_t>());
    uint64_t pre_size = std::accumulate(input->shape, input->shape + axis_tmp, 1ULL, std::multiplies<uint64_t>());
    uint64_t post_size = std::accumulate(input->shape + axis_tmp + 1, input->shape + input->ndim, 1ULL, std::multiplies<uint64_t>());
    uint64_t indices_size = std::accumulate(indices->shape, indices->shape + indices->ndim, 1ULL, std::multiplies<uint64_t>());
    uint64_t axis_size = input->shape[axis_tmp];

    int kernel_type = 0;
    uint64_t sizes[3] = {pre_size * indices_size, indices_size * post_size, pre_size * post_size};
    for(int i = 1; i < 3; i++) {
        if(sizes[i] > sizes[kernel_type]) {
            kernel_type = i;
        }
    }


    *desc_ptr = new GatherCudaDescriptor{
        DevNvGpu,
        output->dt,
        indices->dt,
        handle->device_id,
        output_size,
        pre_size,
        axis_size,
        indices_size,
        post_size,
        kernel_type,
        static_cast<uint64_t>(handle->prop.maxGridSize[0]),
    };


    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyGatherDescriptor(GatherCudaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
