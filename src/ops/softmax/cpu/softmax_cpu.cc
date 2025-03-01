#include "softmax_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <cmath>

infiniopStatus_t cpuCreateSoftmaxDescriptor(infiniopHandle_t handle,
                                            SoftmaxCpuDescriptor_t *desc_ptr,
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
    int *shape = new int[ndim];

    for (int i = 0; i < ndim; i++) {
        shape[i] = static_cast<int>(input_desc->shape[i]);
    }
    *desc_ptr = new SoftmaxCpuDescriptor{
        handle->device,
        input_desc->dt,
        ndim,
        axis,
        shape};

    return STATUS_SUCCESS;
}


infiniopStatus_t cpuDestroySoftmaxDescriptor(SoftmaxCpuDescriptor_t desc) {
    delete[] desc->shape;
    delete desc;
    return STATUS_SUCCESS;
}
infiniopStatus_t cpuGetSoftmaxWorkspaceSize(SoftmaxCpuDescriptor_t desc, unsigned long int *size) {
    *size = 0;
    return STATUS_SUCCESS;
}
void softmax_cpu(SoftmaxCpuDescriptor_t desc,
                 void const *input, void *output) {
    int ndim = desc->ndim;
    int axis = desc->axis;
    auto shape = desc->shape;
    int dimsize = shape[axis];
    int othersize = 1;
    int stride = 1;

    for (int s = ndim - 1; s >= 0; s--) {

        if (s > axis) {
            stride *= shape[s];
        }
        if (s != axis) {
            othersize *= shape[s];
        }
    }
    if (dtype_eq(desc->dtype, F16)) {
        auto source = reinterpret_cast<const uint16_t *>(input);
        auto destination = reinterpret_cast<uint16_t *>(output);
        //假设[I, J, K, S], axis = 1, othersize = IKS
        for (int ind = 0; ind < othersize; ind++) {                 //ind = i(KS) + k(S) + s
            int tid = ind % stride + (ind - ind % stride) * dimsize;//now, tid = i(JKS) + k(S) + s;
            float localM = -__FLT_MAX__;
            for (int j = 0; j < dimsize; j++) {
                int index = tid + j * stride;
                localM = fmax(localM, f16_to_f32(source[index]));
            }
            float localS = 0.0f;
            for (int j = 0; j < dimsize; j++) {
                int index = tid + j * stride;
                localS += std::exp(f16_to_f32(source[index]) - localM);
            }
            for (int j = 0; j < dimsize; j++) {
                int index = tid + j * stride;
                destination[index] = f32_to_f16(std::exp(f16_to_f32(source[index]) - localM) / localS);
            }
        }
    } else if (dtype_eq(desc->dtype, F32)) {
        auto source = reinterpret_cast<const float *>(input);
        auto destination = reinterpret_cast<float *>(output);
        //假设[I, J, K, S], axis = 1, othersize = IKS
        for (int ind = 0; ind < othersize; ind++) {                 //ind = i(KS) + k(S) + s
            int tid = ind % stride + (ind - ind % stride) * dimsize;//now, tid = i(JKS) + k(S) + s;
            float localM = -__FLT_MAX__;
            for (int j = 0; j < dimsize; j++) {
                int index = tid + j * stride;
                localM = fmax(localM, source[index]);
            }
            float localS = 0.0f;
            for (int j = 0; j < dimsize; j++) {
                int index = tid + j * stride;
                localS += std::exp(source[index] - localM);
            }
            for (int j = 0; j < dimsize; j++) {
                int index = tid + j * stride;
                destination[index] = std::exp(source[index] - localM) / localS;
            }
        }
    }
}
infiniopStatus_t cpuSoftmax(SoftmaxCpuDescriptor_t desc, void *workspace,
                                          uint64_t workspace_size,
                            void const *input, void *output,
                            void *stream) {
    if (dtype_eq(desc->dtype, F16) || dtype_eq(desc->dtype, F32)) {
        softmax_cpu(desc, input, output);
        return STATUS_SUCCESS;
    }

    return STATUS_BAD_TENSOR_DTYPE;
}
