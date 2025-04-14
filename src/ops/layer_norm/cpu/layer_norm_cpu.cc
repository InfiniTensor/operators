#include "layer_norm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <cmath>

infiniopStatus_t cpuCreateLayerNormDescriptor(infiniopHandle_t handle, LayerNormCpuDescriptor_t *desc_ptr,                                           
                                            infiniopTensorDescriptor_t x_desc,
                                            infiniopTensorDescriptor_t w_desc,
                                            infiniopTensorDescriptor_t b_desc,
                                            infiniopTensorDescriptor_t y_desc,
                                            float epsilon) {
    if (w_desc->ndim != b_desc->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    int wDim = w_desc->ndim;
    for(int i = 0; i < wDim; i++){
        if(w_desc->shape[i] != b_desc->shape[i]){
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    int ndim = x_desc->ndim;
    for(int i = 0; i < wDim; i++){
        if(x_desc->shape[i + ndim - wDim] != w_desc->shape[i]){
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (!dtype_eq(x_desc->dt, F16) && !dtype_eq(x_desc->dt, F32)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    int size = 1;
    int behindsize = 1;
    for(int i = 0; i < ndim; i++){
        size *= static_cast<int>(x_desc->shape[i]);
        if(i >= ndim - wDim){
            behindsize *= static_cast<int>(x_desc->shape[i]);
        } 
    }

    *desc_ptr = new LayerNormCpuDescriptor{
        handle->device,
        x_desc->dt,
        size,
        behindsize,
        epsilon};

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyLayerNormDescriptor(LayerNormCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

void layer_norm_cpu(LayerNormCpuDescriptor_t desc, void const *x, void const *w, void const *b, void *y) {
    int size = desc->size;
    int behindsize = desc->behindsize;
    int frontsize = size / behindsize;
    float eps = desc->epsilon;
    if (dtype_eq(desc->dtype, F32))
    {
        auto source = reinterpret_cast<const float *>(x);
        auto weight = reinterpret_cast<const float *>(w);
        auto _bias = reinterpret_cast<const float *>(b);
        auto destination = reinterpret_cast<float *>(y);
        for (int i = 0; i < frontsize; i++)
        {
            int tid = i * behindsize;
            float mu = 0.0f;
            for (int id = 0; id < behindsize; id++)
            {
                mu += source[tid + id];
            }
            mu /= behindsize;
            float sigma2Partial = 0.0f;
            for (int id = 0; id < behindsize; id++)
            {
                sigma2Partial += (source[tid + id] - mu) * (source[tid + id] - mu);
            }
            float sigma2 = 1.0f / sqrt(sigma2Partial / behindsize + eps);
            for (int id = 0; id < behindsize; id++)
            {
                destination[tid + id] = (source[tid + id] - mu) * weight[id] * sigma2 + _bias[id];
            }
        }
    }
    else if (dtype_eq(desc->dtype, F16))
    {
        auto source = reinterpret_cast<const uint16_t *>(x);
        auto weight = reinterpret_cast<const uint16_t *>(w);
        auto _bias = reinterpret_cast<const uint16_t *>(b);
        auto destination = reinterpret_cast<uint16_t *>(y);
        for (int i = 0; i < frontsize; i++)
        {
            int tid = i * behindsize;
            float mu = 0.0f;
            for (int id = 0; id < behindsize; id++)
            {
                mu += f16_to_f32(source[tid + id]);
            }
            mu /= behindsize;
            float sigma2Partial = 0.0f;
            for (int id = 0; id < behindsize; id++)
            {
                sigma2Partial += (f16_to_f32(source[tid + id]) - mu) * (f16_to_f32(source[tid + id]) - mu);
            }
            float sigma2 = 1.0f / sqrt(sigma2Partial / behindsize + eps);
            for (int id = 0; id < behindsize; id++)
            {
                float tmp = (f16_to_f32(source[tid + id]) - mu) * f16_to_f32(weight[id]) * sigma2 + f16_to_f32(_bias[id]);
                destination[tid + id] = f32_to_f16(tmp);
            }
        }
    }
}
infiniopStatus_t cpuGetLayerNormWorkspaceSize(LayerNormCpuDescriptor_t desc, unsigned long int *size) {
    *size = 0;
    return STATUS_SUCCESS;
}
infiniopStatus_t cpuLayerNorm(LayerNormCpuDescriptor_t desc, void *workspace,
                                          uint64_t workspace_size,
                            void const *x, void const *w, void const *b, void *y,
                            void *stream) {
    if (dtype_eq(desc->dtype, F16) || dtype_eq(desc->dtype, F32)) {
        layer_norm_cpu(desc, x, w, b, y);
        return STATUS_SUCCESS;
    }

    return STATUS_BAD_TENSOR_DTYPE;
}
