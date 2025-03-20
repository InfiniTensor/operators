#include "reduce_cuda.h"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
// need reduce_size, output_size, output_stride, input_stride

infiniopStatus_t cudaCreateReduceDescriptor(CudaHandle_t handle,
                                            ReduceCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t y,
                                            infiniopTensorDescriptor_t x,
                                            int64_t const *axes,
                                            uint64_t axes_size,
                                            int reduce_op_type,
                                            bool keepdims
                                            ) {
    if (x->dt != F16 && x->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (keepdims) {
        if (x->ndim != y->ndim) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (x->dt != y->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (!is_contiguous(x) || !is_contiguous(y)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    uint64_t element_num = 1;
    uint64_t output_size = 1;
    for (uint64_t i = 0; i < x->ndim; i++) {
        element_num *= x->shape[i];
    }

    for (uint64_t i = 0; i < y->ndim; i++) {
        output_size *= y->shape[i];
    }
    uint64_t reduce_size = element_num / output_size;

    int64_t *input_strides = new int64_t[x->ndim];
    int64_t *output_strides = new int64_t[y->ndim];
    uint64_t *input_shape = new uint64_t[x->ndim];
    uint64_t *output_shape = new uint64_t[y->ndim];
    int64_t *h_axes = new int64_t[axes_size];
    uint64_t* reduce_axes_stride = new uint64_t[axes_size];
    std::fill_n(reduce_axes_stride, axes_size, 1);
    for (int i = axes_size - 2; i >= 0; i--){
        reduce_axes_stride[i] = reduce_axes_stride[i + 1] * x->shape[axes[i + 1]];
    }
    memcpy(h_axes, axes, axes_size * sizeof(int64_t));
    memcpy(input_shape, x->shape, x->ndim * sizeof(uint64_t));
    memcpy(output_shape, y->shape, y->ndim * sizeof(uint64_t));
    memcpy(input_strides, x->strides, x->ndim * sizeof(int64_t));
    memcpy(output_strides, y->strides, y->ndim * sizeof(int64_t));
    int prefix_size = 1, suffix_size = 1;
    bool if_reduce_axes_contiguous = true;
    int reduce_mode = 0;
    std::vector<int64_t> non_reduce_axes;
    for (int j = 0; j < x->ndim; ++j) {
        if (!std::binary_search(axes, axes + axes_size, j)) {
            non_reduce_axes.push_back(j);
        }
    }
    int64_t *h_non_reduce_axes = new int64_t[non_reduce_axes.size()];
    std::copy(non_reduce_axes.begin(), non_reduce_axes.end(), h_non_reduce_axes);

    for (uint64_t i = 0; i < axes_size; i++) {
        if (i < axes_size - 1 && axes[i] != axes[i + 1] - 1) {
            if_reduce_axes_contiguous = false;
        }
    }
    if (if_reduce_axes_contiguous) {
        if (axes_size == x->ndim) {
            reduce_mode = 0;
        } else {
            for (uint64_t i = 0; i < axes[0]; i++) {  
                prefix_size *= x->shape[i];
            }
            for (uint64_t i = axes[axes_size - 1] + 1; i < x->ndim; i++) {  
                suffix_size *= x->shape[i];
            }
            reduce_mode = 1;
        }
    } else {
        reduce_mode = 2;
    }
    int64_t *d_non_reduce_axes;
    int64_t *d_input_strides;
    int64_t *d_output_strides;
    int64_t *d_reduce_axes;
    uint64_t *d_reduce_axes_stride;
    uint64_t *d_input_shape;
    uint64_t *d_output_shape;

    checkCudaErrorWithCode(cudaMalloc((void**)&d_reduce_axes_stride, axes_size * sizeof(uint64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc((void**)&d_reduce_axes, axes_size * sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc((void**)&d_non_reduce_axes, non_reduce_axes.size() * sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc((void**)&d_input_strides, x->ndim * sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc((void**)&d_output_strides, y->ndim * sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc((void**)&d_input_shape, x->ndim * sizeof(uint64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc((void**)&d_output_shape, y->ndim * sizeof(uint64_t)), STATUS_MEMORY_NOT_ALLOCATED);

    checkCudaErrorWithCode(cudaMemcpy(d_reduce_axes_stride, reduce_axes_stride, axes_size * sizeof(uint64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaMemcpy(d_reduce_axes, h_axes, axes_size * sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaMemcpy(d_non_reduce_axes, h_non_reduce_axes, non_reduce_axes.size() * sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaMemcpy(d_input_strides, input_strides, x->ndim * sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaMemcpy(d_output_strides, output_strides, y->ndim * sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaMemcpy(d_input_shape, input_shape, x->ndim * sizeof(uint64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaMemcpy(d_output_shape, output_shape, y->ndim * sizeof(uint64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);

    *desc_ptr = new ReduceCudaDescriptor{
        DevNvGpu,
        x->dt,
        x->ndim,
        y->ndim,
        d_non_reduce_axes,
        d_input_strides,
        d_output_strides,
        d_input_shape,
        d_output_shape,
        d_reduce_axes,
        d_reduce_axes_stride,
        reduce_size,
        element_num,
        output_size,
        static_cast<int>(reduce_op_type),
        reduce_mode,
        axes_size,
        keepdims,
        axes[0],
        axes[axes_size - 1],
        prefix_size,
        suffix_size
    };
    delete [] h_axes;
    delete [] reduce_axes_stride;
    delete [] h_non_reduce_axes;
    delete [] input_strides;
    delete [] output_strides;
    delete [] input_shape;
    delete [] output_shape;
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyReduceDescriptor(ReduceCudaDescriptor_t desc) {
   
    checkCudaErrorWithCode(cudaFree((void*)desc->reduce_axes), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree((void*)desc->reduce_axes_stride), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree((void*)desc->non_reduce_axes), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree((void*)desc->input_strides), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree((void*)desc->output_strides), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree((void*)desc->input_shape), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaFree((void*)desc->output_shape), STATUS_EXECUTION_FAILED);
    delete desc;
    return STATUS_SUCCESS;
}
