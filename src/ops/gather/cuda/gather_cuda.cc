#include "gather_cuda.h"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateGatherDescriptor(CudaHandle_t handle,
                                            GatherCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t y,
                                            infiniopTensorDescriptor_t x,
                                            infiniopTensorDescriptor_t indices,
                                            int64_t axis
                                            ){
    if (y->dt != x->dt){
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != F16 && y->dt != F32){
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (!is_contiguous(y) || !is_contiguous(x)){
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (axis < 0 || axis >= x->ndim){
        return STATUS_BAD_PARAM;
    }  
    int otherDims = 1;
    for (int i = 0; i < axis; i++){
        otherDims *= static_cast<int>(x->shape[i]);
    }
    for (int i = axis + 1; i < x->ndim; i++){
        otherDims *= static_cast<int>(x->shape[i]);
    }
    int stride = 1;
    for (int i = axis + 1; i < x->ndim; i++){
        stride *= static_cast<int>(x->shape[i]);
    }
    int indices_size = 1;
    for (int i = 0; i < indices->ndim; i++){
        indices_size *= static_cast<int>(indices->shape[i]);
    }
    int dim_size = static_cast<int>(x->shape[axis]);
    *desc_ptr = new GatherCudaDescriptor{
        DevNvGpu,
        x->dt,
        indices->dt,
        axis,
        otherDims,
        x->ndim,
        y->ndim,
        dim_size,
        indices_size,
        stride
    };
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyGatherDescriptor(GatherCudaDescriptor_t desc){
    return STATUS_SUCCESS;
}