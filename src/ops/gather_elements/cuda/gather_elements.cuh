#ifndef __CUDA_GATHER_H__
#define __CUDA_GATHER_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <cuda_fp16.h>
#include <numeric>

struct GatherElementsCudaDescriptor {
    Device device;
    DT dtype;
    DT indice_type;
    int device_id;
    uint64_t ndim;
    uint64_t indices_size;
    int64_t* stride_axis_ptr;
    int64_t* shape_axis_ptr;
    int64_t* num_indices_ptr;
    uint64_t max_grid_size;
};

typedef struct GatherElementsCudaDescriptor *GatherElementsCudaDescriptor_t;

infiniopStatus_t cudaCreateGatherElementsDescriptor(CudaHandle_t,
                                         GatherElementsCudaDescriptor_t *,
                                         infiniopTensorDescriptor_t dst,
                                         infiniopTensorDescriptor_t data,
                                         infiniopTensorDescriptor_t indices,
                                         int axis);

infiniopStatus_t cudaGatherElements(GatherElementsCudaDescriptor_t desc,
                            void *dst,     
                            void const *data,
                            void const *indices,
                            void *stream);

infiniopStatus_t cudaDestroyGatherElementsDescriptor(GatherElementsCudaDescriptor_t desc);

#endif
