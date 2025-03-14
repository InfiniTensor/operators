#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "gather_elements.cuh"

template<typename Tdata, typename Idata>
__global__ void gather(
    Tdata *dst,
    const Tdata *data,
    const Idata *indices,
    const int64_t *stride_axis,
    const int64_t *shape_axis,
    const int64_t *num_indices,
    uint64_t offset) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    Idata index = indices[idx];
    int64_t linearId = *shape_axis * *stride_axis * (idx / (*num_indices * *stride_axis)) + index * *stride_axis + idx % *stride_axis;
    dst[idx] = data[linearId];

}

template<typename Tdata, typename Idata>
infiniopStatus_t gather_nv_gpu(GatherElementsCudaDescriptor_t desc, void *dst, void const *data, void const *indices, void *stream) {
    if (desc->indices_size == 0) {
        return STATUS_SUCCESS;
    }
    dim3 blockDims = dim3(std::min(static_cast<uint64_t>(256), desc->indices_size));
    dim3 gridDims = dim3(std::min(ROUND_UP_DIV(desc->indices_size, blockDims.x), desc->max_grid_size));
    uint64_t step = gridDims.x * blockDims.x;

    const auto dst_ = reinterpret_cast<Tdata *>(dst);
    const auto data_ = reinterpret_cast<Tdata const *>(data);
    const auto indices_ = reinterpret_cast<Idata const *>(indices);

    const int64_t *stride_axis = reinterpret_cast<const int64_t *>(desc->stride_axis_ptr);
    const int64_t *shape_axis = reinterpret_cast<const int64_t *>(desc->shape_axis_ptr);
    const int64_t *num_indices = reinterpret_cast<const int64_t *>(desc->num_indices_ptr);
    
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

#pragma unroll
    for (uint64_t i = 0; i < desc->indices_size; i += step) {
        gather<Tdata, Idata><<<gridDims, blockDims, 0, cuda_stream>>>(
            dst_, data_, indices_, stride_axis, shape_axis, num_indices, i);
    }

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaGatherElements(GatherElementsCudaDescriptor_t desc,
    void *dst,     
    void const *data,
    void const *indices,
    void *stream){
    checkCudaError(cudaSetDevice(desc->device_id));
    if(desc->dtype == F32 && desc->indice_type == I32)
    {
        return gather_nv_gpu<float, int32_t>(desc, dst, data, indices, stream);
    }
    else if(desc->dtype == F16 && desc->indice_type == I32)
    {
        return gather_nv_gpu<half, int32_t>(desc, dst, data, indices, stream);
    }
    else if(desc->dtype == F32 && desc->indice_type == I64)
    {
        return gather_nv_gpu<float, int64_t>(desc, dst, data, indices, stream);
    }
    else if(desc->dtype == F16 && desc->indice_type == I64)
    {
        return gather_nv_gpu<half, int64_t>(desc, dst, data, indices, stream);
    }

    return STATUS_BAD_TENSOR_DTYPE;
}