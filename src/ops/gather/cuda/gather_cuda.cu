#include "../../../devices/cuda/cuda_handle.h"
#include "../../utils.h"
#include "gather_cuda.h"
#include <cuda_fp16.h>

#define LDST128BITS_CONST(value) (reinterpret_cast<const float4*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
template<typename IndexType, bool use_shared, typename T>
__global__ void gather_kernel(
    const T* input,
    const IndexType* indices,
    T* output,
    int otherDims,
    int dim_size,
    int stride,
    int indices_size
) {
    extern __shared__ char shared_mem[];
    IndexType* sharedIndices = (IndexType*)shared_mem;
    if constexpr(use_shared) {
        for (int i = threadIdx.x; i < indices_size; i += blockDim.x) {
            if (i < indices_size) {
                sharedIndices[i] = indices[i];
            }
        }
        __syncthreads();
    }
    constexpr int ITEMS_PER_THREAD = 4;
    int tid_base = (blockIdx.x * blockDim.x + threadIdx.x) * ITEMS_PER_THREAD;
    int total_elements = otherDims * indices_size;
    int stride_indices = stride * indices_size;
    
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int tid = tid_base + i;
        if (tid >= total_elements) break;
        int outer_idx = tid / stride_indices; 
        int remaining = tid - outer_idx * stride_indices; 
        int indices_idx = remaining / stride;
        int inner_idx = remaining - indices_idx * stride;
        IndexType gather_idx;
        if constexpr(use_shared) {
            gather_idx = indices_size == 1 ? sharedIndices[0] : sharedIndices[indices_idx];
        } else {
            gather_idx = indices_size == 1 ? indices[0] : indices[indices_idx];
        }
        if (gather_idx >= 0 && gather_idx < dim_size) {
            int outer_offset = outer_idx * stride_indices;
            int indices_offset = indices_idx * stride;
            int input_offset = outer_idx * stride * dim_size + gather_idx * stride;
            output[outer_offset + indices_offset + inner_idx] = input[input_offset + inner_idx];
        }
    }
}
template<typename IndexType, bool use_shared, typename T>
__global__ void gather_vectorized_kernel(
    const T* input,
    const IndexType* indices,
    T* output,
    int otherDims,
    int dim_size,
    int stride,
    int indices_size
) {
    extern __shared__ char shared_mem[];
    IndexType* sharedIndices = (IndexType*)shared_mem;

    if constexpr(use_shared) {
        for (int i = threadIdx.x; i < indices_size; i += blockDim.x) {
            if (i < indices_size) {
                sharedIndices[i] = indices[i];
            }
        }
        __syncthreads();
    }
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_in_vec4;
    int total_vectors;
    if constexpr(std::is_same<T, float>::value){
        stride_in_vec4 = stride / 4; 
        total_vectors = otherDims * indices_size / 4;
    } else {
        stride_in_vec4 = stride / 8; 
        total_vectors = otherDims * indices_size / 8;
    }
    if (tid < total_vectors) {
        int outer_idx = tid / (stride_in_vec4 * indices_size);
        int remaining = tid - (outer_idx * stride_in_vec4 * indices_size);
        int indices_idx = remaining / stride_in_vec4;
        int inner_idx = remaining - (indices_idx * stride_in_vec4);
        IndexType gather_idx;
        if constexpr(use_shared) {
            gather_idx = indices_size == 1 ? sharedIndices[0] : sharedIndices[indices_idx];
        } else {
            gather_idx = indices_size == 1 ? indices[0] : indices[indices_idx];
        }
        if (gather_idx >= 0 && gather_idx < dim_size) {
            int input_idx = (outer_idx * stride_in_vec4 * dim_size + gather_idx * stride_in_vec4 + inner_idx) * 4;
            int output_idx = (outer_idx * stride_in_vec4 * indices_size + indices_idx * stride_in_vec4 + inner_idx) * 4;
            LDST128BITS(output[output_idx]) = LDST128BITS_CONST(input[input_idx]);
        }
    }
}

template <typename T, typename IndexType>
infiniopStatus_t gather_nv_gpu(GatherCudaDescriptor_t desc,
    void const *x,
    void *y,
    void const *indices,
    void *stream){

    const size_t sharedMemSize = desc->indices_size <= 1024 ? desc->indices_size * sizeof(IndexType) : 0;
    if (desc->stride % 4 == 0 && desc->stride >= 4){
        const int block_size = 128;
        const int total_vectors = desc->otherDims * desc->indices_size / 4;
        const int gridSize = (total_vectors + block_size - 1) / block_size;
        if (sharedMemSize == 0) gather_vectorized_kernel<IndexType, false, T><<<gridSize, block_size, 0, (cudaStream_t)stream>>>(
            reinterpret_cast<const T*>(x),
            reinterpret_cast<const IndexType*>(indices),
            reinterpret_cast<T*>(y),
            desc->otherDims,
            desc->dim_size,
            desc->stride,
            desc->indices_size
        );
        else gather_vectorized_kernel<IndexType, true, T><<<gridSize, block_size, sharedMemSize, (cudaStream_t)stream>>>(
            reinterpret_cast<const T*>(x),
            reinterpret_cast<const IndexType*>(indices),
            reinterpret_cast<T*>(y),
            desc->otherDims,
            desc->dim_size,
            desc->stride,
            desc->indices_size
        );
    } else {
        int block_size = 128;
        const int total_elements = desc->otherDims * desc->indices_size;
        const int gridSize = (total_elements + block_size * 4 - 1) / (block_size * 4);
        if (sharedMemSize == 0) gather_kernel<IndexType, false, T><<<gridSize, block_size, 0, (cudaStream_t)stream>>>(
            reinterpret_cast<const T*>(x),
            reinterpret_cast<const IndexType*>(indices),
            reinterpret_cast<T*>(y),
            desc->otherDims,
            desc->dim_size,
            desc->stride,
            desc->indices_size
        );
        else gather_kernel<IndexType, true, T><<<gridSize, block_size, sharedMemSize, (cudaStream_t)stream>>>(
            reinterpret_cast<const T*>(x),
            reinterpret_cast<const IndexType*>(indices),
            reinterpret_cast<T*>(y),
            desc->otherDims,
            desc->dim_size,
            desc->stride,
            desc->indices_size
        );    
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaGather(GatherCudaDescriptor_t desc,
    const void *x,
    const void *indices,
    void *y,
    void *stream){
    if (desc->dtype == F32){
        if (desc->indices_dtype == I32){
            return gather_nv_gpu<float, int32_t>(desc, x, y, indices, stream);
        }
        else if (desc->indices_dtype == I64){
            return gather_nv_gpu<float, int64_t>(desc, x, y, indices, stream);
        }
    }
    if (desc->dtype == F16){
        if (desc->indices_dtype == I32){
            return gather_nv_gpu<half, int32_t>(desc, x, y, indices, stream);
        }
        else if (desc->indices_dtype == I64){
            return gather_nv_gpu<half, int64_t>(desc, x, y, indices, stream);
        }
    }
    return STATUS_BAD_TENSOR_DTYPE;
}