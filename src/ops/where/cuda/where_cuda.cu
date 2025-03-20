#include "../../../devices/cuda/cuda_handle.h"
#include "../../utils.h"
#include "where_cuda.h"
#include <cuda_fp16.h>

#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

__device__ __forceinline__ uint64_t broadcast_map(
    uint64_t idx, 
    const uint64_t* dst_shape,     
    uint64_t dst_ndim,             
    const uint64_t* input_shape,   
    const int64_t* input_strides,  
    uint64_t input_ndim            
) {
    uint64_t index = 0;
    const int offset = dst_ndim - input_ndim;
    for (int i = dst_ndim - 1; i >= 0; idx /= dst_shape[i--]) {
        const uint64_t coord = idx % dst_shape[i];
        if (i >= offset) { 
            const int dim = i - offset;
            const uint64_t size = input_shape[dim];
            index += (size == 1 ? 0 : coord % size) * input_strides[dim];
        }
    }
    return index;
}



__global__ void where_f32x4_kernel(const float* src1, uint64_t const *src1_shape, 
                                 const float* src2, uint64_t const *src2_shape, 
                                 float* dst, uint64_t const *dst_shape, 
                                 const uint8_t* condition, uint64_t const *condition_shape,
                                 uint64_t dst_ndim, uint64_t src1_ndim, 
                                 uint64_t src2_ndim, uint64_t condition_ndim,
                                 int64_t const *condition_strides, 
                                 int64_t const *src1_strides, 
                                 int64_t const *src2_strides, 
                                 int64_t const *dst_strides,
                                 int N) {

    int idx_base = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx_base >= N) return;
    bool valid[4] = {
        (idx_base + 0) < N,
        (idx_base + 1) < N,
        (idx_base + 2) < N,
        (idx_base + 3) < N
    };
    uint64_t condition_offset[4], src1_offset[4], src2_offset[4];
    float4 reg_dst;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (valid[i]) {
            uint64_t idx = idx_base + i;
            condition_offset[i] = broadcast_map(idx, dst_shape, dst_ndim, condition_shape, 
                                                condition_strides, condition_ndim);
            src1_offset[i] = broadcast_map(idx, dst_shape, dst_ndim, src1_shape, 
                                          src1_strides, src1_ndim);
            src2_offset[i] = broadcast_map(idx, dst_shape, dst_ndim, src2_shape, 
                                          src2_strides, src2_ndim);
            float val = condition[condition_offset[i]] ? src1[src1_offset[i]] : src2[src2_offset[i]];
            ((float*)&reg_dst)[i] = val;
        }
    }
    if (valid[0] && valid[1] && valid[2] && valid[3]) {
        FLOAT4(dst[idx_base]) = reg_dst;
    } else {
        for (int i = 0; i < 4; ++i) {
            if (valid[i]) {
                dst[idx_base + i] = ((float*)&reg_dst)[i];
            }
        }
    }
}

__global__ void where_f16x8_kernel(const half* src1, uint64_t const *src1_shape, 
                                const half* src2, uint64_t const *src2_shape, 
                                half* dst, uint64_t const *dst_shape, 
                                const uint8_t* condition, uint64_t const *condition_shape,
                                uint64_t dst_ndim, uint64_t src1_ndim, 
                                uint64_t src2_ndim, uint64_t condition_ndim,
                                int64_t const *condition_strides, 
                                int64_t const *src1_strides, 
                                int64_t const *src2_strides, 
                                int64_t const *dst_strides,
                                int N){
    int idx_base = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx_base >= N) return;
    int64_t condition_offset[8], src1_offset[8], src2_offset[8];
    half pack_dst[8];
    bool valid[8] = {
        (idx_base + 0) < N,
        (idx_base + 1) < N,
        (idx_base + 2) < N,
        (idx_base + 3) < N,
        (idx_base + 4) < N,
        (idx_base + 5) < N,
        (idx_base + 6) < N,
        (idx_base + 7) < N
    };
    #pragma unroll
    for (int i = 0; i < 8; i++){
        if (valid[i]){
            uint64_t idx = idx_base + i;
            condition_offset[i] = broadcast_map(idx, dst_shape, dst_ndim, condition_shape, 
                            condition_strides, condition_ndim);
            src1_offset[i] = broadcast_map(idx, dst_shape, dst_ndim, src1_shape, 
                    src1_strides, src1_ndim);
            src2_offset[i] = broadcast_map(idx, dst_shape, dst_ndim, src2_shape, 
                    src2_strides, src2_ndim);
            pack_dst[i] = condition[condition_offset[i]] ? src1[src1_offset[i]] : src2[src2_offset[i]];
        }
    }
    if (valid[0] && valid[1] && valid[2] && valid[3] && valid[4] && valid[5] && valid[6] && valid[7]){
        LDST128BITS(dst[idx_base]) = LDST128BITS(pack_dst[0]);
    } else {
        for (int i = 0; i < 8; i++){
            if (valid[i]){
                dst[idx_base + i] = pack_dst[i];
            }
        }
    }
}

template<typename Tdata>
infiniopStatus_t where_nv_gpu(
    WhereCudaDescriptor_t desc,
    void *dst, 
    void const *src1,
    void const *src2,
    void const *condition,
    int per_thread_element,
    void *stream){
    uint64_t N = desc->element_num;
    dim3 block(256 / per_thread_element);
    dim3 grid((N + 256 - 1) / 256);
    if constexpr(std::is_same<Tdata, float>::value){
        where_f32x4_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
            reinterpret_cast<const float *>(src1), desc->src1_shape, 
            reinterpret_cast<const float *>(src2), desc->src2_shape, 
            reinterpret_cast<float *>(dst), desc->dst_shape, 
            reinterpret_cast<const uint8_t *>(condition), desc->condition_shape,
            desc->dst_ndim, desc->src1_ndim, 
            desc->src2_ndim, desc->condition_ndim,
            desc->condition_strides, 
            desc->src1_strides, 
            desc->src2_strides, 
            desc->dst_strides,
            N);
    } else if constexpr(std::is_same<Tdata, half>::value){
        where_f16x8_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
            reinterpret_cast<const half *>(src1), desc->src1_shape, 
            reinterpret_cast<const half *>(src2), desc->src2_shape, 
            reinterpret_cast<half *>(dst), desc->dst_shape, 
            reinterpret_cast<const uint8_t *>(condition), desc->condition_shape,
            desc->dst_ndim, desc->src1_ndim, 
            desc->src2_ndim, desc->condition_ndim,
            desc->condition_strides, 
            desc->src1_strides, 
            desc->src2_strides, 
            desc->dst_strides,
            N);
    }
    return STATUS_SUCCESS;
}


infiniopStatus_t cudaWhere(WhereCudaDescriptor_t desc,
    void *dst, 
    void const *src1,
    void const *src2,
    void const *condition,
    void *stream){
    if (desc->dtype == F16){
        return where_nv_gpu<half>(desc, dst, src1, src2, condition, 8, stream);
    }
    if (desc->dtype == F32){
        return where_nv_gpu<float>(desc, dst, src1, src2, condition, 4, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}