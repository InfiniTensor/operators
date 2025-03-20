#include "../../../devices/cuda/cuda_handle.h"
#include "../../utils.h"
#include "clip_cuda.h"
#include <cuda_fp16.h>

#define WARP_SIZE 32
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

#define LDST128BITS_CONST(value) (reinterpret_cast<float4 const *>(&(value))[0])
#define FLOAT4_CONST(value) (reinterpret_cast<float4 const *>(&(value))[0])

__global__ void clip_f32x4_kernel(const float *a, float *b, float max_value, float min_value, int N){
    int idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    if (idx < N) {
        int remaining = N - idx;
        float4 reg_a, reg_b;
        if (remaining >= 4) {
        reg_a = FLOAT4_CONST(a[idx]);
        } else {
            reg_a.x = a[idx];
            reg_a.y = (remaining >= 2) ? a[idx + 1] : 0;
            reg_a.z = (remaining >= 3) ? a[idx + 2] : 0;
            reg_a.w = 0;
        }
        reg_b.x = fminf(fmaxf(reg_a.x, min_value), max_value);
        reg_b.y = fminf(fmaxf(reg_a.y, min_value), max_value);
        reg_b.z = fminf(fmaxf(reg_a.z, min_value), max_value);
        reg_b.w = fminf(fmaxf(reg_a.w, min_value), max_value);
        if (remaining >= 4) {
            FLOAT4(b[idx]) = reg_b;
        } else {
            if (remaining >= 1) b[idx]     = reg_b.x;
            if (remaining >= 2) b[idx + 1] = reg_b.y;
            if (remaining >= 3) b[idx + 2] = reg_b.z;
        }
    }
}


__global__ void clip_f16x8_pack_kernel(const half *a, half *b, float max_value, float min_value, int N){
    int idx = 8 * (blockDim.x * blockIdx.x + threadIdx.x);
    if (idx >= N) return;
    const half min_half = __float2half(min_value);
    const half max_half = __float2half(max_value);
    half pack_a[8], pack_b[8];
    if (idx + 7 < N) {
        LDST128BITS(pack_a[0]) = LDST128BITS_CONST(a[idx]);
    } else {
        for (int i = 0; i < 8 && (idx + i) < N; i++) {
            pack_a[i] = a[idx + i];
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        pack_b[i] = __hlt(pack_a[i], min_half) ? min_half : pack_a[i];
        pack_b[i] = __hgt(pack_a[i], max_half) ? max_half : pack_a[i];
    }
    if (idx + 7 < N) {
        LDST128BITS(b[idx]) = LDST128BITS(pack_b[0]);
    } else {
        for (int i = 0; i < 8 && (idx + i) < N; i++) {
            b[idx + i] = pack_b[i];
        }
    }
}
template<typename Tdata>
infiniopStatus_t clip_nv_gpu(
    ClipCudaDescriptor_t desc,
    void const *x,
    void *y,
    int per_thread_element,
    void* stream) {
    uint64_t N = desc->element_num;
    dim3 block(256 / per_thread_element);
    dim3 grid((N + 256 - 1) / 256);
    if constexpr(std::is_same<Tdata, float>::value){
        clip_f32x4_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(reinterpret_cast<const float *>(x), reinterpret_cast<float *>(y), desc->max, desc->min, N);
    }else{
        clip_f16x8_pack_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(reinterpret_cast<const half *>(x), reinterpret_cast<half *>(y), desc->max, desc->min, N);
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaClip(ClipCudaDescriptor_t desc,
    void const *x,
    void *y,
    void *stream){
    if (desc->dtype == F16) {
        return clip_nv_gpu<half>(desc, x, y, 8, stream);
    }
    if (desc->dtype == F32) {
        return clip_nv_gpu<float>(desc, x, y, 4, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
