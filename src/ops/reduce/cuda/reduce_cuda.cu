#include "../../../devices/cuda/cuda_handle.h"
#include "../../utils.h"
#include "reduce_cuda.h"
#include <cuda_fp16.h>
#include <type_traits>
#include <cuda_runtime.h>
#define WARP_SIZE 32

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
#define FLOAT4_CONST(value) (reinterpret_cast<const float4*>(&(value))[0])
#define LDST128BITS_CONST(value) (reinterpret_cast<const float4*>(&(value))[0])

enum class ReduceOp {
    SUM,
    MIN,
    MAX,
    MEAN
};

__global__ void divide_by_n_kernel(float* y, int N) {
    y[0] /= static_cast<float>(N);
}

__global__ void divide_by_n_kernel(half* y, float* temp_buffer, int N) {
    y[0] = __float2half(temp_buffer[0] / static_cast<float>(N));
}

template <ReduceOp Op, typename T>
__device__ __forceinline__ T init_value() {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (Op == ReduceOp::SUM || Op == ReduceOp::MEAN) 
            return 0.0f;
        else if constexpr (Op == ReduceOp::MAX) 
            return -INFINITY;
        else if constexpr (Op == ReduceOp::MIN) 
            return INFINITY;
    } else if constexpr (std::is_same_v<T, half>) {
        if constexpr (Op == ReduceOp::SUM || Op == ReduceOp::MEAN) 
            return __float2half(0.0f);
            else if constexpr (Op == ReduceOp::MAX) 
                return __float2half(-INFINITY);
            else if constexpr (Op == ReduceOp::MIN) 
                return __float2half(INFINITY);
    }
}

template <ReduceOp Op>
__device__ __forceinline__ float reduce_op(float a, float b) {
    if constexpr (Op == ReduceOp::SUM || Op == ReduceOp::MEAN) 
        return a + b;
    else if constexpr (Op == ReduceOp::MAX) 
        return fmaxf(a, b);
    else if constexpr (Op == ReduceOp::MIN) 
        return fminf(a, b);
}

template <ReduceOp Op, const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        if constexpr (Op == ReduceOp::SUM || Op == ReduceOp::MEAN)
            val += __shfl_down_sync(0xffffffff, val, mask);
        else if constexpr (Op == ReduceOp::MAX)
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, mask));
        else if constexpr (Op == ReduceOp::MIN)
            val = fminf(val, __shfl_down_sync(0xffffffff, val, mask));
    }
    return val;
}

template <ReduceOp Op>
__device__ __forceinline__ float finalize_result(float val, int count) {
    if constexpr (Op == ReduceOp::MEAN)
        return val / static_cast<float>(count);
    else
        return val;
}

template <ReduceOp Op>
__device__ __forceinline__ half reduce_op(half a, half b) {
    if constexpr (Op == ReduceOp::SUM || Op == ReduceOp::MEAN) 
        return __hadd(a, b);
    else if constexpr (Op == ReduceOp::MAX) 
        return __hmax(a, b);
    else if constexpr (Op == ReduceOp::MIN) 
        return __hmin(a, b);
}

template <ReduceOp Op, const int kWarpSize = 32>
__device__ __forceinline__ half warp_reduce(half val) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        if constexpr (Op == ReduceOp::SUM || Op == ReduceOp::MEAN)
            val = __hadd(val, __shfl_down_sync(0xffffffff, val, mask));
        else if constexpr (Op == ReduceOp::MAX)
            val = __hmax(val, __shfl_down_sync(0xffffffff, val, mask));
        else if constexpr (Op == ReduceOp::MIN)
            val = __hmin(val, __shfl_down_sync(0xffffffff, val, mask));
    }
    return val;
}

template <ReduceOp Op>
__device__ __forceinline__ half finalize_result(half val, int count) {
    if constexpr (Op == ReduceOp::MEAN) {
        // 转换为float计算，然后转回half
        float fval = __half2float(val);
        float result = fval / static_cast<float>(count);
        return __float2half(result);
    } else {
        return val;
    }
}

template <ReduceOp Op>
__global__ void warp_final_reduce_kernel(float *temp_in, float *y, int num_blocks) {
    int tid = threadIdx.x;
    float thread_result = init_value<Op, float>();

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        if constexpr (Op == ReduceOp::MAX)
            thread_result = fmaxf(thread_result, temp_in[i]);
        else if constexpr (Op == ReduceOp::MIN)
            thread_result = fminf(thread_result, temp_in[i]);
    }
    thread_result = warp_reduce<Op>(thread_result);
    
    if (tid == 0)
        y[0] = thread_result;
}

template <ReduceOp Op>
__global__ void block_final_reduce_kernel(float *temp_in, float *y, int num_blocks) {
    int tid = threadIdx.x;
    __shared__ float s_data[256];
    
    float thread_result = init_value<Op, float>();

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        if constexpr (Op == ReduceOp::MAX)
            thread_result = fmaxf(thread_result, temp_in[i]);
        else if constexpr (Op == ReduceOp::MIN)
            thread_result = fminf(thread_result, temp_in[i]);
    }

    s_data[tid] = thread_result;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if constexpr (Op == ReduceOp::MAX)
                s_data[tid] = fmaxf(s_data[tid], s_data[tid + stride]);
            else if constexpr (Op == ReduceOp::MIN)
                s_data[tid] = fminf(s_data[tid], s_data[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0)
        y[0] = s_data[0];
}

template <ReduceOp Op>
__global__ void warp_final_reduce_kernel(half *temp_in, half *y, int num_blocks) {
    int tid = threadIdx.x;
    half thread_result = init_value<Op, half>();
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        if constexpr (Op == ReduceOp::MAX)
            thread_result = __hmax(thread_result, temp_in[i]);
        else if constexpr (Op == ReduceOp::MIN)
            thread_result = __hmin(thread_result, temp_in[i]);
    }    
    thread_result = warp_reduce<Op>(thread_result);
    
    if (tid == 0)
        y[0] = thread_result;
}

template <ReduceOp Op>
__global__ void block_final_reduce_kernel(half *temp_in, half *y, int num_blocks) {
    int tid = threadIdx.x;
    __shared__ half s_data[256];
    
    half thread_result = init_value<Op, half>();
    
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        if constexpr (Op == ReduceOp::MAX)
            thread_result = __hmax(thread_result, temp_in[i]);
        else if constexpr (Op == ReduceOp::MIN)
            thread_result = __hmin(thread_result, temp_in[i]);
    }
    
    s_data[tid] = thread_result;
    __syncthreads();
    
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if constexpr (Op == ReduceOp::MAX)
                s_data[tid] = __hmax(s_data[tid], s_data[tid + stride]);
            else if constexpr (Op == ReduceOp::MIN)
                s_data[tid] = __hmin(s_data[tid], s_data[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0)
        y[0] = s_data[0];
}

template <ReduceOp Op>
__global__ void blockall_reduce_f32x4_kernel(const float *x, float *y, uint64_t N){
    int tid = threadIdx.x;
    int base_idx = (tid + 64 * blockIdx.x) * 4;
    constexpr int NUM_WARPS = (64 + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    float result = init_value<Op, float>();
    if (base_idx < N){
        if (base_idx + 3 < N){
            float4 reg_x = FLOAT4_CONST(x[base_idx]);
            if constexpr (Op == ReduceOp::SUM || Op == ReduceOp::MEAN) {
                result += reg_x.x + reg_x.y + reg_x.z + reg_x.w;
            } else if constexpr (Op == ReduceOp::MAX) {
                result = fmaxf(result, fmaxf(fmaxf(reg_x.x, reg_x.y), fmaxf(reg_x.z, reg_x.w)));
            } else if constexpr (Op == ReduceOp::MIN) {
                result = fminf(result, fminf(fminf(reg_x.x, reg_x.y), fminf(reg_x.z, reg_x.w)));
            }
        } else {
            result = reduce_op<Op>(result, x[base_idx]);
            if (base_idx + 1 < N) result = reduce_op<Op>(result, x[base_idx + 1]);
            if (base_idx + 2 < N) result = reduce_op<Op>(result, x[base_idx + 2]);
            if (base_idx + 3 < N) result = reduce_op<Op>(result, x[base_idx + 3]);
        }
    }
    const int warp_id = tid / WARP_SIZE; 
    const int lane = tid % WARP_SIZE;
    result = warp_reduce<Op>(result);
    if (lane == 0) {
        reduce_smem[warp_id] = result;
    }
    __syncthreads(); 
    if (warp_id == 0) {
        result = (lane < NUM_WARPS) ? reduce_smem[lane] : init_value<Op, float>();       
        result = warp_reduce<Op>(result);
        if (tid == 0) {
            if constexpr (Op == ReduceOp::MEAN) {
                atomicAdd(y, result);
            }
            else if constexpr (Op == ReduceOp::MAX || Op == ReduceOp::MIN) y[blockIdx.x] = result;
        }
    }
}

// blockallreduce kernel for half(f16x8) using template operations
template <ReduceOp Op, typename T>
__global__ void blockall_reduce_f16x8_kernel(const half *x, T *y, uint64_t N){
    int tid = threadIdx.x;
    int base_idx = (tid + 32 * blockIdx.x) * 8;
    constexpr int NUM_WARPS = (32 + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ T reduce_smem[NUM_WARPS];
    T result = init_value<Op, T>();
    if (base_idx < N){
        if (base_idx + 7 < N){
            half pack_x[8];
            LDST128BITS(pack_x[0]) = LDST128BITS_CONST(x[base_idx]);
            #pragma unroll
            for (int i = 0; i < 8; i++){
                if constexpr (std::is_same<T, half>::value) {
                    result = reduce_op<Op>(result, pack_x[i]);
                } else {
                    result = reduce_op<Op>(result, __half2float(pack_x[i]));
                }
            }
        } else {
            for (int i = 0; i < 8 && (base_idx + i) < N; i++){
                if constexpr (std::is_same<T, half>::value) {
                    result = reduce_op<Op>(result, x[base_idx + i]);
                } else {
                    result = reduce_op<Op>(result, __half2float(x[base_idx + i]));
                }
            }
        }
    }
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    result = warp_reduce<Op>(result);
    
    if (lane == 0) {
        reduce_smem[warp_id] = result;
    }
    
    __syncthreads();
    
    if (warp_id == 0){
        result = (lane < NUM_WARPS) ? reduce_smem[lane] : init_value<Op, T>();
        result = warp_reduce<Op>(result);
        
        if (tid == 0) {
            if constexpr (Op == ReduceOp::MEAN && std::is_same<T, float>::value) {
                atomicAdd(y, result);
            }
            else if constexpr (Op == ReduceOp::MAX || Op == ReduceOp::MIN) y[blockIdx.x] = result;
        }
    }
}

template <ReduceOp Op>
__global__ void reduce_f32x4_contigous_kernel(
    const float *x, float *y,
    int prefix_size, int suffix_size,
    uint64_t output_size, uint64_t reduce_size){

    extern __shared__ float shared_mem_float[];
    int tid = threadIdx.x;
    int output_idx = blockIdx.x;
    int prefix_idx = output_idx / suffix_size;
    int suffix_idx = output_idx % suffix_size;

    if (output_idx >= output_size) return;
    float result = init_value<Op, float>();
    int base_idx = prefix_idx * reduce_size * suffix_size + suffix_idx;
    // const int vector_reduce_size = reduce_size / 4;
    // suffix_size equals to inner_stride
    for (int i = tid; i < reduce_size; i += blockDim.x){
        result = reduce_op<Op>(result, x[base_idx + i * suffix_size]);
    }

    shared_mem_float[tid] = result;
    __syncthreads();
    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (tid < s) {
            shared_mem_float[tid] = reduce_op<Op>(shared_mem_float[tid], shared_mem_float[tid + s]);
        }
        __syncthreads();
    }
    if (tid < 32){
        result = shared_mem_float[tid];
        result = warp_reduce<Op>(result);
        if (tid == 0) {
            y[output_idx] = finalize_result<Op>(result, reduce_size);
            
        }
    }
}

template <ReduceOp Op>
__global__ void reduce_f16x8_contigous_kernel(
    const half *x, half *y,
    int prefix_size, int suffix_size,
    uint64_t output_size, uint64_t reduce_size){
    extern __shared__ float shared_mem_half[];
    int tid = threadIdx.x;
    int output_idx = blockIdx.x;
    int prefix_idx = output_idx / suffix_size;
    int suffix_idx = output_idx % suffix_size;
 
    if (output_idx >= output_size) return;
    float result = init_value<Op, float>();
    int base_idx = prefix_idx * reduce_size * suffix_size + suffix_idx;
    for (int i = tid; i < reduce_size; i += blockDim.x){
        result = reduce_op<Op>(result, __half2float(x[base_idx + i * suffix_size]));
    }
    shared_mem_half[tid] = result;
    __syncthreads();
    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (tid < s) {
            shared_mem_half[tid] = reduce_op<Op>(shared_mem_half[tid], shared_mem_half[tid + s]);
        }
        __syncthreads();
    }
    if (tid < 32){
        result = shared_mem_half[tid];
        result = warp_reduce<Op>(result);
        // each block has one result
        if (tid == 0) {
            float fresult = finalize_result<Op>(result, reduce_size);
            y[output_idx] = __float2half(fresult);
        }
    }
}
/*
[2, 3, 4, 5] tensor axes = [0, ] -> output tensor [1, 3, 1, 5]
output_idx = 0


*/
template<ReduceOp Op, bool KeepDims>
__global__ void reduce_f32_kernel(
    const float *x, float *y,
    const int64_t *x_strides, const int64_t *y_strides,
    const uint64_t reduce_size, const uint64_t output_size,
    const int64_t *non_reduce_axes,
    const uint64_t axes_size,
    const uint64_t *reduce_axes_stride,
    const int64_t *reduce_axes, const int64_t INPUT_NDIM,
    const uint64_t element_num
){
    extern __shared__ float shared_mem_float[];
    int tid = threadIdx.x;
    int output_idx = blockIdx.x;
    if (output_idx >= output_size) return;
    float result = init_value<Op, float>();
    uint64_t out_coords[10] = {0};
    if constexpr (KeepDims) {
        for (int i = 0; i < INPUT_NDIM - axes_size; i++) {
            out_coords[non_reduce_axes[i]] = output_idx / y_strides[non_reduce_axes[i]];
            output_idx %= y_strides[non_reduce_axes[i]];
        }
    } else {
        for (int i = 0; i < INPUT_NDIM - axes_size; i++) {
            out_coords[non_reduce_axes[i]] = output_idx / y_strides[i];
            output_idx %= y_strides[i];
        }
    }
    uint64_t baseoffset = 0;
    for (int i = 0; i < INPUT_NDIM; i++) {
        baseoffset += out_coords[i] * x_strides[i];
    }
    for (uint64_t i = tid; i < reduce_size; i += blockDim.x){
        uint64_t offset = baseoffset;
        uint64_t remaining = i;
        for (int j = 0; j < axes_size; j++) {
            const uint64_t axis = reduce_axes[j];
            const uint64_t coord = remaining / reduce_axes_stride[j];
            offset += coord * x_strides[axis];
            remaining %= reduce_axes_stride[j];
        }
        result = reduce_op<Op>(result, x[offset]);
    }
    shared_mem_float[tid] = result;
    __syncthreads();
    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (tid < s) {
            shared_mem_float[tid] = reduce_op<Op>(shared_mem_float[tid], shared_mem_float[tid + s]);
        }
        __syncthreads();
    }
    if (tid < 32){
        result = shared_mem_float[tid];
        result = warp_reduce<Op>(result);
        if (tid == 0) {
            y[blockIdx.x] = finalize_result<Op>(result, reduce_size);
        }
    }
}


template<typename Tdata>
infiniopStatus_t reduce_nv_gpu(
    ReduceCudaDescriptor_t desc,
    void *y,
    void const *x,
    void *stream){
    uint64_t N = desc->element_num;
    if (desc->reduce_mode == 0){
        if constexpr(std::is_same<Tdata, float>::value){
            dim3 block(256 / 4);
            dim3 grid((N + block.x * 4 - 1) / (block.x * 4));
            if (desc->reduce_op_type == 0) {
                blockall_reduce_f32x4_kernel<ReduceOp::MEAN><<<grid, block, 0, (cudaStream_t)stream>>>(reinterpret_cast<const float *>(x), reinterpret_cast<float *>(y), N);
                divide_by_n_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(
                    reinterpret_cast<float *>(y), N);
            } else if (desc->reduce_op_type == 1 || desc->reduce_op_type == 2) {
                float* temp_buffer;
                cudaMalloc(&temp_buffer, grid.x * sizeof(float));
                if (desc->reduce_op_type == 1) {            
                    blockall_reduce_f32x4_kernel<ReduceOp::MAX><<<grid, block, 0, (cudaStream_t)stream>>>(reinterpret_cast<const float *>(x), temp_buffer, N);
                } else if (desc->reduce_op_type == 2) {
                    blockall_reduce_f32x4_kernel<ReduceOp::MIN><<<grid, block, 0, (cudaStream_t)stream>>>(reinterpret_cast<const float *>(x), temp_buffer, N);
                }
                int num_blocks = grid.x;
                if (num_blocks <= 32) {
                    if (desc->reduce_op_type == 1) { // MAX
                        warp_final_reduce_kernel<ReduceOp::MAX><<<1, 32, 0, (cudaStream_t)stream>>>(
                            temp_buffer, reinterpret_cast<float *>(y), num_blocks);
                    } else {
                        warp_final_reduce_kernel<ReduceOp::MIN><<<1, 32, 0, (cudaStream_t)stream>>>(
                            temp_buffer, reinterpret_cast<float *>(y), num_blocks);
                    }
                } else {
                    if (desc->reduce_op_type == 1) {
                        block_final_reduce_kernel<ReduceOp::MAX><<<1, 256, 0, (cudaStream_t)stream>>>(
                            temp_buffer, reinterpret_cast<float *>(y), num_blocks);
                    } else {
                        block_final_reduce_kernel<ReduceOp::MIN><<<1, 256, 0, (cudaStream_t)stream>>>(
                            temp_buffer, reinterpret_cast<float *>(y), num_blocks);
                    }
                }
                cudaFree(temp_buffer);
            }
        } else {
            dim3 block(256 / 8);
            dim3 grid((N + block.x * 8 - 1) / (block.x * 8));
            if (desc->reduce_op_type == 0) {
                float* temp_buffer;
                cudaMalloc(&temp_buffer, sizeof(float));
                blockall_reduce_f16x8_kernel<ReduceOp::MEAN, float><<<grid, block, 0, (cudaStream_t)stream>>>(reinterpret_cast<const half *>(x), temp_buffer, N);
                divide_by_n_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(
                    reinterpret_cast<half *>(y), temp_buffer, N);
            } else if (desc->reduce_op_type == 1 || desc->reduce_op_type == 2) {
                half *temp_buffer;
                cudaMalloc(&temp_buffer, grid.x * sizeof(half));
                if (desc->reduce_op_type == 1) {
                    blockall_reduce_f16x8_kernel<ReduceOp::MAX, half><<<grid, block, 0, (cudaStream_t)stream>>>(reinterpret_cast<const half *>(x), reinterpret_cast<half *>(temp_buffer), N);
                } else if (desc->reduce_op_type == 2) {
                    blockall_reduce_f16x8_kernel<ReduceOp::MIN, half><<<grid, block, 0, (cudaStream_t)stream>>>(reinterpret_cast<const half *>(x), reinterpret_cast<half *>(temp_buffer), N);
                }
                int num_blocks = grid.x;
                if (num_blocks <= 32) {
                    if (desc->reduce_op_type == 1) {
                        warp_final_reduce_kernel<ReduceOp::MAX><<<1, 32, 0, (cudaStream_t)stream>>>(
                            temp_buffer, reinterpret_cast<half *>(y), num_blocks);
                    } else {
                        warp_final_reduce_kernel<ReduceOp::MIN><<<1, 32, 0, (cudaStream_t)stream>>>(
                            temp_buffer, reinterpret_cast<half *>(y), num_blocks);
                    }
                } else {
                    if (desc->reduce_op_type == 1) {
                        block_final_reduce_kernel<ReduceOp::MAX><<<1, 256, 0, (cudaStream_t)stream>>>(
                            temp_buffer, reinterpret_cast<half *>(y), num_blocks);
                    } else {
                        block_final_reduce_kernel<ReduceOp::MIN><<<1, 256, 0, (cudaStream_t)stream>>>(
                            temp_buffer, reinterpret_cast<half *>(y), num_blocks);
                    }
                }
                cudaFree(temp_buffer);
            }
        }
    }
    // contiguous axes
    else if (desc->reduce_mode == 1){
        if constexpr(std::is_same<Tdata, float>::value){
            dim3 block(128);
            dim3 grid(desc->output_size);
            size_t shared_mem_size = block.x * sizeof(float);
            if (desc->reduce_op_type == 0) {
                reduce_f32x4_contigous_kernel<ReduceOp::MEAN><<<grid, block, shared_mem_size, (cudaStream_t)stream>>>(
                    reinterpret_cast<const float *>(x), reinterpret_cast<float *>(y), desc->prefix_size, desc->suffix_size, desc->output_size, desc->reduce_size);
            } else if (desc->reduce_op_type == 1) {
                reduce_f32x4_contigous_kernel<ReduceOp::MAX><<<grid, block, shared_mem_size, (cudaStream_t)stream>>>(
                    reinterpret_cast<const float *>(x), reinterpret_cast<float *>(y), desc->prefix_size, desc->suffix_size, desc->output_size, desc->reduce_size);
            } else if (desc->reduce_op_type == 2) {
                reduce_f32x4_contigous_kernel<ReduceOp::MIN><<<grid, block, shared_mem_size, (cudaStream_t)stream>>>(
                    reinterpret_cast<const float *>(x), reinterpret_cast<float *>(y), desc->prefix_size, desc->suffix_size, desc->output_size, desc->reduce_size);
            }
        } else {
            dim3 block(128);
            dim3 grid(desc->output_size);
            size_t shared_mem_size = block.x * sizeof(half);
            if (desc->reduce_op_type == 0) {
                reduce_f16x8_contigous_kernel<ReduceOp::MEAN><<<grid, block, shared_mem_size, (cudaStream_t)stream>>>(
                    reinterpret_cast<const half *>(x), reinterpret_cast<half *>(y), desc->prefix_size, desc->suffix_size, desc->output_size, desc->reduce_size);
            }else if (desc->reduce_op_type == 1) {
                reduce_f16x8_contigous_kernel<ReduceOp::MAX><<<grid, block, shared_mem_size, (cudaStream_t)stream>>>(
                    reinterpret_cast<const half *>(x), reinterpret_cast<half *>(y), desc->prefix_size, desc->suffix_size, desc->output_size, desc->reduce_size);
            } else if (desc->reduce_op_type == 2) {
                reduce_f16x8_contigous_kernel<ReduceOp::MIN><<<grid, block, shared_mem_size, (cudaStream_t)stream>>>(
                    reinterpret_cast<const half *>(x), reinterpret_cast<half *>(y), desc->prefix_size, desc->suffix_size, desc->output_size, desc->reduce_size);
            }
        }
    }
    // not contiguous axes
    else if (desc->reduce_mode == 2){
        dim3 block(128);
        dim3 grid(desc->output_size);
        size_t shared_mem_size = block.x * sizeof(float);
        if constexpr(std::is_same<Tdata, float>::value){
            if (desc->keepdims){
                if (desc->reduce_op_type == 0) {
                    reduce_f32_kernel<ReduceOp::MEAN, true><<<grid, block, shared_mem_size, (cudaStream_t)stream>>>(
                        reinterpret_cast<const float *>(x), reinterpret_cast<float *>(y),
                        desc->input_strides, desc->output_strides, desc->reduce_size, desc->output_size,
                        desc->non_reduce_axes, desc->axes_size, desc->reduce_axes_stride, desc->reduce_axes, desc->input_ndim, desc->element_num);
                }
                else if (desc->reduce_op_type == 1) {
                    reduce_f32_kernel<ReduceOp::MAX, true><<<grid, block, shared_mem_size, (cudaStream_t)stream>>>(
                        reinterpret_cast<const float *>(x), reinterpret_cast<float *>(y),
                        desc->input_strides, desc->output_strides, desc->reduce_size, desc->output_size,
                        desc->non_reduce_axes, desc->axes_size, desc->reduce_axes_stride, desc->reduce_axes, desc->input_ndim, desc->element_num);
                }
                else if (desc->reduce_op_type == 2) {
                    reduce_f32_kernel<ReduceOp::MIN, true><<<grid, block, shared_mem_size, (cudaStream_t)stream>>>(
                        reinterpret_cast<const float *>(x), reinterpret_cast<float *>(y),
                        desc->input_strides, desc->output_strides, desc->reduce_size, desc->output_size,
                        desc->non_reduce_axes, desc->axes_size, desc->reduce_axes_stride, desc->reduce_axes, desc->input_ndim, desc->element_num);
                }
            }else{
                if (desc->reduce_op_type == 0) {
                    reduce_f32_kernel<ReduceOp::MEAN, false><<<grid, block, shared_mem_size, (cudaStream_t)stream>>>(
                        reinterpret_cast<const float *>(x), reinterpret_cast<float *>(y),
                        desc->input_strides, desc->output_strides, desc->reduce_size, desc->output_size,
                        desc->non_reduce_axes, desc->axes_size, desc->reduce_axes_stride, desc->reduce_axes, desc->input_ndim, desc->element_num);
                }
                else if (desc->reduce_op_type == 1) {
                    reduce_f32_kernel<ReduceOp::MAX, false><<<grid, block, shared_mem_size, (cudaStream_t)stream>>>(
                        reinterpret_cast<const float *>(x), reinterpret_cast<float *>(y),
                        desc->input_strides, desc->output_strides, desc->reduce_size, desc->output_size,
                        desc->non_reduce_axes, desc->axes_size, desc->reduce_axes_stride, desc->reduce_axes, desc->input_ndim, desc->element_num);
                }
                else if (desc->reduce_op_type == 2) {
                    reduce_f32_kernel<ReduceOp::MIN, false><<<grid, block, shared_mem_size, (cudaStream_t)stream>>>(
                        reinterpret_cast<const float *>(x), reinterpret_cast<float *>(y),
                        desc->input_strides, desc->output_strides, desc->reduce_size, desc->output_size,
                        desc->non_reduce_axes, desc->axes_size, desc->reduce_axes_stride, desc->reduce_axes, desc->input_ndim, desc->element_num);
                }
            }
        }    
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaReduce(
    ReduceCudaDescriptor_t desc,
    void *y,
    void const *x,
    void *stream){
    if (desc->dtype == F16) {
        return reduce_nv_gpu<half>(desc, y, x, stream);
    }
    if (desc->dtype == F32) {
        return reduce_nv_gpu<float>(desc, y, x, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}