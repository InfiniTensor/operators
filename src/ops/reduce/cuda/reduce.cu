#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "reduce.cuh"
#include <inttypes.h>

template<typename Tdata>
__global__ void reduce(
    Tdata *dst,
    Tdata const*src,
    const int64_t *stride_axis,
    const int64_t *shape_axis,
    int axis_num,
    uint64_t add_dim,
    uint64_t offset,
    int reduce_type) {
    int64_t threadId = threadIdx.y * blockDim.x + threadIdx.x;
    int64_t idx = offset + threadId + blockIdx.x * add_dim;
    
    extern __shared__ unsigned char sharedMemory[];

    Tdata* sharedData = reinterpret_cast<Tdata*>(sharedMemory);


        int64_t shift_idx = idx;
    #pragma unroll
        for (int i=0; i<axis_num; i++){
            shift_idx = shift_idx/(stride_axis[i] * shape_axis[i]) * (stride_axis[i] * shape_axis[i]) + shift_idx % shape_axis[i] * stride_axis[i] + shift_idx / shape_axis[i] % stride_axis[i];
        }
        sharedData[threadId] = src[shift_idx];
        
        __syncthreads();

        int64_t pre_s = add_dim;
        int64_t s = add_dim / 2;
    #pragma unroll
        while(s > 0) {
            if (threadId < s && (threadId + s) < pre_s) {
                switch(reduce_type) {
                    case 0:
                        if constexpr (std::is_same<Tdata, half>::value) {
                            sharedData[threadId] = __hmax(sharedData[threadId], sharedData[threadId + s]);
                        } else {
                            sharedData[threadId] = fmaxf(sharedData[threadId], sharedData[threadId + s]);
                        }
                        break;
                    case 1: 
                        if constexpr (std::is_same<Tdata, half>::value) {
                            sharedData[threadId] = __hmin(sharedData[threadId], sharedData[threadId + s]);
                        } else {
                            sharedData[threadId] = fminf(sharedData[threadId], sharedData[threadId + s]);
                        }
                        break;
                    default: 
                        if constexpr (std::is_same<Tdata, half>::value) {
                            sharedData[threadId] = __hadd(sharedData[threadId], sharedData[threadId + s]);
                        } else {
                            sharedData[threadId] += sharedData[threadId + s];
                        }
                        break;
                }
            }
            __syncthreads();
            if (pre_s % 2 != 0 && threadId == 0) {
                switch(reduce_type) {
                    case 0:
                        if constexpr (std::is_same<Tdata, half>::value) {
                            sharedData[0] = __hmax(sharedData[0], sharedData[pre_s - 1]);
                        } else {
                            sharedData[0] = fmaxf(sharedData[0], sharedData[pre_s - 1]);
                        }
                        break;
                    case 1: 
                        if constexpr (std::is_same<Tdata, half>::value) {
                            sharedData[0] = __hmin(sharedData[0], sharedData[pre_s - 1]);
                        } else {
                            sharedData[0] = fminf(sharedData[0], sharedData[pre_s - 1]);
                        }
                        break;
                    default: 
                        if constexpr (std::is_same<Tdata, half>::value) {
                            sharedData[0] = __hadd(sharedData[0], sharedData[pre_s-1]);
                        } else {
                            sharedData[0] += sharedData[pre_s-1];
                        }
                        break;
                }
            }
            __syncthreads();

            pre_s = s;
            s = pre_s / 2;
        }

        if(threadId == 0)
        {
            int64_t index = shift_idx;
        #pragma unroll  
            for(int j = 0; j < axis_num; j++){
                index = index / (stride_axis[j] * shape_axis[j]) * stride_axis[j] + index % stride_axis[j];
            }
            dst[index] = sharedData[threadId];
            if(reduce_type == 2)
            {
                float shape = static_cast<float>(add_dim);
                if constexpr (std::is_same<Tdata, half>::value) {
                    dst[index] = __float2half(__half2float(dst[index]) / (shape));
                    //dst[index] /= __float2half(shape);
                } else {
                    dst[index] /= shape;
                }
            }
        }
    
}

template<typename Tdata>
infiniopStatus_t reduce_nv_gpu(ReduceCudaDescriptor_t desc, void *dst, void const *src, void *stream) {
    if (desc->data_size == 0) {
        return STATUS_SUCCESS;
    }
    const int64_t *stride_axis = reinterpret_cast<const int64_t *>(desc->stride_axis_ptr);
    const int64_t *shape_axis = reinterpret_cast<const int64_t *>(desc->shape_axis_ptr);

    dim3 blockDims, gridDims;
    size_t sharedMemSize;
    uint64_t step;
    if (desc->add_dim > 1024){
        blockDims = dim3(1024, ROUND_UP_DIV(desc->add_dim, 1024));
        sharedMemSize = desc->add_dim * sizeof(Tdata);
        step = blockDims.x * blockDims.y;
    }else{
        blockDims = dim3(desc->add_dim);
        sharedMemSize = blockDims.x * sizeof(Tdata);
        step = blockDims.x;
    }
    gridDims = dim3(std::min(ROUND_UP_DIV(desc->data_size, desc->add_dim), desc->max_grid_size));

    step *= gridDims.x;
    

    const auto dst_ = reinterpret_cast<Tdata *>(dst);
    const auto src_ = reinterpret_cast<Tdata const *>(src);
    
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

#pragma unroll
        for (uint64_t i = 0; i < desc->data_size; i += step) {
            reduce<Tdata><<<gridDims, blockDims, sharedMemSize, cuda_stream>>>(
                dst_, src_, stride_axis, shape_axis, desc->axis_num, desc->add_dim, i, desc->reduce_mode);
        }

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaReduce(ReduceCudaDescriptor_t desc,
    void *workspace, 
    uint64_t workspace_size,
    void *dst,     
    void const *src,
    void *stream){
    checkCudaError(cudaSetDevice(desc->device_id));
    if (desc->dtype == F16) {
        return reduce_nv_gpu<half>(desc, dst, src, stream);
    }
    if (desc->dtype == F32) {
        return reduce_nv_gpu<float>(desc, dst, src, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}