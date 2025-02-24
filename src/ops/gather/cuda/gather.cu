#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "gather.cuh"

constexpr int num_elem_per_thread = 2;

template<typename Tdata, typename Tind>
__global__ void gather_0(
    Tdata const *data,
    Tind const *indices,
    uint64_t indices_size,
    Tdata *output,
    uint64_t pre_size,
    uint64_t axis_size,
    uint64_t post_size,
    uint64_t offset_x,
    uint64_t offset_y) {
    const uint64_t x = blockIdx.x * blockDim.x + threadIdx.x + offset_x;
    const uint64_t y = blockIdx.y * blockDim.y + threadIdx.y + offset_y;
    if (x >= pre_size * indices_size || y >= post_size) return;

    for (uint64_t i = 0; i < num_elem_per_thread; i++) {
        const uint64_t idx = x * num_elem_per_thread + i;
        if (idx >= pre_size * indices_size) return;

        const uint64_t pre_idx = idx / indices_size;
        const uint64_t indices_idx = idx % indices_size;
        const uint64_t post_idx = y;

        const uint64_t data_idx = pre_idx * axis_size * post_size + indices[indices_idx] * post_size + post_idx;
        const uint64_t output_idx = pre_idx * indices_size * post_size + indices_idx * post_size + post_idx;
        output[output_idx] = data[data_idx];
    }
}

template<typename Tdata, typename Tind>
__global__ void gather_1(
    Tdata const *data,
    Tind const *indices,
    uint64_t indices_size,
    Tdata *output,
    uint64_t pre_size,
    uint64_t axis_size,
    uint64_t post_size,
    uint64_t offset_x,
    uint64_t offset_y) {
    const uint64_t x = blockIdx.x * blockDim.x + threadIdx.x + offset_x;
    const uint64_t y = blockIdx.y * blockDim.y + threadIdx.y + offset_y;
    if (x >= indices_size * post_size || y >= pre_size) return;

    for (uint64_t i = 0; i < num_elem_per_thread; i++) {
        const uint64_t idx = x * num_elem_per_thread + i;
        if (idx >= indices_size * post_size) return;

        const uint64_t pre_idx = y;
        const uint64_t indices_idx = idx / post_size;
        const uint64_t post_idx = idx % post_size;

        const uint64_t data_idx = pre_idx * axis_size * post_size + indices[indices_idx] * post_size + post_idx;
        const uint64_t output_idx = pre_idx * indices_size * post_size + indices_idx * post_size + post_idx;
        output[output_idx] = data[data_idx];
    }
}

template<typename Tdata, typename Tind>
__global__ void gather_2(
    Tdata const *data,
    Tind const *indices,
    uint64_t indices_size,
    Tdata *output,
    uint64_t pre_size,
    uint64_t axis_size,
    uint64_t post_size,
    uint64_t offset_x,
    uint64_t offset_y) {
    const uint64_t x = blockIdx.x * blockDim.x + threadIdx.x + offset_x;
    const uint64_t y = blockIdx.y * blockDim.y + threadIdx.y + offset_y;
    if (x >= pre_size * post_size || y >= indices_size) return;

    for (uint64_t i = 0; i < num_elem_per_thread; i++) {
        const uint64_t idx = x * num_elem_per_thread + i;
        if (idx >= pre_size * post_size) return;

        const uint64_t pre_idx = idx / post_size;
        const uint64_t indices_idx = y;
        const uint64_t post_idx = idx % post_size;

        const uint64_t data_idx = pre_idx * axis_size * post_size + indices[indices_idx] * post_size + post_idx;
        const uint64_t output_idx = pre_idx * indices_size * post_size + indices_idx * post_size + post_idx;
        output[output_idx] = data[data_idx];
    }
}

template<typename Tdata, typename Tind>
infiniopStatus_t gather_nv_gpu(GatherCudaDescriptor_t desc, void *output, void const *input, void const *indices, void *stream) {
    if (desc->output_size == 0) {
        return STATUS_SUCCESS;
    }

    const auto input_ = reinterpret_cast<Tdata const *>(input);
    const auto output_ = reinterpret_cast<Tdata *>(output);
    const auto indices_ = reinterpret_cast<Tind const *>(indices);
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    uint64_t pre_size = desc->pre_size;
    uint64_t axis_size = desc->axis_size;
    uint64_t post_size = desc->post_size;
    uint64_t indices_size = desc->indices_size;

    int kernel_type = desc->kernel_type;
    switch (kernel_type) {
        case 0: {
            dim3 block_size(std::min(static_cast<uint64_t>(64), pre_size * indices_size / num_elem_per_thread),
                            std::min(static_cast<uint64_t>(16), post_size));
            dim3 grid_size(std::min(ROUND_UP_DIV(pre_size * indices_size / num_elem_per_thread, block_size.x), desc->max_grid_size),
                           std::min(ROUND_UP_DIV(post_size, block_size.y), desc->max_grid_size));
            uint64_t step_x = grid_size.x * block_size.x;
            uint64_t step_y = grid_size.y * block_size.y;
        #pragma unroll
            for(uint64_t i = 0; i < pre_size * indices_size; i += step_x) {
                for(uint64_t j = 0; j < post_size; j += step_y) {
                    gather_0<Tdata, Tind><<<grid_size, block_size, 0, cuda_stream>>>(
                        input_, indices_, indices_size, output_, pre_size, axis_size, post_size, i, j);
                }
            }
        }
        break;
        case 1: {
            dim3 block_size(std::min(static_cast<uint64_t>(64), indices_size * post_size / num_elem_per_thread),
                            std::min(static_cast<uint64_t>(16), pre_size));
            dim3 grid_size(std::min(ROUND_UP_DIV(indices_size * post_size / num_elem_per_thread, block_size.x), desc->max_grid_size),
                           std::min(ROUND_UP_DIV(pre_size, block_size.y), desc->max_grid_size));
            uint64_t step_x = grid_size.x * block_size.x;
            uint64_t step_y = grid_size.y * block_size.y;
        #pragma unroll
            for(uint64_t i = 0; i < indices_size * post_size; i += step_x) {
                for(uint64_t j = 0; j < pre_size; j += step_y) {
                    gather_1<Tdata, Tind><<<grid_size, block_size, 0, cuda_stream>>>(
                        input_, indices_, indices_size, output_, pre_size, axis_size, post_size, i, j);
                }
            }
        }
        break;
        case 2: {
            dim3 block_size(std::min(static_cast<uint64_t>(64), pre_size * post_size / num_elem_per_thread),
                            std::min(static_cast<uint64_t>(16), indices_size));
            dim3 grid_size(std::min(ROUND_UP_DIV(pre_size * post_size / num_elem_per_thread, block_size.x), desc->max_grid_size),
                           std::min(ROUND_UP_DIV(indices_size, block_size.y), desc->max_grid_size));
            uint64_t step_x = grid_size.x * block_size.x;
            uint64_t step_y = grid_size.y * block_size.y;
        #pragma unroll
            for(uint64_t i = 0; i < pre_size * post_size; i += step_x) {
                for(uint64_t j = 0; j < indices_size; j += step_y) {
                    gather_2<Tdata, Tind><<<grid_size, block_size, 0, cuda_stream>>>(
                        input_, indices_, indices_size, output_, pre_size, axis_size, post_size, i, j);
                }
            }
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaGather(GatherCudaDescriptor_t desc, void *output, void const *input, void const *indices, void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    if (desc->dtype == F16) {
        if (desc->indices_dtype == I32) {
            return gather_nv_gpu<half, int32_t>(desc, output, input, indices, stream);
        }
        if (desc->indices_dtype == I64) {
            return gather_nv_gpu<half, int64_t>(desc, output, input, indices, stream);
        }
    }
    if (desc->dtype == F32) {
        if (desc->indices_dtype == I32) {
            return gather_nv_gpu<float, int32_t>(desc, output, input, indices, stream);
        }
        if (desc->indices_dtype == I64) {
            return gather_nv_gpu<float, int64_t>(desc, output, input, indices, stream);
        }
    }
    return STATUS_BAD_TENSOR_DTYPE;
}