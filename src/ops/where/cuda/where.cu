#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "status.h"
#include "where.cuh"

__device__ inline uint64_t broadcast_map(uint64_t idx, uint64_t ndim, uint64_t const *shapes, int64_t const *strides) {
    uint64_t index = 0;
    for (uint64_t i = 0; i < ndim; i++) {
        index += (idx / strides[i]) % shapes[i] * strides[i];
    }
    return index;
}

template<typename Tdata>
__global__ void where(
    Tdata const *x,
    bool x_broadcast,
    uint64_t x_ndim,
    uint64_t const *x_shapes,
    int64_t const *x_strides,
    bool y_broadcast,
    Tdata const *y,
    uint64_t y_ndim,
    uint64_t const *y_shapes,
    int64_t const *y_strides,
    bool condition_broadcast,
    uint8_t const *condition,
    uint64_t condition_ndim,
    uint64_t const *condition_shapes,
    int64_t const *condition_strides,
    Tdata *output,
    uint64_t output_size,
    uint64_t offset) {
    const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx >= output_size) return;

    if (condition_broadcast) {
        uint64_t condition_idx = broadcast_map(idx, condition_ndim, condition_shapes, condition_strides);
        if (condition[condition_idx] == 0) {
            output[idx] = y_broadcast ? y[broadcast_map(idx, y_ndim, y_shapes, y_strides)] : y[idx];
        } else {
            output[idx] = x_broadcast ? x[broadcast_map(idx, x_ndim, x_shapes, x_strides)] : x[idx];
        }
    } else {
        output[idx] = condition[idx] == 0 ? (y_broadcast ? y[broadcast_map(idx, y_ndim, y_shapes, y_strides)] : y[idx])
                                          : (x_broadcast ? x[broadcast_map(idx, x_ndim, x_shapes, x_strides)] : x[idx]);
    }
}

template<typename Tdata>
infiniopStatus_t where_nv_gpu(
    WhereCudaDescriptor_t desc,
    void const *x,
    void const *y,
    void const *condition,
    void *output,
    cudaStream_t stream) {

    const uint64_t output_size = desc->output_size;
    const uint64_t max_grid_size = desc->max_grid_size;

    auto x_ = reinterpret_cast<const Tdata *>(x);
    auto y_ = reinterpret_cast<const Tdata *>(y);
    auto condition_ = reinterpret_cast<const uint8_t *>(condition);
    auto output_ = reinterpret_cast<Tdata *>(output);

    bool x_broadcast = desc->x_broadcast;
    bool y_broadcast = desc->y_broadcast;
    bool condition_broadcast = desc->condition_broadcast;

    uint64_t x_ndim = desc->x_ndim;
    uint64_t y_ndim = desc->y_ndim;
    uint64_t condition_ndim = desc->condition_ndim;

    uint64_t const *x_shapes = reinterpret_cast<uint64_t const *>(desc->x_shapes);
    int64_t const *x_strides = reinterpret_cast<int64_t const *>(desc->x_strides);
    uint64_t const *y_shapes = reinterpret_cast<uint64_t const *>(desc->y_shapes);
    int64_t const *y_strides = reinterpret_cast<int64_t const *>(desc->y_strides);
    uint64_t const *condition_shapes = reinterpret_cast<uint64_t const *>(desc->condition_shapes);
    int64_t const *condition_strides = reinterpret_cast<int64_t const *>(desc->condition_strides);

    dim3 blockDims(std::min(static_cast<uint64_t>(256), output_size));
    dim3 gridDims(std::min(ROUND_UP_DIV(output_size, blockDims.x), max_grid_size));
    uint64_t step = gridDims.x * blockDims.x;

#pragma unroll
    for (uint64_t offset = 0; offset < output_size; offset += step) {
        where<<<gridDims, blockDims, 0, stream>>>(
            x_, x_broadcast, x_ndim, x_shapes, x_strides, y_broadcast, y_, y_ndim, y_shapes, y_strides, condition_broadcast, condition_, condition_ndim, condition_shapes, condition_strides, output_, output_size, offset);
    }

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaWhere(WhereCudaDescriptor_t desc, void *output, const void *x, const void *y, const void *condition, void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    if (desc->dtype == F16) {
        return where_nv_gpu<half>(desc, reinterpret_cast<const half *>(x), reinterpret_cast<const half *>(y), reinterpret_cast<const uint8_t *>(condition), reinterpret_cast<half *>(output), reinterpret_cast<cudaStream_t>(stream));
    }
    if (desc->dtype == F32) {
        return where_nv_gpu<float>(desc, reinterpret_cast<const float *>(x), reinterpret_cast<const float *>(y), reinterpret_cast<const uint8_t *>(condition), reinterpret_cast<float *>(output), reinterpret_cast<cudaStream_t>(stream));
    }
    return STATUS_SUCCESS;
}