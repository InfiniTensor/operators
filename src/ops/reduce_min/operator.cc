#include "../reduce/reduce.h"
#include "../utils.h"
#include "ops/reduce_min/reduce_min.h"

struct _ReduceMinDescriptor {
    Device device;
    infiniopReduceDescriptor_t reduce_desc;
};

typedef struct _ReduceMinDescriptor *_ReduceMinDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReduceMinDescriptor(infiniopHandle_t handle,
                                                                infiniopReduceMinDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t reduced,
                                                                infiniopTensorDescriptor_t data,
                                                                int64_t const *axes,
                                                                size_t axes_size,
                                                                int keepdims,
                                                                int noop_with_empty_axes) {
    infiniopReduceDescriptor_t reduce_desc;
    CHECK_STATUS(
        infiniopCreateReduceDescriptor(handle, &reduce_desc, reduced, data, axes, axes_size, keepdims, noop_with_empty_axes, 2),
        STATUS_SUCCESS);

    *(_ReduceMinDescriptor_t *)desc_ptr = new _ReduceMinDescriptor{handle->device, reduce_desc};

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopReduceMin(infiniopReduceMinDescriptor_t desc,
                                                void *reduced,
                                                void const *data,
                                                int64_t const *axes,
                                                void *stream) {
    auto _desc = (_ReduceMinDescriptor_t)desc;
    CHECK_STATUS(infiniopReduce(_desc->reduce_desc, reduced, data, axes, stream),
                 STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyReduceMinDescriptor(infiniopReduceMinDescriptor_t desc) {
    auto _desc = (_ReduceMinDescriptor_t)desc;
    CHECK_STATUS(infiniopDestroyReduceDescriptor(_desc->reduce_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}
