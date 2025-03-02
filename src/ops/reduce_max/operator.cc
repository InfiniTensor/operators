#include "../reduce/reduce.h"
#include "../utils.h"
#include "ops/reduce_max/reduce_max.h"

struct _ReduceMaxDescriptor {
    Device device;
    infiniopReduceDescriptor_t reduce_desc;
};

typedef struct _ReduceMaxDescriptor *_ReduceMaxDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReduceMaxDescriptor(infiniopHandle_t handle,
                                                                infiniopReduceMaxDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t reduced,
                                                                infiniopTensorDescriptor_t data,
                                                                int64_t const *axes,
                                                                size_t axes_size,
                                                                int keepdims,
                                                                int noop_with_empty_axes) {
    infiniopReduceDescriptor_t reduce_desc;
    CHECK_STATUS(
        infiniopCreateReduceDescriptor(handle, &reduce_desc, reduced, data, axes, axes_size, keepdims, noop_with_empty_axes, 1),
        STATUS_SUCCESS);
    
    *(_ReduceMaxDescriptor_t *)desc_ptr = new _ReduceMaxDescriptor{handle->device, reduce_desc};

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopReduceMax(infiniopReduceMaxDescriptor_t desc,
                                                void *reduced,
                                                void const *data,
                                                int64_t const *axes,
                                                void *stream) {
    auto _desc = (_ReduceMaxDescriptor_t)desc;
    CHECK_STATUS(infiniopReduce(_desc->reduce_desc, reduced, data, axes, stream),
                 STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyReduceMaxDescriptor(infiniopReduceMaxDescriptor_t desc) {
    auto _desc = (_ReduceMaxDescriptor_t)desc;
    CHECK_STATUS(infiniopDestroyReduceDescriptor(_desc->reduce_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}
