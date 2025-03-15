#include "../reduce/reduce.h"
#include "../utils.h"
#include "ops/reduce_mean/reduce_mean.h"

struct _ReduceMeanDescriptor {
    Device device;
    infiniopReduceDescriptor_t reduce_desc;
};

typedef struct _ReduceMeanDescriptor *_ReduceMeanDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReduceMeanDescriptor(infiniopHandle_t handle,
                                                                infiniopReduceMeanDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t reduced,
                                                                infiniopTensorDescriptor_t data,
                                                                int64_t const *axes,
                                                                size_t axes_size,
                                                                int keepdims,
                                                                int noop_with_empty_axes) {
    infiniopReduceDescriptor_t reduce_desc;
    CHECK_STATUS(
        infiniopCreateReduceDescriptor(handle, &reduce_desc, reduced, data, axes, axes_size, keepdims, noop_with_empty_axes, 3),
        STATUS_SUCCESS);

    *(_ReduceMeanDescriptor_t *)desc_ptr = new _ReduceMeanDescriptor{handle->device, reduce_desc};

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopReduceMean(infiniopReduceMeanDescriptor_t desc,
                                                void *reduced,
                                                void const *data,
                                                int64_t const *axes,
                                                void *stream) {
    auto _desc = (_ReduceMeanDescriptor_t)desc;
    CHECK_STATUS(infiniopReduce(_desc->reduce_desc, reduced, data, axes, stream),
                 STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyReduceMeanDescriptor(infiniopReduceMeanDescriptor_t desc) {
    auto _desc = (_ReduceMeanDescriptor_t)desc;
    CHECK_STATUS(infiniopDestroyReduceDescriptor(_desc->reduce_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}
