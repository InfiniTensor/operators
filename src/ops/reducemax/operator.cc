#include "../utils.h"
#include "../reduce/reduce.h"
#include "ops/reducemax/reducemax.h"

struct _ReducemaxDescriptor {
    Device device;
    infiniopReduceDescriptor_t reduce_desc;
};

typedef struct _ReducemaxDescriptor *_ReducemaxDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReducemaxDescriptor(
    infiniopHandle_t handle,
    infiniopReducemaxDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    int64_t const *axes,
    uint64_t n,
    bool keepdims,
    bool noop_with_empty_axes
    ) {
    infiniopReduceDescriptor_t reduce_desc;
    CHECK_STATUS(infiniopCreateReduceDescriptor(handle, &reduce_desc, y, x, axes, n, keepdims, noop_with_empty_axes, 1), STATUS_SUCCESS);
    *(_ReducemaxDescriptor_t *) desc_ptr = new _ReducemaxDescriptor{
        handle->device,
        reduce_desc
    };
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopReducemax(infiniopReducemaxDescriptor_t desc, void *y, const void *x, void *dynamic_axes, uint64_t dynamic_axes_size, void *stream) {
    auto _desc = (_ReducemaxDescriptor_t) desc;
    CHECK_STATUS(infiniopReduce(_desc->reduce_desc, y, x, dynamic_axes, dynamic_axes_size, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyReducemaxDescriptor(infiniopReducemaxDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyReduceDescriptor(((_ReducemaxDescriptor_t) desc)->reduce_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}