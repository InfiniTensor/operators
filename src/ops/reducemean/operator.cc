#include "../utils.h"
#include "../reduce/reduce.h"
#include "ops/reducemean/reducemean.h"

struct _ReducemeanDescriptor {
    Device device;
    infiniopReduceDescriptor_t reduce_desc;
};

typedef struct _ReducemeanDescriptor *_ReducemeanDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReducemeanDescriptor(
    infiniopHandle_t handle,
    infiniopReducemeanDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    int64_t const *axes,
    uint64_t n,
    bool keepdims,
    bool noop_with_empty_axes
    ) {
    infiniopReduceDescriptor_t reduce_desc;
    CHECK_STATUS(infiniopCreateReduceDescriptor(handle, &reduce_desc, y, x, axes, n, keepdims, noop_with_empty_axes, 0), STATUS_SUCCESS);
    *(_ReducemeanDescriptor_t *) desc_ptr = new _ReducemeanDescriptor{
        handle->device,
        reduce_desc
    };
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopReducemean(infiniopReducemeanDescriptor_t desc, void *y, void const *x, void *dynamic_axes, uint64_t dynamic_axes_size, void *stream) {
    auto _desc = (_ReducemeanDescriptor_t) desc;
    CHECK_STATUS(infiniopReduce(_desc->reduce_desc, y, x, dynamic_axes, dynamic_axes_size, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyReducemeanDescriptor(infiniopReducemeanDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyReduceDescriptor(((_ReducemeanDescriptor_t) desc)->reduce_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}