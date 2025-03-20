#include "../utils.h"
#include "../reduce/reduce.h"
#include "ops/reducemin/reducemin.h"

struct _ReduceminDescriptor {
    Device device;
    infiniopReduceDescriptor_t reduce_desc;
};

typedef struct _ReduceminDescriptor *_ReduceminDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReduceminDescriptor(
    infiniopHandle_t handle,
    infiniopReduceminDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    int64_t const *axes,
    uint64_t n,
    bool keepdims,
    bool noop_with_empty_axes
    ) {
    infiniopReduceDescriptor_t reduce_desc;
    CHECK_STATUS(infiniopCreateReduceDescriptor(handle, &reduce_desc, y, x, axes, n, keepdims, noop_with_empty_axes,2), STATUS_SUCCESS);
    *(_ReduceminDescriptor_t *) desc_ptr = new _ReduceminDescriptor{
        handle->device,
        reduce_desc
    };
    return STATUS_SUCCESS;
}
__C __export infiniopStatus_t infiniopReducemin(infiniopReduceminDescriptor_t desc, void *y, void const *x, void *dynamic_axes, uint64_t dynamic_axes_size, void *stream) {
    auto _desc = (_ReduceminDescriptor_t) desc;
    CHECK_STATUS(infiniopReduce(_desc->reduce_desc, y, x, dynamic_axes, dynamic_axes_size, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyReduceminDescriptor(infiniopReduceminDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyReduceDescriptor(((_ReduceminDescriptor_t) desc)->reduce_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}