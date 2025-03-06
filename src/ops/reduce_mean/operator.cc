#include "../reduce/reduce.h"
#include "../utils.h"
#include "export.h"
#include "ops/reduce_mean/reduce_mean.h"

struct _ReduceMeanDescriptor {
    Device device;
    infiniopReduceDescriptor_t reduce_desc;
};

typedef struct _ReduceMeanDescriptor *_ReduceMeanDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReduceMeanDescriptor(infiniopHandle_t handle,
                                                                 infiniopReduceMeanDescriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t y,
                                                                 infiniopTensorDescriptor_t x,
                                                                 int64_t const *axes,
                                                                 uint64_t n_axes,
                                                                 int keep_dims) {
    infiniopReduceDescriptor_t reduce_desc;
    // reduce_mean: 2
    CHECK_STATUS(infiniopCreateReduceDescriptor(handle, &reduce_desc, y, x, axes, n_axes, keep_dims, 2), STATUS_SUCCESS);

    *(_ReduceMeanDescriptor_t *) desc_ptr = new _ReduceMeanDescriptor{
        handle->device,
        reduce_desc};

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopReduceMean(infiniopReduceMeanDescriptor_t desc, void *y, void const *x, void *stream) {
    auto _desc = (_ReduceMeanDescriptor_t) desc;
    CHECK_STATUS(infiniopReduce(_desc->reduce_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyReduceMeanDescriptor(infiniopReduceMeanDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyReduceDescriptor(((_ReduceMeanDescriptor_t) desc)->reduce_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}
