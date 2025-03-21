#include "../reduce/reduce.h"
#include "../utils.h"
#include "ops/reduce_max/reduce_max.h"

struct _ReduceMaxDescriptor {
    Device device;
    infiniopReduceDescriptor_t reduce_desc;
    uint64_t workspace_size;
};

typedef struct _ReduceMaxDescriptor *_ReduceMaxDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReduceMaxDescriptor(infiniopHandle_t handle,
                                                                infiniopReduceMaxDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t reduced,
                                                                infiniopTensorDescriptor_t data,
                                                                int64_t *axes,
                                                                uint64_t axes_ndim,
                                                                bool keepdims,
                                                                bool noop_with_empty_axes) {
    infiniopReduceDescriptor_t reduce_desc;
    // case 0: { // ReduceMin
    // case 1: { // ReduceMax
    // case 2: { // ReduceMean
    CHECK_STATUS(infiniopCreateReduceDescriptor(handle, &reduce_desc, reduced, data, axes, axes_ndim,
                                                keepdims, noop_with_empty_axes, 1), STATUS_SUCCESS);

    uint64_t workspace_size = 0;
    CHECK_STATUS(infiniopGetReduceWorkspaceSize(reduce_desc, &workspace_size), STATUS_SUCCESS);

    *(_ReduceMaxDescriptor_t *) desc_ptr = new _ReduceMaxDescriptor{
        handle->device,
        reduce_desc,
        workspace_size};

    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopGetReduceMaxWorkspaceSize(infiniopReduceMaxDescriptor_t desc, uint64_t *size) {
    *size = ((_ReduceMaxDescriptor_t) desc)->workspace_size;
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopReduceMax(infiniopReduceMaxDescriptor_t desc, void *workspace, uint64_t workspace_size,
                                                void *reduced, void const *data, void *stream) {
    auto _desc = (_ReduceMaxDescriptor_t) desc;
    if (workspace_size < _desc->workspace_size) {
        return STATUS_MEMORY_NOT_ALLOCATED;
    }

    CHECK_STATUS(infiniopReduce(_desc->reduce_desc, reduced, data, stream),
                 STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyReduceMaxDescriptor(infiniopReduceMaxDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyReduceDescriptor(((_ReduceMaxDescriptor_t) desc)->reduce_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}
