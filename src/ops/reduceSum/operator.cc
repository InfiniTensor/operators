#include "../reduce/reduce.h"
#include "../utils.h"
#include "ops/reduceSum/reduceSum.h"

struct _ReduceSumDescriptor {
    Device device;
    infiniopReduceDescriptor_t reduce_desc;
    uint64_t workspace_size;
};

typedef struct _ReduceSumDescriptor *_ReduceSumDescriptor_t;
__C __export infiniopStatus_t infiniopCreateReduceSumDescriptor(infiniopHandle_t handle,
                                                              infiniopReduceSumDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t dst,
                                                              infiniopTensorDescriptor_t src,
                                                              int* axis,
                                                              const int num_axis, 
                                                              int const keepdims) {
infiniopReduceDescriptor_t reduce_desc;
CHECK_STATUS(infiniopCreateReduceDescriptor(handle, &reduce_desc, dst, src, axis, num_axis, keepdims, 3), STATUS_SUCCESS);
uint64_t workspace_size = 0;
CHECK_STATUS(infiniopGetReduceWorkspaceSize(reduce_desc, &workspace_size), STATUS_SUCCESS);

*(_ReduceSumDescriptor_t *) desc_ptr = new _ReduceSumDescriptor{
handle->device,
reduce_desc,
workspace_size,
};

return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopGetReduceSumWorkspaceSize(infiniopReduceSumDescriptor_t desc, uint64_t *size) {
    *size = ((_ReduceSumDescriptor_t) desc)->workspace_size;
    return STATUS_SUCCESS;
}
__C __export infiniopStatus_t infiniopReduceSum(infiniopReduceSumDescriptor_t desc, void *workspace, uint64_t workspace_size, void *dst, void const *src, void *stream) {
auto _desc = (_ReduceSumDescriptor_t) desc;
if (workspace_size < _desc->workspace_size) {
    return STATUS_MEMORY_NOT_ALLOCATED;
}
CHECK_STATUS(infiniopReduce(_desc->reduce_desc, workspace, workspace_size, dst, src, stream),
STATUS_SUCCESS);
return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyReduceSumDescriptor(infiniopReduceSumDescriptor_t desc) {
CHECK_STATUS(infiniopDestroyReduceDescriptor(((_ReduceSumDescriptor_t) desc)->reduce_desc), STATUS_SUCCESS);
delete desc;
return STATUS_SUCCESS;
}
