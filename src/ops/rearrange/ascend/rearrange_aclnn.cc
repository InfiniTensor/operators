#include "rearrange_aclnn.h"
#include "../../utils.h"

RearrangeAclnnDescriptor::RearrangeAclnnDescriptor(Device _device) {
    device = _device;
    device_id = 0;
    executor = nullptr;
    dstDesc = new aclnnTensorDescriptor();
    srcDesc = new aclnnTensorDescriptor();
    workspaceSize = 0;
    workspaceAddr = nullptr;
}

infiniopStatus_t aclnnCreateRearrangeDescriptor(AscendHandle_t handle,
                                                RearrangeAclnnDescriptor_t *desc_ptr,
                                                infiniopTensorDescriptor_t dst,
                                                infiniopTensorDescriptor_t src) {
    *desc_ptr = new RearrangeAclnnDescriptor(handle->device);
    (*desc_ptr)->device_id = handle->device_id;

    auto &dstDesc = (*desc_ptr)->dstDesc;
    auto &srcDesc = (*desc_ptr)->srcDesc;

    CHECK_STATUS(dstDesc->fromInfiniOpTensorDescriptor(dst), STATUS_SUCCESS);
    CHECK_STATUS(srcDesc->fromInfiniOpTensorDescriptor(src), STATUS_SUCCESS);

    // CHECK_STATUS(dstDesc->createTensor(), STATUS_SUCCESS);
    // CHECK_STATUS(srcDesc->createTensor(), STATUS_SUCCESS);

    // aclTensor *td = dstDesc->t;
    // aclTensor *ts = srcDesc->t;

    // auto &workspaceSize = (*desc_ptr)->workspaceSize;
    // auto &executor = (*desc_ptr)->executor;

    // auto ret = aclnnInplaceCopyGetWorkspaceSize(td,
    //                                             ts,
    //                                             &workspaceSize,
    //                                             &executor);
    // aclSetAclOpExecutorRepeatable(executor);
    // CHECK_RET(ret == ACL_SUCCESS,
    //           LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret);
    //           return STATUS_EXECUTION_FAILED);

    // (*desc_ptr)->workspaceAddr = mallocWorkspace(workspaceSize);

    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnRearrange(RearrangeAclnnDescriptor_t desc,
                                void *dst,
                                void const *src,
                                void *stream) {
    // Set runing on handle device
    // aclrtSetDevice(desc->device_id);

    /// TODO: something is wrong with aclSetTensorAddr, do all the preparation here for now
    desc->dstDesc->t = aclCreateTensor(desc->dstDesc->shape.data(),
                                       desc->dstDesc->ndim,
                                       desc->dstDesc->dataType,
                                       desc->dstDesc->strides.data(),
                                       desc->dstDesc->offset,
                                       desc->dstDesc->format,
                                       desc->dstDesc->storageShape.data(),
                                       desc->dstDesc->storageNdim,
                                       dst);
    desc->srcDesc->t = aclCreateTensor(desc->srcDesc->shape.data(),
                                       desc->srcDesc->ndim,
                                       desc->srcDesc->dataType,
                                       desc->srcDesc->strides.data(),
                                       desc->srcDesc->offset,
                                       desc->srcDesc->format,
                                       desc->srcDesc->storageShape.data(),
                                       desc->srcDesc->storageNdim,
                                       (void *) src);

    aclTensor *td = desc->dstDesc->t;
    aclTensor *ts = desc->srcDesc->t;
    aclOpExecutor *executor;
    uint64_t workspaceSize;
    aclnnInplaceCopyGetWorkspaceSize(td,
                                     ts,
                                     &workspaceSize,
                                     &executor);
    CHECK_STATUS(mallocWorkspace(&(desc->workspaceAddr), workspaceSize), STATUS_SUCCESS);


    // AclSetTensorAddr(executor, 0, td, dst);
    // AclSetTensorAddr(executor, 1, ts, (void *) src);
    auto ret = aclnnInplaceCopy(desc->workspaceAddr,
                                desc->workspaceSize,
                                executor,
                                stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret);
              return STATUS_EXECUTION_FAILED);

    desc->dstDesc->destroyTensor();
    desc->srcDesc->destroyTensor();
    CHECK_STATUS(freeWorkspace(desc->workspaceAddr), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnDestroyRearrangeDescriptor(RearrangeAclnnDescriptor_t desc) {
    delete desc->srcDesc;
    delete desc->dstDesc;
    /// TODO: this aclDestroyAclOpExecutor seems to trigger a double free error
    // aclDestroyAclOpExecutor(desc->executor);
    // freeWorkspace(desc->workspaceAddr);
    delete desc;

    return STATUS_SUCCESS;
}
