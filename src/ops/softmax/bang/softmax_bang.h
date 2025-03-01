#ifndef __BANG_SOFTMAX_H__
#define __BANG_SOFTMAX_H__

#include "../../../devices/bang/bang_handle.h"
#include "../../utils.h"
#include "operators.h"

struct SoftmaxBangDescriptor {
    Device device;
    int device_id;
    DT dtype;
    int ndim;
    int axis;
    int dimsize;
    int stride;
    int othersize;
    int frontsize;
};

typedef struct SoftmaxBangDescriptor *SoftmaxBangDescriptor_t;

infiniopStatus_t bangCreateSoftmaxDescriptor(BangHandle_t handle,
                                             SoftmaxBangDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t input_desc, int axis, infiniopTensorDescriptor_t output_desc);

infiniopStatus_t bangGetSoftmaxWorkspaceSize(SoftmaxBangDescriptor_t desc, unsigned long int *size);
infiniopStatus_t bangSoftmax(SoftmaxBangDescriptor_t desc, void *workspace,
                                          uint64_t workspace_size,
                             void const *input,
                             void *output,
                             void *stream);

infiniopStatus_t bangDestroySoftmaxDescriptor(SoftmaxBangDescriptor_t desc);


#endif