#ifndef __CNNL_SOFTMAX_H__
#define __CNNL_SOFTMAX_H__
#include "../../../devices/bang/bang_handle.h"
#include "cnnl.h"
#include "cnnl_extra.h"
#include "operators.h"

struct SoftmaxCnnlDescriptor {
    Device device;
    int device_id;
    DT dtype;
    std::shared_ptr<Pool<cnnlHandle_t>> cnnl_handles;
    cnnlSoftmaxMode_t mode;
    cnnlTensorDescriptor_t aDesc;
    cnnlTensorDescriptor_t cDesc;
    float alpha;
    float beta;
};
typedef struct SoftmaxCnnlDescriptor *SoftmaxCnnlDescriptor_t;

infiniopStatus_t cnnlCreateSoftmaxDescriptor(BangHandle_t handle,
                                             SoftmaxCnnlDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t input_desc,
                                             int axis,
                                             infiniopTensorDescriptor_t output_desc);

infiniopStatus_t cnnlGetSoftmaxWorkspaceSize(SoftmaxCnnlDescriptor_t desc, unsigned long int *size);
infiniopStatus_t cnnlSoftmax(SoftmaxCnnlDescriptor_t desc, void *workspace,
                                          uint64_t workspace_size, void const *input, void *output, void *stream);

infiniopStatus_t cnnlDestroySoftmaxDescriptor(SoftmaxCnnlDescriptor_t desc);


#endif// __CNNL_SOFTMAX_H__
