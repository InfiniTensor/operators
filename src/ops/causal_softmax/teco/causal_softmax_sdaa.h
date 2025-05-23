#ifndef __SDAA_CAUSAL_SOFTMAX_H__
#define __SDAA_CAUSAL_SOFTMAX_H__
#include "../../../devices/teco/teco_handle.h"
#include "../../utils.h"
#include "operators.h"
#include <sdaa_runtime.h>
struct CausalSoftmaxTecoDescriptor {
    Device device;
    int device_id;
    DT dtype;
    int ndim;
    int *stride;
    int *shape;
};

typedef struct CausalSoftmaxTecoDescriptor *CausalSoftmaxTecoDescriptor_t;


infiniopStatus_t tecoCreateCausalSoftmaxDescriptor(TecoHandle_t handle,
                                                   CausalSoftmaxTecoDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t y_desc);

infiniopStatus_t tecoGetCausalSoftmaxWorkspaceSize(CausalSoftmaxTecoDescriptor_t desc, uint64_t *size);

infiniopStatus_t tecoCausalSoftmax(CausalSoftmaxTecoDescriptor_t desc,
                                   void *workspace,
                                   uint64_t workspace_size,
                                   void *data,
                                   void *stream);

infiniopStatus_t tecoDestroyCausalSoftmaxDescriptor(CausalSoftmaxTecoDescriptor_t desc);


#endif
