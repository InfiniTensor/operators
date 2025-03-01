#ifndef __CPU_SOFTMAX_H__
#define __CPU_SOFTMAX_H__

#include "../../../devices/cpu/cpu_handle.h"
#include "../../utils.h"
#include "operators.h"

struct SoftmaxCpuDescriptor {
    Device device;
    DT dtype;
    int ndim;
    int axis;
    int *shape;
};

typedef struct SoftmaxCpuDescriptor *SoftmaxCpuDescriptor_t;

infiniopStatus_t cpuCreateSoftmaxDescriptor(infiniopHandle_t handle,
                                            SoftmaxCpuDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t input_desc, int axis, infiniopTensorDescriptor_t output_desc);

infiniopStatus_t cpuGetSoftmaxWorkspaceSize(SoftmaxCpuDescriptor_t desc, unsigned long int *size);

infiniopStatus_t cpuSoftmax(SoftmaxCpuDescriptor_t desc, void *workspace,
                                          uint64_t workspace_size,
                            void const *input,
                            void *output,
                            void *stream);

infiniopStatus_t cpuDestroySoftmaxDescriptor(SoftmaxCpuDescriptor_t desc);


#endif
