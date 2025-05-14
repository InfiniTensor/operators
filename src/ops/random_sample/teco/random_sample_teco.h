#ifndef __SDAA_RANDOM_SAMPLE_H__
#define __SDAA_RANDOM_SAMPLE_H__

#include "../../../devices/teco/teco_handle.h"
#include "../../utils.h"
#include "operators.h"
#include <sdaa_runtime.h>

struct RandomSampleTecoDescriptor {
    Device device;
    int device_id;
    tecodnnHandle_t handle;
    sdaaStream_t stream;
    DT dtype;
    int voc;
    DT rDtype;
    int rLength;
};

typedef struct RandomSampleTecoDescriptor *RandomSampleTecoDescriptor_t;

infiniopStatus_t tecoCreateRandomSampleDescriptor(TecoHandle_t handle,
                                                  RandomSampleTecoDescriptor_t *desc_ptr, infiniopTensorDescriptor_t result,
                                                  infiniopTensorDescriptor_t probs);

infiniopStatus_t tecoGetRandomSampleWorkspaceSize(RandomSampleTecoDescriptor_t desc, uint64_t *size);

infiniopStatus_t tecoRandomSample(RandomSampleTecoDescriptor_t desc,
                                  void *workspace,
                                  uint64_t workspace_size,
                                  void *result,
                                  void const *probs,
                                  float random_val,
                                  float topp,
                                  int topk,
                                  float temperature,
                                  void *stream);

infiniopStatus_t tecoDestroyRandomSampleDescriptor(RandomSampleTecoDescriptor_t desc);


#endif
