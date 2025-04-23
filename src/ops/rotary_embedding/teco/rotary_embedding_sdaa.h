#ifndef __SDAA_ROPE_H__
#define __SDAA_ROPE_H__
#include "../../../devices/teco/teco_handle.h"
#include "../../utils.h"
#include "operators.h"
#include <sdaa_runtime.h>
struct RoPETecoDescriptor {
    Device device;
    int device_id;
    DT dtype;
    uint64_t seqlen;
    uint64_t nhead;
    uint64_t dhead;
    uint64_t total_seqlen;
    int x_stride_seqlen;
    int x_stride_nhead;
};

typedef struct RoPETecoDescriptor *RoPETecoDescriptor_t;


infiniopStatus_t tecoCreateRoPEDescriptor(TecoHandle_t handle,
                                          RoPETecoDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t t,
                                          infiniopTensorDescriptor_t pos_ids,
                                          infiniopTensorDescriptor_t sin_table,
                                          infiniopTensorDescriptor_t cos_table);

infiniopStatus_t tecoGetRoPEWorkspaceSize(RoPETecoDescriptor_t desc, uint64_t *size);

infiniopStatus_t tecoRoPE(RoPETecoDescriptor_t desc,
                          void *workspace,
                          uint64_t workspace_size,
                          void *t,
                          void const *pos_ids,
                          void const *sin_table,
                          void const *cos_table,
                          void *stream);

infiniopStatus_t tecoDestroyRoPEDescriptor(RoPETecoDescriptor_t desc);


#endif
