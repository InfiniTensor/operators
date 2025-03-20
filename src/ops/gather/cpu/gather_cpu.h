#ifndef __CPU_GATHER_H__
#define __CPU_GATHER_H__

#include "operators.h"
struct GatherCpuDescriptor {
    Device device;
    DT dtype;
    DT indices_dtype;
    uint64_t const *dst_shape;
    uint64_t const *src_shape;
    uint64_t const *indices_shape;
    uint64_t indices_ndim;
    uint64_t src_ndim;
    uint64_t dst_ndim;
    int64_t axis;
};

typedef struct GatherCpuDescriptor *GatherCpuDescriptor_t;

infiniopStatus_t cpuCreateGatherDescriptor(infiniopHandle_t handle,
                                        GatherCpuDescriptor_t *desc_ptr,
                                        infiniopTensorDescriptor_t y,
                                        infiniopTensorDescriptor_t x,
                                        infiniopTensorDescriptor_t indices,
                                        int64_t axis
                                        );

infiniopStatus_t cpuGather(GatherCpuDescriptor_t desc,
                                  void const *data, 
                                  void const *indices,
                                  void *dst,
                                  void *stream);

infiniopStatus_t cpuDestroyGatherDescriptor(GatherCpuDescriptor_t desc);

#endif