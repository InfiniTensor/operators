#ifndef __CPU_WHERE_H__
#define __CPU_WHERE_H__

#include "operators.h"
struct WhereCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t const *src1_shape;
    uint64_t const *src2_shape;
    uint64_t const *condition_shape;
    uint64_t const *dst_shape;
    int64_t const *dst_strides;
    int64_t const *src1_strides;
    int64_t const *src2_strides;
    int64_t const *condition_strides;
    uint64_t dst_ndim;
    uint64_t src1_ndim;
    uint64_t src2_ndim;
    uint64_t condition_ndim;
    uint64_t element_num;
};

typedef struct WhereCpuDescriptor *WhereCpuDescriptor_t;

infiniopStatus_t cpuCreateWhereDescriptor(infiniopHandle_t handle,
                                        WhereCpuDescriptor_t *desc_ptr,
                                        infiniopTensorDescriptor_t dst,
                                        infiniopTensorDescriptor_t src1,
                                        infiniopTensorDescriptor_t src2,
                                        infiniopTensorDescriptor_t condition
                                        );

infiniopStatus_t cpuWhere(WhereCpuDescriptor_t desc,
                                  void *dst, 
                                  void const *src1,
                                  void const *src2,
                                  void const *condition,
                                  void *stream);

infiniopStatus_t cpuDestroyWhereDescriptor(WhereCpuDescriptor_t desc);

#endif
