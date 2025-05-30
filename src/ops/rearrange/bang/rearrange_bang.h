#ifndef __BANG_REARRANGE_H__
#define __BANG_REARRANGE_H__

#include "../../../devices/bang/bang_handle.h"
#include "operators.h"

struct RearrangeBangDescriptor {
    Device device;
    int device_id;
    DT dtype;
    uint64_t ndim;
    uint64_t dim_last, dim_2nd_last, dim_3rd_last;
    int64_t dst_stride_2nd_last, dst_stride_3rd_last,
        src_stride_2nd_last, src_stride_3rd_last;
};

typedef struct RearrangeBangDescriptor *RearrangeBangDescriptor_t;

infiniopStatus_t bangCreateRearrangeDescriptor(BangHandle_t handle,
                                               RearrangeBangDescriptor_t *desc_ptr,
                                               infiniopTensorDescriptor_t dst,
                                               infiniopTensorDescriptor_t src);

infiniopStatus_t bangRearrange(RearrangeBangDescriptor_t desc,
                               void *dst,
                               void const *src,
                               void *stream);

infiniopStatus_t bangDestroyRearrangeDescriptor(RearrangeBangDescriptor_t desc);


#endif// __BANG_REARRANGE_H__
