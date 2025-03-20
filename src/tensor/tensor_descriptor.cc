#include "tensor/tensor_descriptor.h"
#include <cstring>


__C __export infiniopStatus_t infiniopCreateTensorDescriptor(infiniopTensorDescriptor_t *desc_ptr, uint64_t ndim, uint64_t const *shape_, int64_t const *strides_, DataLayout datatype) {
    uint64_t *shape = new uint64_t[ndim];
    int64_t *strides = new int64_t[ndim];
    std::memcpy(shape, shape_, ndim * sizeof(uint64_t));
    if (strides_) {
        std::memcpy(strides, strides_, ndim * sizeof(int64_t));
    } else {
        int64_t dsize = 1;
        for (int i = ndim - 1; i >= 0; i--) {
            strides[i] = dsize;
            dsize *= shape[i];
        }
    }
    *desc_ptr = new TensorDescriptor{datatype, ndim, shape, strides};
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyTensorDescriptor(infiniopTensorDescriptor_t desc) {
    delete[] desc->shape;
    delete[] desc->strides;
    delete desc;
    return STATUS_SUCCESS;
}
