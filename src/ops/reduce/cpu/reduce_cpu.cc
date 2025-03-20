#include "reduce_cpu.h"
#include "../../utils.h"
#include <iostream>
infiniopStatus_t cpuCreateReduceDescriptor(infiniopHandle_t handle,
                                            ReduceCpuDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t y,
                                            infiniopTensorDescriptor_t x,
                                            int64_t const *axes,
                                            uint64_t n,
                                            int reduce_type,
                                            bool noop_with_empty_axes,
                                            bool keepdims) {    
    uint64_t ndim = y->ndim;
    uint64_t x_ndim = x->ndim;
    if (reduce_type > 2){
        return STATUS_BAD_PARAM;
    }
    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (!is_contiguous(y) || !is_contiguous(x)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    uint64_t *x_shape = new uint64_t[x_ndim];
    uint64_t *y_shape = new uint64_t[ndim];
    int64_t *x_strides = new int64_t[x_ndim];
    int64_t *y_strides = new int64_t[ndim]; 
    uint64_t y_size = 1;
    for (uint64_t i = 0; i < ndim; i++) {
        y_size *= y->shape[i];
    } 
    memcpy(y_shape, y->shape, ndim * sizeof(uint64_t));
    memcpy(y_strides, y->strides, ndim * sizeof(int64_t));
    memcpy(x_strides, x->strides, x_ndim * sizeof(int64_t));
    memcpy(x_shape, x->shape, x_ndim * sizeof(uint64_t));

    if (axes != nullptr && n > 0) {
        bool is_axes_static = true;
        for (uint64_t i = 0; i < n; i++) {
            if (axes[i] >= x->ndim) {
                return STATUS_BAD_PARAM; // 轴索引越界
            }
        }
        std::unordered_set<int64_t> axes_set;
        for (uint64_t i = 0; i < n; i++) {
            if (axes_set.find(axes[i]) != axes_set.end()) {
                return STATUS_BAD_PARAM; // 轴重复
            }
            axes_set.insert(axes[i]);
        }
        std::vector<int64_t> axes_vec(axes_set.begin(), axes_set.end());
        std::sort(axes_vec.begin(), axes_vec.end());
        int64_t* unique_axes = new int64_t[axes_vec.size()]; 
        std::copy(axes_vec.begin(), axes_vec.end(), unique_axes);
        uint64_t *reduce_axes_stride = new uint64_t[axes_vec.size()];
        for (int i = axes_vec.size() - 1; i >= 0; i--){
            reduce_axes_stride[i] = 1;
        }
        for (int i = axes_vec.size() - 2; i >= 0; i--){
            reduce_axes_stride[i] = reduce_axes_stride[i + 1] * x->shape[axes_vec[i + 1]];
        }
        int reduce_element_num = 1;
        for (auto axis: axes_vec) {
            reduce_element_num *= x->shape[axis];
        }
        *desc_ptr = new ReduceCpuDescriptor{
            DevCpu,
            y->dt,
            ndim,
            x_ndim,
            x_shape,
            unique_axes,
            x_strides,
            reduce_axes_stride,
            y_strides,
            y_size,
            reduce_type,
            reduce_element_num,
            n,
            is_axes_static,
            noop_with_empty_axes,
            true,
            keepdims
        };
    }
    else {
        *desc_ptr = new ReduceCpuDescriptor{
            DevCpu,
            y->dt,
            ndim,
            x_ndim,
            x_shape,
            nullptr,
            x_strides,
            nullptr,
            y_strides,
            y_size,
            reduce_type,
            1,
            0,
            false,
            noop_with_empty_axes,
            false,
            keepdims
        };
    }
    return STATUS_SUCCESS;
}   


infiniopStatus_t cpuDestroyReduceDescriptor(ReduceCpuDescriptor_t desc) {
    delete[] desc->x_shape;
    delete[] desc->axes;
    delete[] desc->x_strides;
    delete[] desc->reduce_axes_stride;
    delete desc;
    return STATUS_SUCCESS;
}
template<typename Tdata, bool keepdims>
infiniopStatus_t reduce_cpu(ReduceCpuDescriptor_t desc,
                            void *y,
                            void const *x){
    auto y_ = reinterpret_cast<Tdata *>(y);
    auto x_ = reinterpret_cast<Tdata const *>(x);
    auto input_strides = desc->x_strides;
    auto reduce_axes_stride = desc->reduce_axes_stride;
    auto num_axes = desc->axes_num; 
    auto axes = desc->axes;
    auto ndim = desc->ndim;
    auto x_shape = desc->x_shape; 
    auto y_strides = desc->y_strides;
    int i, j;
    int indices[desc->x_ndim];
    std::vector<int> non_reduce_axes;
    for (int j = 0; j < desc->x_ndim; ++j) {
        if (!std::binary_search(axes, axes + num_axes, j)) {
            non_reduce_axes.push_back(j);
        }
    }
    for (i = 0; i < desc->y_size; i++) {
        float sum_value = 0;
        std::vector<int> indices(desc->x_ndim, 0);
        int global_index = i;
        if (!non_reduce_axes.empty()) {
            if constexpr (keepdims){
                for (int j = 0; j < non_reduce_axes.size(); j++) {
                    indices[non_reduce_axes[j]] = global_index / y_strides[non_reduce_axes[j]];
                    global_index %= y_strides[non_reduce_axes[j]];
                }
            }else{
                for (int j = 0; j < non_reduce_axes.size(); j++) {
                    indices[non_reduce_axes[j]] = global_index / y_strides[j];
                    global_index %= y_strides[j];
                }
            }
        }
        int64_t base_offset = 0;
        for (int j = 0; j < desc->x_ndim; ++j) {
            base_offset += indices[j] * desc->x_strides[j];
        }
        for (j = 0; j < desc->reduce_element_num; j++){
            int64_t offset = base_offset;
            uint64_t remaining = j;
            int k;
            for (k = 0; k < num_axes; k++){
                const int axis = axes[k];
                const uint64_t coord = remaining / reduce_axes_stride[k];
                offset += coord * input_strides[axis];
                remaining %= reduce_axes_stride[k]; 
            }
            switch(desc->reduce_mode){
                case 0:
                    if constexpr (std::is_same<Tdata, uint16_t>::value){
                        sum_value += f16_to_f32(x_[offset]);
                    }
                    else{
                        sum_value += x_[offset];
                    }
                    break;
                case 1:
                    if constexpr (std::is_same<Tdata, uint16_t>::value){
                        y_[i] = f32_to_f16(std::fmax(f16_to_f32(y_[i]), f16_to_f32(x_[offset])));
                    }
                    else{
                        y_[i] = std::max(y_[i], x_[offset]);
                    }
                    break;
                case 2:
                    if constexpr (std::is_same<Tdata, uint16_t>::value){
                        y_[i] = f32_to_f16(std::fmin(f16_to_f32(y_[i]), f16_to_f32(x_[offset])));
                    }
                    else{
                        y_[i] = std::min(y_[i], x_[offset]);
                    }
                    break;
            }
        }
        if (desc->reduce_mode == 0){
            sum_value /= desc->reduce_element_num;
            if constexpr (std::is_same<Tdata, uint16_t>::value){
                y_[i] = f32_to_f16(sum_value);
            }
            else{
                y_[i] = sum_value;
            }
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuReduce(ReduceCpuDescriptor_t desc,
                            void *y,
                            void const *x,
                            void *dynamic_axes,
                            uint64_t dynamic_axes_size,
                            void *stream){
    if (desc->is_axes_static == true && dynamic_axes_size > 0){
        return STATUS_BAD_PARAM;
    }
    if (desc->is_axes_static == false && dynamic_axes_size == 0){
        if (desc->noop_with_empty_axes){
            if (desc->dt == F16){
                memcpy(y, x, desc->y_size * sizeof(uint16_t));
                return STATUS_SUCCESS;
            }
            else if (desc->dt == F32){
                memcpy(y, x, desc->y_size * sizeof(float));
                return STATUS_SUCCESS;
            }
            else return STATUS_BAD_TENSOR_DTYPE;
        }
        delete[] desc->axes;
        delete[] desc->reduce_axes_stride;
        desc->axes = new int64_t[desc->x_ndim];
        desc->reduce_axes_stride = new uint64_t[desc->x_ndim];
        std::vector<int64_t> full_axes(desc->x_ndim);
        std::iota(full_axes.begin(), full_axes.end(), 0);
        std::copy(full_axes.begin(), full_axes.end(), desc->axes);
        for (int i = desc->x_ndim - 1; i >= 0; i--){
            desc->reduce_axes_stride[i] = 1;
        }
        for (int i = desc->x_ndim - 2; i >= 0; i--){
            desc->reduce_axes_stride[i] = desc->reduce_axes_stride[i + 1] * desc->x_shape[full_axes[i + 1]];
        }

        desc->axes_num = desc->x_ndim;
        desc->reduce_element_num = 1;
        for (int i = 0; i < desc->x_ndim; i++){
            desc->reduce_element_num *= desc->x_shape[i];
        }
        desc->owns_axes_memory = true;
    }
    if (desc->is_axes_static == false && dynamic_axes_size > 0){
        auto dynamic_axes_data = reinterpret_cast<int64_t const *>(dynamic_axes);

        for (uint64_t i = 0; i < dynamic_axes_size; i++){
            if (dynamic_axes_data[i] >= desc->x_ndim){
                return STATUS_BAD_PARAM;
            }
        }
        std::unordered_set<int64_t> axes_set;
        for (uint64_t i = 0; i < dynamic_axes_size; i++){
            if (axes_set.find(dynamic_axes_data[i]) != axes_set.end()){
                return STATUS_BAD_PARAM;
            }
            axes_set.insert(dynamic_axes_data[i]);
        }
        std::vector<int64_t> axes_vec(axes_set.begin(), axes_set.end());
        std::sort(axes_vec.begin(), axes_vec.end());

        delete[] desc->axes;
        delete[] desc->reduce_axes_stride;
        desc->axes = new int64_t[axes_vec.size()];
        desc->reduce_axes_stride = new uint64_t[axes_vec.size()];
        desc->axes_num = axes_vec.size();
        std::copy(axes_vec.begin(), axes_vec.end(), desc->axes);
        for (int i = axes_vec.size() - 1; i >= 0; i--){
            desc->reduce_axes_stride[i] = 1;
        }
        for (int i = axes_vec.size() - 2; i >= 0; i--){
            desc->reduce_axes_stride[i] = desc->reduce_axes_stride[i + 1] * desc->x_shape[axes_vec[i + 1]];
        }
        for (auto axis: axes_vec){
            desc->reduce_element_num *= desc->x_shape[axis];
        }
        desc->owns_axes_memory = true;
    }
    if (desc->dt == F16) {
        if (desc->keepdims == true) {
            return reduce_cpu<uint16_t, true>(desc, y, x);
        }
        return reduce_cpu<uint16_t, false>(desc, y, x);
    }
    if (desc->dt == F32) {
        if (desc->keepdims == true) {
            return reduce_cpu<float, true>(desc, y, x);
        }
        return reduce_cpu<float, false>(desc, y, x);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}