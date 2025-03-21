#include "reduce_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <cassert>

inline void incrementOne(uint64_t *indices, uint64_t const *shape, uint64_t ndim) {
    for (int64_t i = ndim - 1; i >= 0; --i) {
        if (++indices[i] != shape[i]) {
            return;
        }
        indices[i] = 0;
    }
}

inline uint64_t compactToFlat(uint64_t const *indices, uint64_t const *strides, uint64_t ndim) {
    return std::inner_product(indices, indices + ndim, strides, uint64_t(0));
}

infiniopStatus_t cpuCreateReduceDescriptor(infiniopHandle_t handle,
                                           ReduceCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t reduced,
                                           infiniopTensorDescriptor_t data,
                                           int64_t *axes,
                                           uint64_t axes_ndim,
                                           bool keepdims,
                                           bool noop_with_empty_axes,
                                           int reduce_type) {
    if (!is_contiguous(reduced) || !is_contiguous(data)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (reduce_type > 2){
        return STATUS_BAD_PARAM;
    }
    if (reduced->dt != data->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (reduced->dt != F16 && reduced->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    uint64_t data_ndim = data->ndim;
    uint64_t reduced_ndim = data_ndim - axes_ndim;

    uint64_t data_size = std::accumulate(data->shape, data->shape + data->ndim, 1ULL, std::multiplies<uint64_t>());
    // 
    uint64_t axes_size = 1;
    uint64_t reduced_size = 1;
    uint64_t reduced_size_ = std::accumulate(reduced->shape, reduced->shape + reduced->ndim, 1ULL, std::multiplies<uint64_t>());

    int64_t *reduced_axes;
    int64_t *axes_axes;
    
    int64_t *data_strides = new int64_t[data_ndim];
    memcpy(data_strides, data->strides, data_ndim * sizeof(int64_t));
    int64_t *reduced_strides;
    int64_t *axes_strides;
    
    // axes is empty
    if (axes_ndim == 0) {
        if (noop_with_empty_axes == false) {
            // reduce all axes
            axes_ndim = data_ndim;
            axes_size = data_size;
            axes_axes = new int64_t[axes_ndim];
            for (int i = 0; i < axes_ndim; i ++) {
                axes_axes[i] = i;
            }
            axes_strides = new int64_t[axes_ndim];
            memcpy(axes_strides, data_strides, axes_ndim * sizeof(int64_t));

            reduced_ndim = 1;
            reduced_size = 1;
            reduced_axes = nullptr;
            reduced_strides = new int64_t[reduced_ndim];
            reduced_strides[0] = 1;
        } else {
            // input tensor will not be reduced
            axes_ndim = 0;
            axes_size = 0;
            axes_axes = nullptr;
            axes_strides = new int64_t[axes_ndim];

            reduced_ndim = data_ndim;
            reduced_size = data_size;
            reduced_axes = new int64_t[reduced_ndim];
            for (int i = 0; i < reduced_ndim; i ++) {
                reduced_axes[i] = i;
            }
            reduced_strides = new int64_t[reduced_ndim];
            memcpy(reduced_strides, data_strides, reduced_ndim * sizeof(int64_t));
        }
    } else {
        // if (axes != nullptr && axes_ndim > 0) 
        if (keepdims == true) {
            // 保持
            // reduced_ndim = reduced->ndim;
        } else {
            // 减少张量维度
            reduced_ndim = data_ndim - axes_ndim;
        }
        // reduced_shape = new uint64_t[reduced_ndim];

        axes_size = 1;
        reduced_size = 1;
        std::vector<int> axes_vec, reduced_axes_vec;
        for(uint64_t i = 0; i < data_ndim; i++) {
            bool is_axis = false;
            for(uint64_t j = 0; j < axes_ndim; j++) {
                // printf("\t\tdata-dim %lu, axes-dim %lu, axes is %lu\n", i, j, axes[j]);
                if (axes[j] == i) {
                    is_axis = true;
                    break;
                }
            }
            
            if (is_axis) {
                axes_size *= data->shape[i];
                axes_vec.emplace_back(i);
                if (keepdims == true) {
                    // reduced_axes_vec.emplace_back(i);
                } 
            } else {
                reduced_size *= data->shape[i];
                // printf("\t\tdim %lu, data_shape is %lu, reduced_size now is %lu\n", i, data->shape[i], reduced_size);
                reduced_axes_vec.emplace_back(i);
            }
        }
        // ?
        // sort(axes_.begin(), axes_.end());
        // sort(reduced_axes_.begin(), reduced_axes_.end());
        axes_axes = new int64_t[axes_ndim];
        for (int i = 0; i < axes_ndim; i ++) {
            axes_axes[i] = axes_vec[i];
        }
        reduced_axes = new int64_t[reduced_ndim];
        for (int i = 0; i < reduced_ndim; i ++) {
            reduced_axes[i] = reduced_axes_vec[i];
        }

        // 
        axes_strides = new int64_t[axes_ndim];
        for(uint64_t i = 0; i < axes_ndim; i++) {
            axes_strides[i] = 1;
            for(uint64_t j = i + 1; j < axes_ndim; j++) {
                axes_strides[i] *= data->shape[axes[j]];
            }
        }

        // 
        reduced_strides = new int64_t[reduced_ndim];
        for(uint64_t i = 0; i < reduced_ndim; i++) {
            reduced_strides[i] = 1;
            for (uint64_t j = i + 1; j < reduced_ndim; j++) {
                uint64_t shape = data->shape[reduced_axes_vec[j]];
                if (keepdims == true) {
                    // shape = 1;
                } 
                reduced_strides[i] *= shape;
            }
        }
    }
    // printf("\tdata_ndim: %lu, reduced_ndim: %lu\n", data_ndim, reduced_ndim);
    assert(reduced_size == reduced_size_);
    
    *desc_ptr = new ReduceCpuDescriptor{
        DevCpu,
        reduced->dt,

        reduced_ndim,
        data_ndim,
        axes_ndim,

        reduced_size,
        data_size,
        axes_size,
        
        reduced_axes,
        axes_axes,

        reduced_strides,
        data_strides,
        axes_strides,

        keepdims,
        reduce_type
    };
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyReduceDescriptor(ReduceCpuDescriptor_t desc) {
    // printf("\t--cpuDestroyReduceDescriptor\n");
    // for (int i = 0; i < desc->reduced_ndim; i ++)
        // printf("\treduced_axes[%d]=%lu\n", i, desc->reduced_axes[i]);

    delete[] (desc->reduced_axes);
    delete[] (desc->axes);
    delete[] (desc->reduced_strides);
    delete[] (desc->data_strides);
    delete[] (desc->axes_strides);
    delete desc;
    // printf("\t--delete desc\n");
    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t reduce_cpu(ReduceCpuDescriptor_t desc,
                            void *reduced,
                            void const *data) {
    auto reduced_ = reinterpret_cast<Tdata *>(reduced);
    auto data_ = reinterpret_cast<Tdata const *>(data);

    auto reduced_ndim_ = desc->reduced_ndim;
    auto data_ndim_ = desc->data_ndim;
    auto axes_ndim_ = desc->axes_ndim;
    
    auto reduced_size_ = desc->reduced_size;
    auto data_size_ = desc->data_size;
    auto axes_size_ = desc->axes_size;
    // printf("\tdata_size: %lu, reduced_size: %lu, axes_size: %lu\n", data_size_, reduced_size_, axes_size_);

    auto reduced_axes_ = desc->reduced_axes;
    auto axes_ = desc->axes;

    auto reduced_strides_ = desc->reduced_strides;
    auto data_strides_ = desc->data_strides;
    auto axes_strides_ = desc->axes_strides;

    auto reduce_mode_ = desc->reduce_mode;
    
    // input tensor will not be reduced
    if (axes_ == nullptr) {
        for (uint64_t i = 0; i < data_size_; i++) {
            reduced_[i] = data_[i];
        }
        return STATUS_SUCCESS;
    }

    // reduce all axes
    if (reduced_axes_ == nullptr) {
        float result;
        // intitialzation
        switch (reduce_mode_) {
            case 0: { // ReduceMin
                result = std::numeric_limits<float>::max();
            }
            break;
            case 1: { // ReduceMax
                result = std::numeric_limits<float>::lowest();
            }
            break;
            case 2: { // ReduceMean
                result = 0;
            }
            break;
        }
        for (int i = 0; i < data_size_; i++) {
            if constexpr (std::is_same<Tdata, uint16_t>::value) {
                switch (reduce_mode_) {
                    case 0: { // ReduceMin
                        result = std::min(result, f16_to_f32(data_[i]));
                    }
                    break;
                    case 1: { // ReduceMax
                        result = std::max(result, f16_to_f32(data_[i]));
                    }
                    break;
                    case 2: { // ReduceMean
                        result += f16_to_f32(data_[i]);
                    }
                    break;
                }
            } else {
                switch (reduce_mode_) {
                    case 0: { // ReduceMin
                        result = std::min(result, data_[i]);
                    }
                    break;
                    case 1: { // ReduceMax
                        result = std::max(result, data_[i]);
                    }
                    break;
                    case 2: { // ReduceMean
                        result += data_[i];
                    }
                    break;
                }
            }
        }
        reduced_[0] = result;
        return STATUS_SUCCESS;
    }

    // printf("reduce_cpu handling\n");
// #pragma omp parallel for
    for(uint64_t i = 0; i < reduced_size_; i++) {
        float result;
        // intitialzation
        switch (reduce_mode_) {
            case 0: { // ReduceMin
                result = std::numeric_limits<float>::max();
            }
            break;
            case 1: { // ReduceMax
                result = std::numeric_limits<float>::lowest();
            }
            break;
            case 2: { // ReduceMean
                result = 0;
            }
            break;
        }

        uint64_t idx = 0;
        uint64_t temp_i = i;
        for(uint64_t j = 0; j < reduced_ndim_; j++) {
            idx += temp_i / reduced_strides_[j] * data_strides_[reduced_axes_[j]];
            temp_i %= reduced_strides_[j];
        }
        for(uint64_t j = 0; j < axes_size_; j++) {
            uint64_t data_idx_ = idx;
            uint64_t temp_j = j;
            for(uint64_t k = 0; k < axes_ndim_; k++) {
                data_idx_ += temp_j / axes_strides_[k] * data_strides_[axes_[k]];
                temp_j %= axes_strides_[k];
            }

            if constexpr (std::is_same<Tdata, uint16_t>::value) {
                switch (reduce_mode_) {
                    case 0: { // ReduceMin
                        result = std::min(result, f16_to_f32(data_[data_idx_]));
                    }
                    break;
                    case 1: { // ReduceMax
                        result = std::max(result, f16_to_f32(data_[data_idx_]));
                    }
                    break;
                    case 2: { // ReduceMean
                        result += f16_to_f32(data_[data_idx_]);
                    }
                    break;
                }
            } else {
                switch (reduce_mode_) {
                    case 0: { // ReduceMin
                        result = std::min(result, data_[data_idx_]);
                    }
                    break;
                    case 1: { // ReduceMax
                        result = std::max(result, data_[data_idx_]);
                    }
                    break;
                    case 2: { // ReduceMean
                        result += data_[data_idx_];
                    }
                    break;
                }
            }
        }

        // final
        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            switch (reduce_mode_) {
                case 0: { // ReduceMin
                    reduced_[i] = f32_to_f16(result);
                }
                break;
                case 1: { // ReduceMax
                    reduced_[i] = f32_to_f16(result);
                }
                break;
                case 2: { // ReduceMean
                    reduced_[i] = f32_to_f16(result / axes_size_);
                }
                break;
            }
        } else {
            switch (reduce_mode_) {
                case 0: { // ReduceMin
                    reduced_[i] = result;
                }
                break;
                case 1: { // ReduceMax
                    reduced_[i] = result;
                }
                break;
                case 2: { // ReduceMean
                    reduced_[i] = result / axes_size_;
                }
                break;
            }
        }
    }

    if (desc->keepdims == true && false) {
        auto new_reduced_strides = new int64_t[reduced_ndim_];
        uint64_t j = 0;
        for(uint64_t i = 0, j = 0; i < data_ndim_; i++) {
            new_reduced_strides[i] = 1;
            
            for (; j < reduced_ndim_; j++) {
                if (i == reduced_axes_[j]) {
                    new_reduced_strides[i] = reduced_strides_[j];
                    break;
                }
            }
        }
        
        desc->reduced_ndim = data_ndim_;
        delete[] desc->reduced_strides;
        printf("\tdelete reduced_strides\n");
        desc->reduced_strides = new_reduced_strides;
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuReduce(ReduceCpuDescriptor_t desc,
                           void *reduced,
                           void const *data,
                           void *stream) {
    if (desc->dt == F16) {
        return reduce_cpu<uint16_t>(desc, reduced, data);
    }
    if (desc->dt == F32) {
        return reduce_cpu<float>(desc, reduced, data);
    }

    return STATUS_BAD_TENSOR_DTYPE;
}
