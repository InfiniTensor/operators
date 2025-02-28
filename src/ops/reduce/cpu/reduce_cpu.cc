#include "reduce_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateReduceDescriptor(infiniopHandle_t,
                                           ReduceCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t y,
                                           infiniopTensorDescriptor_t x,
                                           int *axes,
                                           uint64_t axes_ndim,
                                           int reduce_op) {
    if (!is_contiguous(y) || !is_contiguous(x)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }

    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    bool use_1Dreduce = axes_ndim == x->ndim; // means all axes are reduced, seem as 1D reduce to avoid out_ndim=0
    bool is_y_scalar = std::accumulate(y->shape, y->shape + y->ndim, 1ULL, std::multiplies<uint64_t>()) == 1;
    if(use_1Dreduce != is_y_scalar) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    std::vector<int> axes_, out_of_axes;

    uint64_t ndim = x->ndim;
    uint64_t out_ndim = ndim - axes_ndim;
    uint64_t axes_size = 1;
    uint64_t out_size = 1;

    for(uint64_t i = 0; i < ndim; i++) {
        bool is_axis = false;
        for(uint64_t j = 0; j < axes_ndim; j++) {
            if (axes[j] == i) {
                is_axis = true;
                break;
            }
        }
        if (is_axis) {
            axes_size *= x->shape[i];
            axes_.emplace_back(i);
        } else {
            out_size *= x->shape[i];
            out_of_axes.emplace_back(i);
        }
    }
    sort(axes_.begin(), axes_.end());
    sort(out_of_axes.begin(), out_of_axes.end());

    std::vector<int> axes_strides(axes_ndim, 1);
    std::vector<int> out_strides(out_ndim, 1);

    for(uint64_t i = 0; i < axes_ndim; i++) {
        for(uint64_t j = i + 1; j < axes_ndim; j++) {
            axes_strides[i] *= x->shape[axes[j]];
        }
    }
    for(uint64_t i = 0; i < out_ndim; i++) {
        for(uint64_t j = i + 1; j < out_ndim; j++) {
            out_strides[i] *= x->shape[out_of_axes[j]];
        }
    }
    
    int64_t *strides = new int64_t[ndim];
    std::memcpy(strides, x->strides, ndim * sizeof(int64_t));

    
    *desc_ptr = new ReduceCpuDescriptor{
        DevCpu,
        x->dt,
        use_1Dreduce,
        axes_,
        out_of_axes,
        ndim,
        axes_ndim,
        out_ndim,
        axes_size,
        out_size,
        strides,
        axes_strides,
        out_strides,
        reduce_op
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyReduceDescriptor(ReduceCpuDescriptor_t desc) {
    delete[] (desc->strides);
    delete desc;
    return STATUS_SUCCESS;
}


template<typename Tdata>
infiniopStatus_t reduce_cpu_1D(ReduceCpuDescriptor_t desc, void *y, void const *x) {
    auto x_ = reinterpret_cast<Tdata const *>(x);
    auto y_ = reinterpret_cast<Tdata *>(y);
    auto data_size_ = desc->out_size * desc->axes_size;
    auto reduce_op_ = desc->reduce_op;

    if constexpr (std::is_same<Tdata, uint16_t>::value) {
        switch (reduce_op_) {
            case 0: { // ReduceMin
                float result = std::numeric_limits<float>::max();
                for (uint64_t i = 0; i < data_size_; i++) {
                    result = std::min(result, f16_to_f32(x_[i]));
                }
                y_[0] = f32_to_f16(result);
            }
            break;
            case 1: { // ReduceMax
                float result = std::numeric_limits<float>::lowest();
                for (uint64_t i = 0; i < data_size_; i++) {
                    result = std::max(result, f16_to_f32(x_[i]));
                }
                y_[0] = f32_to_f16(result);
            }
            break;
            case 2: { // ReduceMean
                float sum = 0;
                for (uint64_t i = 0; i < data_size_; i++) {
                    sum += f16_to_f32(x_[i]);
                }
                y_[0] = f32_to_f16(sum / data_size_);
            }
            break;
        }
    } else {
        switch (reduce_op_) {
            case 0: { // ReduceMin
                Tdata result = std::numeric_limits<Tdata>::max();
                for (uint64_t i = 0; i < data_size_; i++) {
                    result = std::min(result, x_[i]);
                }
                y_[0] = result;
            }
            break;
            case 1: { // ReduceMax
                Tdata result = std::numeric_limits<Tdata>::lowest();
                for (uint64_t i = 0; i < data_size_; i++) {
                    result = std::max(result, x_[i]);
                }
                y_[0] = result;
            }
            break;
            case 2: { // ReduceMean
                Tdata sum = 0;
                for (uint64_t i = 0; i < data_size_; i++) {
                    sum += x_[i];
                }
                y_[0] = sum / data_size_;
            }
            break;
        }
    }
    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t reduce_cpu(ReduceCpuDescriptor_t desc, void *y, void const *x) {
    auto x_ = reinterpret_cast<Tdata const *>(x);
    auto y_ = reinterpret_cast<Tdata *>(y);
    auto axes_ = desc->axes;
    auto out_of_axes_ = desc->out_of_axes;
    auto ndim_ = desc->ndim;
    auto axes_ndim_ = desc->axes_ndim;
    auto out_ndim_ = desc->out_ndim;
    auto axes_size_ = desc->axes_size;
    auto out_size_ = desc->out_size;
    auto strides_ = desc->strides;
    auto axes_strides_ = desc->axes_strides;
    auto out_strides_ = desc->out_strides;
    auto reduce_op_ = desc->reduce_op;

#pragma omp parallel for
    for(uint64_t i = 0; i < out_size_; i++) {
        uint64_t idx = 0;
        uint64_t temp_i = i;
        for(uint64_t j = 0; j < out_ndim_; j++) {
            idx += temp_i / out_strides_[j] * strides_[out_of_axes_[j]];
            temp_i %= out_strides_[j];
        }

        float result;
        switch (reduce_op_) {
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

        for(uint64_t j = 0; j < axes_size_; j++) {
            uint64_t idx_ = idx;
            uint64_t temp_j = j;
            for(uint64_t k = 0; k < axes_ndim_; k++) {
                idx_ += temp_j / axes_strides_[k] * strides_[axes_[k]];
                temp_j %= axes_strides_[k];
            }

            if constexpr (std::is_same<Tdata, uint16_t>::value) {
                switch (reduce_op_) {
                    case 0: { // ReduceMin
                        result = std::min(result, f16_to_f32(x_[idx_]));
                    }
                    break;
                    case 1: { // ReduceMax
                        result = std::max(result, f16_to_f32(x_[idx_]));
                    }
                    break;
                    case 2: { // ReduceMean
                        result += f16_to_f32(x_[idx_]);
                    }
                    break;
                }
            } else {
                switch (reduce_op_) {
                    case 0: { // ReduceMin
                        result = std::min(result, x_[idx_]);
                    }
                    break;
                    case 1: { // ReduceMax
                        result = std::max(result, x_[idx_]);
                    }
                    break;
                    case 2: { // ReduceMean
                        result += x_[idx_];
                    }
                    break;
                }
            }
        }
        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            switch (reduce_op_) {
                case 0: { // ReduceMin
                    y_[i] = f32_to_f16(result);
                }
                break;
                case 1: { // ReduceMax
                    y_[i] = f32_to_f16(result);
                }
                break;
                case 2: { // ReduceMean
                    y_[i] = f32_to_f16(result / axes_size_);
                }
                break;
            }
        } else {
            switch (reduce_op_) {
                case 0: { // ReduceMin
                    y_[i] = result;
                }
                break;
                case 1: { // ReduceMax
                    y_[i] = result;
                }
                break;
                case 2: { // ReduceMean
                    y_[i] = result / axes_size_;
                }
                break;
            }
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuReduce(ReduceCpuDescriptor_t desc, void *y, void const *x, void *stream) {
    if (desc->dtype == F16) {
        if(desc->use_1Dreduce) {
            return reduce_cpu_1D<uint16_t>(desc, y, x);
        } else {
            return reduce_cpu<uint16_t>(desc, y, x);
        }   
    }
    if (desc->dtype == F32) {
        if(desc->use_1Dreduce) {
            return reduce_cpu_1D<float>(desc, y, x);
        } else {
            return reduce_cpu<float>(desc, y, x);
        }
    }
    return STATUS_BAD_TENSOR_DTYPE;
}