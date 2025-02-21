#include "conv_act_cpu.h"
#include "../../utils.h"

// get the total number of elements in arr
inline uint64_t getTotalSize(const uint64_t *arr, uint64_t ndim) {
    return std::accumulate(arr, arr + ndim, 1ULL, std::multiplies<uint64_t>());
}

// check if padding is needed
inline bool requirePadding(uint64_t const *pads, uint64_t ndim) {
    return std::any_of(pads, pads + ndim - 2,
                       [](uint64_t pad) { return pad > 0; });
}

// check if bias is needed
template<typename Tdata>
inline bool requireBias(const Tdata *b, uint64_t length) {
    return std::any_of(b, b + length, [](const Tdata &bias) { return bias != 0; });
}

template<typename Tdata>
Tdata relu(const Tdata &x) {
    return (x > Tdata(0)) ? x : Tdata(0);
}

template<typename Tdata>
Tdata sigmoid(const Tdata &x) {
    return Tdata(1) / (Tdata(1) + std::exp(-x));
}

infiniopStatus_t cpuCreateConvActDescriptor(infiniopHandle_t,
                                            ConvActCpuDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t y,
                                            infiniopTensorDescriptor_t x,
                                            infiniopTensorDescriptor_t w,
                                            infiniopTensorDescriptor_t b,
                                            uint64_t const *pads,
                                            int64_t const *strides,
                                            uint64_t const *dilations,
                                            uint64_t n,
                                            InfiniActivationMode_t activation_mode,
                                            ConvActParam_t act_params) {
    uint64_t ndim = y->ndim;
    if (ndim < 3 || ndim != x->ndim || ndim != w->ndim || n != ndim - 2) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (x->shape[0] != y->shape[0] || w->shape[0] != y->shape[1] || x->shape[1] != w->shape[1]) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != x->dt || y->dt != w->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (b) {
        if (b->ndim != 1 || b->shape[0] != w->shape[0]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
        if (y->dt != b->dt) {
            return STATUS_BAD_TENSOR_DTYPE;
        }
    }
    // check if the activation_mode is valid
    if (activation_mode < 0 || activation_mode >= INFINI_ACTIVATION_COUNT) {
        return STATUS_BAD_PARAM;
    }
    // check if the activation_mode is currently supported by this platform
    switch (activation_mode) {
        case INFINI_ACTIVATION_IDENTITY:
        case INFINI_ACTIVATION_RELU:
        case INFINI_ACTIVATION_SIGMOID:
            break;
        default:
            return STATUS_BAD_PARAM;
    }

    uint64_t y_size = getTotalSize(y->shape, ndim);
    uint64_t padded_x_size = requirePadding(pads, ndim) ? getPaddedSize(ndim, x->shape, pads) : 0;
    uint64_t *x_shape = new uint64_t[ndim];
    uint64_t *w_shape = new uint64_t[ndim];
    uint64_t *b_shape = new uint64_t[1];
    uint64_t *y_shape = new uint64_t[ndim];
    uint64_t *pads_ = new uint64_t[n];
    int64_t *strides_ = new int64_t[n];
    uint64_t *dilations_ = new uint64_t[n];
    memcpy(x_shape, x->shape, ndim * sizeof(*x->shape));
    memcpy(w_shape, w->shape, ndim * sizeof(*w->shape));
    memcpy(b_shape, &(w->shape[0]), sizeof(*w->shape));
    memcpy(y_shape, y->shape, ndim * sizeof(*y->shape));
    memcpy(pads_, pads, n * sizeof(*pads));
    memcpy(strides_, strides, n * sizeof(*strides));
    memcpy(dilations_, dilations, n * sizeof(*dilations));

    uint64_t *padded_shape = nullptr;

    if (padded_x_size > 0) {
        padded_shape = new uint64_t[ndim];
        getPaddedShape(ndim, x_shape, pads, padded_shape);
    }

    *desc_ptr = new ConvActCpuDescriptor{
        DevCpu,
        y->dt,
        ndim,
        y_size,
        padded_x_size,
        padded_shape,
        x_shape,
        w_shape,
        b_shape,
        y_shape,
        pads_,
        strides_,
        dilations_,
        activation_mode,
        b == nullptr,
        act_params,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuGetConvActWorkspaceSize(ConvActCpuDescriptor_t desc, uint64_t *size) {
    *size = desc->padded_x_size * desc->dtype.size;
    if (desc->dtype == F16) {
        *size += desc->y_size * sizeof(float);
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyConvActDescriptor(ConvActCpuDescriptor_t desc) {
    delete[] desc->x_shape;
    delete[] desc->w_shape;
    delete[] desc->b_shape;
    delete[] desc->y_shape;
    delete[] desc->pads;
    delete[] desc->strides;
    delete[] desc->dilations;
    delete[] desc->padded_shape;
    delete desc;
    return STATUS_SUCCESS;
}

// initialize the padded input with the data from the original input
template<typename Tdata>
void fillPaddedInput(ConvActCpuDescriptor_t desc, uint64_t const *padded_x_shape,
                     Tdata *padded_x, Tdata const *x,
                     uint64_t const *pads, uint64_t x_index,
                     uint64_t padded_x_index, uint64_t ndim) {
    const auto x_shape = desc->x_shape[ndim];
    const auto padded_x_shape_ = padded_x_shape[ndim];
    const auto x_base_index = x_index * x_shape;
    const auto padded_x_base_index = padded_x_index * padded_x_shape_ +
                                     (x_shape == padded_x_shape_ ? 0 : pads[ndim - 2]);
#pragma omp parallel for
    for (size_t i = 0; i < x_shape; ++i) {
        // base case (last dimension)
        if (ndim == desc->ndim - 1) {
            padded_x[padded_x_base_index + i] = x[x_base_index + i];
        }
        // recursive case
        else {
            fillPaddedInput(desc, padded_x_shape, padded_x, x, pads, x_base_index + i,
                            padded_x_base_index + i, ndim + 1);
        }
    }
}

// Recursive convolution function
template<typename Xdata, typename Ydata>
void _applyConvAct(ConvActCpuDescriptor_t desc, Ydata *y, Xdata const *x,
                   Xdata const *w, uint64_t const *x_shape,
                   uint64_t x_index, uint64_t w_index, uint64_t y_index,
                   uint64_t ndim) {
    const auto dim_size = x_shape[ndim];
    const auto kernel_size = desc->w_shape[ndim];
    const auto dilation = desc->dilations[ndim - 2];
    const auto stride = desc->strides[ndim - 2];
    const auto steps =
        (dim_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    x_index *= dim_size;
    w_index *= kernel_size;
    y_index *= desc->y_shape[ndim];

    // perform all the convolutions along this axis
    for (size_t i = 0; i < steps; ++i, ++y_index) {
// perform a single convolution
#pragma unroll
        for (size_t k = 0; k < kernel_size; ++k) {
            // calculate the current indices
            const auto curr_x_index = x_index + i * stride + k * dilation;
            const auto curr_w_index = w_index + k;

            // base case (last dimension)
            if (ndim == desc->ndim - 1) {
                if constexpr (std::is_same_v<Xdata, uint16_t>) {
                    y[y_index] += f16_to_f32(x[curr_x_index]) * f16_to_f32(w[curr_w_index]);
                } else {
                    y[y_index] += x[curr_x_index] * w[curr_w_index];
                }
            }
            // recursive case
            else {
                _applyConvAct(desc, y, x, w, x_shape, curr_x_index, curr_w_index,
                              y_index, ndim + 1);
            }
        }
    }
}

// add bias b to the output y
template<typename Xdata, typename Ydata>
void addBias(Ydata *y, const Xdata *b, uint64_t batch_size, uint64_t out_channel_size, uint64_t in_channel_size, uint64_t num_channel_elements) {
#pragma omp parallel for collapse(2)
    // batch
    for (size_t i = 0; i < batch_size; ++i) {
        // output channel
        for (size_t j = 0; j < out_channel_size; ++j) {
            uint64_t y_index = (i * in_channel_size + j) * num_channel_elements;

            // Add the bias to the output channel for the current batch
            for (size_t yi = 0; yi < num_channel_elements; ++yi) {
                if constexpr (std::is_same_v<Xdata, uint16_t> && std::is_same_v<Ydata, float>) {
                    y[y_index + yi] += f16_to_f32(b[j]);
                } else {
                    y[y_index + yi] += b[j];
                }
            }
        }
    }
}

// apply activation function given the mode on the array arr with length n
template<typename Tdata>
void applyActivation(Tdata *arr, uint64_t n, InfiniActivationMode_t mode) {
    if (mode != INFINI_ACTIVATION_IDENTITY) {
        std::function<void(Tdata &)> activation_fn = [](Tdata &value) { value = value; };

        switch (mode) {
            case INFINI_ACTIVATION_RELU:
                activation_fn = [](Tdata &value) { value = relu(value); };
                break;
            case INFINI_ACTIVATION_SIGMOID:
                activation_fn = [](Tdata &value) { value = sigmoid(value); };
                break;
            default:
                break;
        }

#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            activation_fn(arr[i]);
        }
    }
}

template<typename Xdata, typename Ydata>
void applyConvAct(ConvActCpuDescriptor_t desc, Ydata *y, Xdata const *x,
                  Xdata const *w, Xdata const *b, uint64_t const *x_shape) {
    const auto y_num_channel_elements =
        getTotalSize(desc->y_shape + 2, desc->ndim - 2);
    bool biasRequired = b && !desc->bias_is_optional && requireBias(b, desc->b_shape[0]);

#pragma omp parallel for collapse(2) schedule(dynamic)
    // batch
    for (size_t i = 0; i < x_shape[0]; ++i) {

        // output channel
        for (size_t j = 0; j < desc->w_shape[0]; ++j) {
            uint64_t y_index = i * desc->y_shape[1] + j;

            // input channel
            for (size_t k = 0; k < x_shape[1]; ++k) {
                uint64_t x_index = i * x_shape[1] + k;
                uint64_t w_index = j * desc->w_shape[1] + k;
                _applyConvAct(desc, y, x, w, x_shape, x_index, w_index, y_index, 2);
            }
        }
    }

    if (biasRequired) {
        addBias(y, b, x_shape[0], desc->w_shape[0], desc->y_shape[1], y_num_channel_elements);
    }

    // apply activation function
    applyActivation(y, desc->y_size, desc->mode);
}

template<typename Xdata, typename Ydata>
void _conv_bias_act_cpu(ConvActCpuDescriptor_t desc, void *workspace, uint64_t workspace_size,
                        Ydata *y, Xdata const *x, Xdata const *w, Xdata const *b) {
    if (desc->padded_x_size > 0) {
        auto padded_x = reinterpret_cast<Xdata *>(workspace);
        std::fill(padded_x, padded_x + desc->padded_x_size, 0);
        fillPaddedInput<Xdata>(desc, desc->padded_shape, padded_x, x, desc->pads, 0, 0, 0);
        applyConvAct<Xdata, Ydata>(desc, y, padded_x, w, b, desc->padded_shape);
    } else {
        applyConvAct<Xdata, Ydata>(desc, y, x, w, b, desc->x_shape);
    }
}

// Convolution function
template<typename Tdata>
infiniopStatus_t conv_bias_act_cpu(ConvActCpuDescriptor_t desc, void *workspace, uint64_t workspace_size,
                                   void *y, void const *x, void const *w, void const *b) {
    auto y_ = reinterpret_cast<Tdata *>(y);
    auto x_ = reinterpret_cast<Tdata const *>(x);
    auto w_ = reinterpret_cast<Tdata const *>(w);
    auto b_ = reinterpret_cast<Tdata const *>(b);
    std::fill(y_, y_ + desc->y_size, 0);
    _conv_bias_act_cpu<Tdata, Tdata>(desc, workspace, workspace_size, y_, x_, w_, b_);
    return STATUS_SUCCESS;
}

// sepcial case for fp16 (uint16_t)
template<>
infiniopStatus_t conv_bias_act_cpu<uint16_t>(ConvActCpuDescriptor_t desc, void *workspace, uint64_t workspace_size,
                                             void *y, void const *x, void const *w, void const *b) {
    auto y_ = reinterpret_cast<float *>(workspace);
    auto x_ = reinterpret_cast<uint16_t const *>(x);
    auto w_ = reinterpret_cast<uint16_t const *>(w);
    auto b_ = reinterpret_cast<uint16_t const *>(b);
    std::fill(y_, y_ + desc->y_size, 0);

    _conv_bias_act_cpu<uint16_t, float>(desc, y_ + desc->y_size, workspace_size, y_, x_, w_, b_);

    // copy data from y_ to y
    auto y_16 = reinterpret_cast<uint16_t *>(y);
#pragma omp parallel for
    for (size_t i = 0; i < desc->y_size; ++i) {
        y_16[i] = f32_to_f16(y_[i]);
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuConvAct(ConvActCpuDescriptor_t desc,
                            void *workspace, uint64_t workspace_size,
                            void *y, void const *x, void const *w,
                            void const *b, void *stream) {
    if (desc->dtype == F16) {
        return conv_bias_act_cpu<uint16_t>(desc, workspace, workspace_size, y, x, w, b);
    }
    if (desc->dtype == F32) {
        return conv_bias_act_cpu<float>(desc, workspace, workspace_size, y, x, w, b);
    }

    return STATUS_BAD_TENSOR_DTYPE;
}
