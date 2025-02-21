#ifndef __ACTIVATIONS_H__
#define __ACTIVATIONS_H__

/**
 * @brief Specifies the type of activation function
 */
typedef enum InfiniActivationMode {
    // activation functions
    INFINI_ACTIVATION_IDENTITY = 0,
    INFINI_ACTIVATION_RELU = 1,
    INFINI_ACTIVATION_LEAKY_RELU = 2,
    INFINI_ACTIVATION_CLIPPED_RELU = 3,
    INFINI_ACTIVATION_SIGMOID = 4,
    INFINI_ACTIVATION_HEAVISIDE_STEP = 5,
    INFINI_ACTIVATION_ELU = 6,
    INFINI_ACTIVATION_GELU = 7,
    INFINI_ACTIVATION_SIN = 8,
    INFINI_ACTIVATION_TANH = 9,

    // Count
    // NOTE: new activation functions should add before "Count"
    INFINI_ACTIVATION_COUNT,
} InfiniActivationMode_t;

#endif
