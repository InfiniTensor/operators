from ctypes import POINTER, Structure, c_int32, c_void_p, c_uint64, c_int64
import ctypes
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    DeviceEnum,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    destroy_handle,
    check_error,
)

from operatorspy.tests.test_utils import get_args
import torch
from typing import Tuple

# constant for control whether profile the pytorch and lib functions
# NOTE: need to manually add synchronization function to the lib function,
#       e.g., cudaDeviceSynchronize() for CUDA
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class ReduceMaxDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopReduceMaxDescriptor_t = POINTER(ReduceMaxDescriptor)


def reduce_max(x, axes, keepdim=False):
    return torch.amax(x, dim=axes, keepdim=keepdim)


def inferShape(x_shape, axes, keep_dims):
    output_shape = list(x_shape)
    
    if keep_dims:
        for axis in axes:
            # Convert negative axis to positive
            actual_axis = axis if axis >= 0 else len(x_shape) + axis
            output_shape[actual_axis] = 1
    else:
        # Sort axes in descending order to avoid index shifting after removal
        sorted_axes = sorted([axis if axis >= 0 else len(x_shape) + axis for axis in axes], reverse=True)
        for axis in sorted_axes:
            output_shape.pop(axis)
    
    return tuple(output_shape)

# convert a python tuple to a ctype void pointer
def tuple_to_void_p(py_tuple: Tuple):
    array = ctypes.c_int64 * len(py_tuple)
    data_array = array(*py_tuple)
    return ctypes.cast(data_array, ctypes.c_void_p)

def test(
    lib,
    handle,
    torch_device,
    x_shape,
    axes,
    keep_dims,
    tensor_dtype=torch.float32,
):
    print(
        f"Testing ReduceMax on {torch_device} with x_shape:{x_shape} axes:{axes} keep_dims:{keep_dims} dtype:{tensor_dtype}"
    )

    x = torch.rand(x_shape, dtype=tensor_dtype).to(torch_device)
    y_shape = inferShape(x_shape, axes, keep_dims)
    y = torch.zeros(y_shape, dtype=tensor_dtype).to(torch_device)

    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = reduce_max(x, axes, keep_dims)
    
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            ans = reduce_max(x, axes, keep_dims)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :6f}")

    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib)
    descriptor = infiniopReduceMaxDescriptor_t()

    check_error(
        lib.infiniopCreateReduceMaxDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
            tuple_to_void_p(axes),
            len(axes),
            1 if keep_dims else 0,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()

    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(
            lib.infiniopReduceMax(
                descriptor,
                y_tensor.data,
                x_tensor.data,
                None,
            )
        )
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopReduceMax(
                    descriptor,
                    y_tensor.data,
                    x_tensor.data,
                    None,
                )
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"    lib time: {elapsed :6f}")

    if (tensor_dtype == torch.float16):
        assert torch.allclose(y, ans, atol=0, rtol=1e-3)
    else:
        assert torch.allclose(y, ans, atol=0, rtol=1e-5)
    check_error(lib.infiniopDestroyReduceMaxDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape, axes, keep_dims in test_cases:
        test(lib, handle, "cpu", x_shape, axes, keep_dims, tensor_dtype=torch.float32)
        test(lib, handle, "cpu", x_shape, axes, keep_dims, tensor_dtype=torch.float16)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # x_shape, axes, keep_dims
        ((2, 3, 4, 5), (1,), True),
        ((2, 3, 4, 5), (1,), False),
        ((2, 3, 4, 5), (2, 1), True),
        ((2, 3, 4, 5), (1, 2), False),
        ((2, 3, 4, 5), (0, 1, 2, 3), True),
        ((2, 3, 4, 5), (-1, -2), True),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateReduceMaxDescriptor.restype = c_int32
    lib.infiniopCreateReduceMaxDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopReduceMaxDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_void_p,
        c_uint64,
        c_int32,
    ]
    lib.infiniopReduceMax.restype = c_int32
    lib.infiniopReduceMax.argtypes = [
        infiniopReduceMaxDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyReduceMaxDescriptor.restype = c_int32
    lib.infiniopDestroyReduceMaxDescriptor.argtypes = [
        infiniopReduceMaxDescriptor_t,
    ]

    test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
