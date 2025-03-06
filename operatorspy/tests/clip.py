from ctypes import POINTER, Structure, c_int32, c_void_p, c_float
import ctypes
import sys
import os

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
from enum import Enum, auto
import torch
import numpy as np


class ClipDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopClipDescriptor_t = POINTER(ClipDescriptor)


def clip(x, min, max):
    return torch.clip(x, min, max)


def test(
    lib,
    handle,
    torch_device,
    c_shape,
    min,
    max,
    tensor_dtype=torch.float16,
):
    print(
        f"Testing Clip on {torch_device} with c_shape:{c_shape} dtype:{tensor_dtype}"
    )

    input = torch.rand(c_shape, dtype=tensor_dtype).to(torch_device)
    output = torch.empty(c_shape, dtype=tensor_dtype).to(torch_device)
    min_v = min if min else torch.finfo(tensor_dtype).min
    max_v = max if max else torch.finfo(tensor_dtype).max
    min_val = torch.tensor(min_v, dtype=tensor_dtype).to(torch_device)
    max_val = torch.tensor(max_v, dtype=tensor_dtype).to(torch_device)
    # min = np.random.uniform(0, 1)
    # max = np.random.uniform(0, 1)
    min_fp16_value = min_val.item()
    max_fp16_value = max_val.item()

    ans = clip(input, min_val, max_val)

    input_tensor = to_tensor(input, lib)
    output_tensor = to_tensor(output, lib)
    descriptor = infiniopClipDescriptor_t()

    min_c = c_float(min_fp16_value)
    max_c = c_float(max_fp16_value)

    check_error(
        lib.infiniopCreateClipDescriptor(
            handle,
            ctypes.byref(descriptor),
            output_tensor.descriptor,
            input_tensor.descriptor,
            ctypes.byref(min_c) if min else None,
            ctypes.byref(max_c) if max else None
        )
    )

    input_tensor.descriptor.contents.invalidate()
    output_tensor.descriptor.contents.invalidate()

    check_error(
        lib.infiniopClip(descriptor, output_tensor.data, input_tensor.data, None)
    )

    assert torch.allclose(output, ans, atol=0, rtol=0)
    check_error(lib.infiniopDestroyClipDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for c_shape, min, max in test_cases:
        test(lib, handle, "cpu", c_shape, min, max, tensor_dtype=torch.float16)
        test(lib, handle, "cpu", c_shape, min, max, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # c_shape
        ((1, 3), 0.2, 0.4),
        ((3, 3), -0.1, 0.7),
        ((2, 20, 3), 0.5, 0.9),
        ((32, 20, 512), -0.2, 0.9),
        ((32, 256, 112, 112), 0.1, None),
        ((3, 2, 4, 5), None, None),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateClipDescriptor.restype = c_int32
    lib.infiniopCreateClipDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopClipDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        POINTER(c_float),
        POINTER(c_float),
    ]
    lib.infiniopClip.restype = c_int32
    lib.infiniopClip.argtypes = [
        infiniopClipDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyClipDescriptor.restype = c_int32
    lib.infiniopDestroyClipDescriptor.argtypes = [
        infiniopClipDescriptor_t,
    ]

    test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
