from ctypes import POINTER, Structure, c_int32, c_void_p
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


class WhereDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopWhereDescriptor_t = POINTER(WhereDescriptor)


def where(condition, x, y):
    return torch.where(condition.bool(), x, y)


def test(
    lib,
    handle,
    torch_device,
    output_shape,
    condition_shape,
    x_shape,
    y_shape,
    tensor_dtype=torch.float16,
):
    print(
        f"Testing Where on {torch_device} with output_shape:{output_shape} condition_shape:{condition_shape} x_shape:{x_shape} y_shape:{y_shape} dtype:{tensor_dtype}"
    )

    condition = torch.randint(0, 2, condition_shape, dtype=torch.uint8).to(torch_device)
    x = torch.rand(x_shape, dtype=tensor_dtype).to(torch_device)
    y = torch.rand(y_shape, dtype=tensor_dtype).to(torch_device)
    output = torch.rand(output_shape, dtype=tensor_dtype).to(torch_device)

    ans = where(condition, x, y)

    condition_tensor = to_tensor(condition, lib)
    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib)
    output_tensor = to_tensor(output, lib)
    descriptor = infiniopWhereDescriptor_t()

    check_error(
        lib.infiniopCreateWhereDescriptor(
            handle,
            ctypes.byref(descriptor),
            output_tensor.descriptor,
            condition_tensor.descriptor,
            x_tensor.descriptor,
            y_tensor.descriptor,
        )
    )

    condition_tensor.descriptor.contents.invalidate()
    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()
    output_tensor.descriptor.contents.invalidate()

    check_error(
        lib.infiniopWhere(descriptor, output_tensor.data, condition_tensor.data, x_tensor.data, y_tensor.data, None)
    )
    assert torch.allclose(output, ans, atol=0, rtol=0)
    check_error(lib.infiniopDestroyWhereDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for output_shape, condition_shape, x_shape, y_shape in test_cases:
        test(lib, handle, "cpu", output_shape, condition_shape, x_shape, y_shape, tensor_dtype=torch.float16)
        test(lib, handle, "cpu", output_shape, condition_shape, x_shape, y_shape, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # output_shape, condition_shape, x_shape, y_shape
        ((1, 3), (1, 3), (1, 3), (1, 3)),
        ((), (), (), ()),
        ((3, 3), (3, 3), (3, 3), (3, 3)),
        ((2, 20, 3), (2, 1, 3), (2, 20, 3), (2, 20, 3)),
        ((32, 20, 512), (32, 20, 512), (32, 20, 512), (32, 20, 512)),
        ((32, 256, 112, 112), (32, 256, 112, 1), (32, 256, 112, 112), (32, 256, 112, 112)),
        ((2, 4, 3), (2, 1, 3), (4, 3), (4, 3)),
        ((2, 3, 4, 5), (2, 3, 4, 5), (5,), (5,)),
        ((3, 2, 4, 5), (4, 5), (3, 2, 1, 1), (3, 2, 1, 1)),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateWhereDescriptor.restype = c_int32
    lib.infiniopCreateWhereDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopWhereDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopWhere.restype = c_int32
    lib.infiniopWhere.argtypes = [
        infiniopWhereDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyWhereDescriptor.restype = c_int32
    lib.infiniopDestroyWhereDescriptor.argtypes = [
        infiniopWhereDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)
    if not args.cpu:
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
