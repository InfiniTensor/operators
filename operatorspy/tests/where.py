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


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()
    INPLACE_Y = auto()


class WhereDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopWhereDescriptor_t = POINTER(WhereDescriptor)

def where(condition, x, y):
    return torch.where(condition, x, y)


def test(
    lib,
    handle,
    torch_device,
    o_shape, 
    condition_shape,
    x_shape,
    y_shape,
    tensor_dtype=torch.float16,
    inplace=Inplace.OUT_OF_PLACE,
):
    print(
        f"Testing Where on {torch_device} with o_shape:{o_shape} x_shape:{x_shape} y_shape:{y_shape}\
        dtype:{tensor_dtype} inplace: {inplace.name}"
    )
    if x_shape != y_shape and inplace != Inplace.OUT_OF_PLACE:
        print("Unsupported test: broadcasting does not support in-place")
        return

    x = torch.rand(x_shape, dtype=tensor_dtype).to(torch_device)
    y = torch.rand(y_shape, dtype=tensor_dtype).to(torch_device)
    output = torch.rand(o_shape, dtype=tensor_dtype).to(torch_device) \
        if inplace == Inplace.OUT_OF_PLACE else (x if inplace == Inplace.INPLACE_X else y)
    condition = torch.randint(0, 2, condition_shape, dtype=torch.uint8).to(torch_device)

    descriptor = infiniopWhereDescriptor_t()
    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib)
    o_tensor = to_tensor(output, lib) \
        if inplace == Inplace.OUT_OF_PLACE else (x_tensor if inplace == Inplace.INPLACE_X else y_tensor)
    condition_tensor = to_tensor(condition, lib)

    ans = where(condition, x, y)

    check_error(
        lib.infiniopCreateWhereDescriptor(
            handle,
            ctypes.byref(descriptor),
            o_tensor.descriptor,
            condition_tensor.descriptor,
            x_tensor.descriptor,
            y_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    o_tensor.descriptor.contents.invalidate()
    condition_tensor.descriptor.contents.invalidate()
    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()

    check_error(
        lib.infiniopWhere(descriptor, 
                          o_tensor.data, 
                          condition_tensor.data,
                          x_tensor.data,
                          y_tensor.data,
                          None)
    )

    # print(f"--ans:{ans}, output:{output}")
    assert torch.allclose(output, ans, atol=0, rtol=0)
    check_error(lib.infiniopDestroyWhereDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for o_shape, condition_shape, x_shape, y_shape, inplace in test_cases:
        test(lib, handle, "cpu", o_shape, condition_shape, x_shape, y_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "cpu", o_shape, condition_shape, x_shape, y_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # o_shape, condition_shape, x_shape, y_shape, inplace
        ((1, 3), (1, 3), (1, 3), (1, 3), Inplace.OUT_OF_PLACE),
        ((3, 3), (3, 3), (3, 3), (3, 3), Inplace.OUT_OF_PLACE),
        ((), (), (), (), Inplace.OUT_OF_PLACE),
        ((2, 20, 3), (2, 20, 3), (2, 1, 3), (2, 20, 3), Inplace.OUT_OF_PLACE),

        ((32, 20, 512), (32, 20, 512), (32, 20, 512), (32, 20, 512), Inplace.INPLACE_X),
        ((32, 20, 512), (32, 20, 512), (32, 20, 512), (32, 20, 512), Inplace.INPLACE_Y),
        ((32, 150, 512), (32, 150, 512), (32, 150, 512), (32, 150, 1), Inplace.OUT_OF_PLACE),
        ((32, 256, 112, 112), (32, 256, 112, 112), (32, 256, 112, 1), (32, 256, 112, 112), Inplace.OUT_OF_PLACE),

        ((2, 4, 3), (2, 4, 3), (2, 1, 3), (4, 3), Inplace.OUT_OF_PLACE),
        ((2, 3, 4, 5), (2, 3, 4, 5), (2, 3, 4, 5), (5,), Inplace.OUT_OF_PLACE),
        ((3, 2, 4, 5), (3, 2, 4, 5), (4, 5), (3, 2, 1, 1), Inplace.OUT_OF_PLACE),
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
        # 
        c_void_p,
    ]
    lib.infiniopDestroyWhereDescriptor.restype = c_int32
    lib.infiniopDestroyWhereDescriptor.argtypes = [
        infiniopWhereDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)
    # if args.cuda:
    #     test_cuda(lib, test_cases)
    # if args.bang:
    #     test_bang(lib, test_cases)
    if not (args.cpu or args.cuda or args.bang):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
