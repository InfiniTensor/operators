from ctypes import POINTER, Structure, c_int32, c_uint64, c_uint8, c_void_p, c_float
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
from enum import Enum, auto
import torch

PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()


class WhereDescriptor(Structure):
    _fields_ = [("device", c_int32)]

infiniopWhereDescriptor_t = POINTER(WhereDescriptor)

def where(condition, x, y):
    return torch.where(condition, x, y)

def test(
    lib,
    handle,
    torch_device,
    tensor_shape, 
    tensor_dtype=torch.float16,
    inplace=Inplace.OUT_OF_PLACE,
):
    print(
        f"Testing Where on {torch_device} with tensor_shape:{tensor_shape} dtype:{tensor_dtype} inplace: {inplace.name}"
    )    

    x = torch.rand(tensor_shape, dtype=tensor_dtype).to(torch_device) * 2 - 1
    y = torch.rand(tensor_shape, dtype=tensor_dtype).to(torch_device) 
    condition = torch.randint(0, 2, tensor_shape, dtype=torch.uint8).to(torch_device)
    dst = torch.empty_like(x)if inplace == Inplace.OUT_OF_PLACE else x

    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = where(condition, y, x)        
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = where(condition, y, x)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :6f}")

    x_tensor = to_tensor(x, lib)
    dst_tensor = to_tensor(dst, lib)if inplace == Inplace.OUT_OF_PLACE else x_tensor
    y_tensor = to_tensor(y, lib) 
    condition_tensor = to_tensor(condition, lib)
    descriptor = infiniopWhereDescriptor_t()

    check_error(
        lib.infiniopCreateWhereDescriptor(
            handle,
            ctypes.byref(descriptor),
            dst_tensor.descriptor,
            x_tensor.descriptor,
            y_tensor.descriptor,
            condition_tensor.descriptor,
        )
    )

    dst_tensor.descriptor.contents.invalidate()
    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()
    condition_tensor.descriptor.contents.invalidate()

    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(lib.infiniopWhere(descriptor, dst_tensor.data, x_tensor.data, y_tensor.data, condition_tensor.data, None))
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopWhere(descriptor, dst_tensor.data, x_tensor.data, y_tensor.data, condition_tensor.data, None)
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"    lib time: {elapsed :6f}")
    assert torch.allclose(dst, ans, atol=0, rtol=0)
    check_error(lib.infiniopDestroyWhereDescriptor(descriptor))

def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for tensor_shape, inplace in test_cases:
        test(lib, handle, "cpu", tensor_shape, tensor_dtype=torch.float32, inplace=inplace)
        test(lib, handle, "cpu", tensor_shape, tensor_dtype=torch.float16, inplace=inplace)
    destroy_handle(lib, handle)

def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for tensor_shape, inplace in test_cases:
        test(lib, handle, "cuda", tensor_shape, tensor_dtype=torch.float32, inplace=inplace)
        test(lib, handle, "cuda", tensor_shape, tensor_dtype=torch.float16, inplace=inplace)
    destroy_handle(lib, handle)

if __name__ == "__main__":
    test_cases = [
        # tensor_shape, inplace
        ((), Inplace.OUT_OF_PLACE),
        ((), Inplace.INPLACE_X),
        ((1, 3), Inplace.OUT_OF_PLACE),
        ((3, 3), Inplace.OUT_OF_PLACE),
        ((3, 3, 13, 9, 17), Inplace.INPLACE_X),
        ((32, 20, 512), Inplace.INPLACE_X),
        ((33, 333, 333), Inplace.OUT_OF_PLACE),
        ((32, 256, 112, 112), Inplace.OUT_OF_PLACE),
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

    if args.cuda:
        test_cuda(lib, test_cases)

    print("\033[92mTest passed!\033[0m")