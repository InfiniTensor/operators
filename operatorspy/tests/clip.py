from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p, c_float
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


class ClipDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopClipDescriptor_t = POINTER(ClipDescriptor)

def clip(x, min_val = -torch.inf, max_val = torch.inf):
    return torch.clip(x, min_val, max_val)

def test(
    lib,
    handle,
    torch_device,
    tensor_shape, 
    max_val,
    min_val,
    tensor_dtype=torch.float16,
    inplace=Inplace.OUT_OF_PLACE,
):
    print(
        f"Testing Clip on {torch_device} with tensor_shape:{tensor_shape} dtype:{tensor_dtype} inplace: {inplace.name}"
    )

    x = torch.rand(tensor_shape, dtype=tensor_dtype).to(torch_device) * 2 - 1
    y = torch.rand(tensor_shape, dtype=tensor_dtype).to(torch_device) if inplace == Inplace.OUT_OF_PLACE else x



    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = clip(x,min_val,max_val)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = clip(x,min_val,max_val)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :6f}")

    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib) if inplace == Inplace.OUT_OF_PLACE else x_tensor
    descriptor = infiniopClipDescriptor_t()

    check_error(
        lib.infiniopCreateClipDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
            max_val,
            min_val
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()

    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(lib.infiniopClip(descriptor, y_tensor.data, x_tensor.data, None))   
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopClip(descriptor, y_tensor.data, x_tensor.data, None)
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"    lib time: {elapsed :6f}")

    assert torch.allclose(y, ans, atol=1e-0, rtol=1e-0)
    check_error(lib.infiniopDestroyClipDescriptor(descriptor))
def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for tensor_shape, max_val, min_val, inplace in test_cases:
        test(lib, handle, "cpu", tensor_shape, max_val, min_val, tensor_dtype=torch.float32, inplace=inplace)
        test(lib, handle, "cpu", tensor_shape, max_val, min_val, tensor_dtype=torch.float16, inplace=inplace)
    destroy_handle(lib, handle)

def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for tensor_shape, max_val, min_val, inplace in test_cases:
        test(lib, handle, "cuda", tensor_shape, max_val, min_val, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "cuda", tensor_shape, max_val, min_val, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)

if __name__ == "__main__":
    test_cases = [
        # tensor_shape, inplace
        ((), 0.5, -0.5, Inplace.OUT_OF_PLACE),
        ((), 0.5, -0.5, Inplace.INPLACE_X),
        ((1, 3), 0.5, -0.5, Inplace.OUT_OF_PLACE),
        ((3, 3), 0.5, -0.5, Inplace.OUT_OF_PLACE),
        ((3, 3, 13, 9, 17), 0.5, -0.5, Inplace.INPLACE_X),
        ((32, 20, 512), 0.5, -0.5, Inplace.INPLACE_X),
        ((2, 3, 4, 5), 0.5, -0.5, Inplace.INPLACE_X),
        ((33, 333, 333), 0.5, -0.5, Inplace.OUT_OF_PLACE),
        ((32, 256, 112, 112), 0.5, -0.5, Inplace.OUT_OF_PLACE),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateClipDescriptor.restype = c_int32
    lib.infiniopCreateClipDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopClipDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_float,
        c_float,
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

    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    
    print("\033[92mTest passed!\033[0m")