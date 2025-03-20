from ctypes import POINTER, Structure, c_int32, c_void_p, c_uint64, c_bool, c_float
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
import numpy as np

PROFILE = True
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

class ClipDescriptor(Structure):
    _fields_ = [("device", c_int32)]

infiniopClipDescriptor_t = POINTER(ClipDescriptor)

def clip(input, min, max):
    return torch.clamp(input, min, max)


def tuple_to_void_p(py_tuple: Tuple):
    array = ctypes.c_int64 * len(py_tuple)
    data_array = array(*py_tuple)
    return ctypes.cast(data_array, ctypes.c_void_p)

def test(
    lib,
    handle,
    torch_device,
    x_shape,
    min,
    max,
    tensor_dtype=torch.float32
):
    print(
        f"Testing clip on {torch_device} with x_shape:{x_shape} dtype:{tensor_dtype} max:{max} min:{min}"
    )
    x = torch.randn(x_shape, dtype=torch.float32, device=torch_device)
    
    output = torch.randn(x_shape, dtype=torch.float32, device=torch_device)
    if min != None:
        min_t = torch.tensor(min, dtype=torch.float32, device=torch_device)
    else:
        min_t = torch.tensor(float("-inf"), dtype=torch.float32, device=torch_device)
    if max != None:
        max_t = torch.tensor(max, dtype=torch.float32, device=torch_device)
    else:
        max_t = torch.tensor(float("inf"), dtype=torch.float32, device=torch_device)
    for i in range(NUM_PRERUN if PROFILE else 1):
        if min == None and max == None:
            break
        ans = clip(x, min_t, max_t)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = clip(x, min_t, max_t)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :10f}")
    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(output, lib)
    descriptor = infiniopClipDescriptor_t()
    check_error(
        lib.infiniopCreateClipDescriptor(
            handle,
            ctypes.byref(descriptor),
            x_tensor.descriptor,
            y_tensor.descriptor,
            ctypes.byref(c_float(min)) if min != None else None,
            ctypes.byref(c_float(max)) if max != None else None,
        )
    )
    #Ss = [1024, 2048, 4096]
    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()
    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(
            lib.infiniopClip(
                descriptor,
                x_tensor.data,
                y_tensor.data,
                None,
            )
        )
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
                check_error(
                    lib.infiniopClip(
                    descriptor,
                    x_tensor.data,
                    y_tensor.data,
                    None,
                )
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"lib time: {elapsed :10f}")
    assert torch.allclose(output, ans, atol=0, rtol=0) if max != None or min != None else torch.allclose(output, x, atol=0, rtol=0)
    check_error(lib.infiniopDestroyClipDescriptor(descriptor))

def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape, min, max, tensor_type in test_cases:
        test(lib, handle, "cpu", x_shape, min, max, tensor_dtype=tensor_type)
    destroy_handle(lib, handle)

def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for x_shape, min, max, tensor_type in test_cases:
        test(lib, handle, "cuda", x_shape, min, max, tensor_dtype=tensor_type)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        ((3, 4), -1, 1, torch.float32),
        ((3, 4), None, 1, torch.float32),
        ((3, 4), -1, None, torch.float32),
        ((3, 4), None, None, torch.float32),
        ((16), -1, 1, torch.float32),
        # ((1024, 1024), -1, 1, torch.float32),
        # ((4096, 4096), -1, 1, torch.float32),
        
        ((13), -1, 1, torch.float32),
        ((3, 4), -1, 1, torch.float16),
        ((3, 4), None, 1, torch.float16),
        ((3, 4), -1, None, torch.float16),
        ((3, 4), None, None, torch.float16),
        ((16), -1, 1, torch.float16),
        # ((1024, 1024), -1, 1, torch.float16),
        # ((4096, 4096), -1, 1, torch.float16),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateClipDescriptor.restype = c_int32
    lib.infiniopCreateClipDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopClipDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t
    ]
    lib.infiniopClip.restype = c_int32
    lib.infiniopClip.argtypes = [
        infiniopClipDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p
    ]
    lib.infiniopDestroyClipDescriptor.restype = c_int32
    lib.infiniopDestroyClipDescriptor.argtypes = [infiniopClipDescriptor_t]
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.cpu:
        test_cpu(lib, test_cases)
    print("All tests passed!")