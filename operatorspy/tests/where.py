from ctypes import POINTER, Structure, c_int32, c_void_p, c_uint64, c_bool
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

PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

class WhereDescriptor(Structure):
    _fields_ = [("device", c_int32)]

infiniopWhereDescriptor_t = POINTER(WhereDescriptor)

def where(condition, x, y):
    return torch.where(condition, x, y)


def tuple_to_void_p(py_tuple: Tuple):
    array = ctypes.c_int64 * len(py_tuple)
    data_array = array(*py_tuple)
    return ctypes.cast(data_array, ctypes.c_void_p)

def inferShape(x_shape, y_shape):
    ndim_x = len(x_shape)
    ndim_y = len(y_shape)
    ndim = max(ndim_x, ndim_y)
    output_shape = []
    
    for i in range(-1, -ndim-1, -1):
        dim_x = x_shape[i] if i >= -ndim_x else 1
        dim_y = y_shape[i] if i >= -ndim_y else 1
        
        if dim_x != dim_y:
            if dim_x != 1 and dim_y != 1:
                raise ValueError(f"Shapes {x_shape} and {y_shape} cannot be broadcast together")
        
        output_dim = max(dim_x, dim_y)
        output_shape.insert(0, output_dim)
    
    return tuple(output_shape)

    
def test(
    lib,
    handle,
    torch_device,
    condition_shape,
    src1_shape,
    src2_shape,
    tensor_dtype=torch.float16
):
    print(
        f"Testing where on {torch_device} with condition_shape:{condition_shape} dtype:{tensor_dtype}"
    )
    condition = torch.randint(0, 2, condition_shape, dtype=torch.uint8).to(torch_device)
    src1 = torch.randn(src1_shape, dtype=tensor_dtype, device=torch_device)
    src2 = torch.randn(src2_shape, dtype=tensor_dtype, device=torch_device)
    output = torch.randn(inferShape(inferShape(src1_shape, src2_shape), condition_shape), dtype=tensor_dtype, device=torch_device)

    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = where(condition, src1, src2)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = where(condition, src1, src2)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :10f}")
    src1_tensor = to_tensor(src1, lib)
    src2_tensor = to_tensor(src2, lib)
    output_tensor = to_tensor(output, lib)
    condition_tensor = to_tensor(condition, lib)
    descriptor = infiniopWhereDescriptor_t()
    check_error(
        lib.infiniopCreateWhereDescriptor(
            handle,
            ctypes.byref(descriptor),
            output_tensor.descriptor,
            src1_tensor.descriptor,
            src2_tensor.descriptor,
            condition_tensor.descriptor,
        )
    )
    src1_tensor.descriptor.contents.invalidate()
    src2_tensor.descriptor.contents.invalidate()
    output_tensor.descriptor.contents.invalidate()
    condition_tensor.descriptor.contents.invalidate()
    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(
            lib.infiniopWhere(
                descriptor,
                output_tensor.data,
                src1_tensor.data,
                src2_tensor.data,
                condition_tensor.data,
                None,
            )
        )
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
                check_error(
                    lib.infiniopWhere(
                    descriptor,
                    output_tensor.data,
                    src1_tensor.data,
                    src2_tensor.data,
                    condition_tensor.data,
                    None,
                )
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"lib time: {elapsed :10f}")
    assert torch.allclose(output, ans, atol=0, rtol=0)
    check_error(lib.infiniopDestroyWhereDescriptor(descriptor))
    
def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for condition_shape, src1_shape, src2_shape, tensor_dtype in test_cases:
        test(lib, handle, "cpu", condition_shape, src1_shape, src2_shape, tensor_dtype=tensor_dtype)
        print("\n")
    destroy_handle(lib, handle)

def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for condition_shape, src1_shape, src2_shape, tensor_dtype in test_cases:
        test(lib, handle, "cuda", condition_shape, src1_shape, src2_shape, tensor_dtype=tensor_dtype)
        print("\n")
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        ((2, 16), (2, 16), (2, 16), torch.float32),
        ((2, 3, 1, 1), (1, 4, 5), (2, 3, 4, 5), torch.float32),
        ((3, 1), (3, 4), (1, 4), torch.float32),
        ((1,), (3, 4), (3, 4), torch.float32),
        ((2, 1, 3), (1, 4, 3), (2, 4, 1), torch.float32),

        ((2, 16), (2, 16), (2, 16), torch.float16),
        ((2, 3, 1, 1), (1, 4, 5), (2, 3, 4, 5), torch.float16),
        ((3, 1), (3, 4), (1, 4), torch.float16),
        ((1,), (3, 4), (3, 4), torch.float16),
        ((2, 1, 3), (1, 4, 3), (2, 4, 1), torch.float16),
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
        infiniopTensorDescriptor_t
    ]
    lib.infiniopWhere.restype = c_int32
    lib.infiniopWhere.argtypes = [
        infiniopWhereDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p
    ]
    lib.infiniopDestroyWhereDescriptor.restype = c_int32
    lib.infiniopDestroyWhereDescriptor.argtypes = [infiniopWhereDescriptor_t]
    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    print("All tests passed!")