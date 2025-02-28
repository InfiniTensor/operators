
from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p
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
import math
import ctypes
from torch.nn import functional as F
from typing import List, Tuple

# constant for control whether profile the pytorch and lib functions
# NOTE: need to manually add synchronization function to the lib function,
#       e.g., cudaDeviceSynchronize() for CUDA
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class ReduceMeanDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopReduceMeanDescriptor_t = POINTER(ReduceMeanDescriptor)

def reduce_mean(x, axes, keepdim=False):
    return torch.mean(x, dim=axes, keepdim=keepdim)

# convert a python tuple to a ctype void pointer
def tuple_to_int_p(py_tuple: Tuple):
    array = ctypes.c_int * len(py_tuple)
    data_array = array(*py_tuple)
    return ctypes.cast(data_array, ctypes.POINTER(ctypes.c_int))


def test(
    lib,
    handle,
    torch_device,
    x_shape,
    axes,
    keepdim,
    tensor_dtype=torch.float16,
):
    print(
        f"Testing ReduceMean on {torch_device} with x_shape: {x_shape}, axes:{axes}, keepdim:{keepdim}, dtype:{tensor_dtype}"
    )
    x = torch.rand(x_shape, dtype=tensor_dtype).to(torch_device)
    ans = reduce_mean(x, axes, keepdim)
    y = torch.zeros(ans.shape, dtype=tensor_dtype).to(torch_device)
    
    # print(f'y_shape: {y.shape}')

    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = reduce_mean(x, axes, keepdim)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = reduce_mean(x, axes, keepdim)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :8f}")
        

    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib)
    descriptor = infiniopReduceMeanDescriptor_t()
    
    # print("!")

    check_error(
        lib.infiniopCreateReduceMeanDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
            tuple_to_int_p(axes),
            len(axes)
        )
    )
    # print("!")

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()

    workspaceSize = ctypes.c_uint64(0)
    check_error(
        lib.infiniopGetReduceMeanWorkspaceSize(descriptor, ctypes.byref(workspaceSize))
    )
    workspace = torch.zeros(int(workspaceSize.value), dtype=torch.uint8).to(torch_device)
    workspace_ptr = ctypes.cast(workspace.data_ptr(), ctypes.POINTER(ctypes.c_uint8))


    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(
            lib.infiniopReduceMean(
                descriptor,
                workspace_ptr,
                workspaceSize,
                y_tensor.data,
                x_tensor.data,
                None,
            )
        )
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopReduceMean(
                    descriptor,
                    workspace_ptr,
                    workspaceSize,
                    y_tensor.data,
                    x_tensor.data,
                    None,
                )
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"    lib time: {elapsed :8f}")

    if(tensor_dtype == torch.float16):
        assert torch.allclose(y, ans, atol=0, rtol=1e-3)
    else:
        assert torch.allclose(y, ans, atol=0, rtol=1e-5)
    # ans_ = ans.cpu().numpy().flatten()
    # y_ = y.cpu().numpy().flatten()
    # atol = max(abs(ans_ - y_))
    # rtol = atol / max(abs(y_) + 1e-8)
    # # print(f"ans: {ans_}")
    # # print(f"y: {y_}")

    # print(f"atol: {atol}, rtol: {rtol}")
    check_error(lib.infiniopDestroyReduceMeanDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape, axes, keepdim, tensor_dtype in test_cases:
        test(lib, handle, "cpu", x_shape, axes, keepdim, tensor_dtype)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for x_shape, axes, keepdim, tensor_dtype in test_cases:
        test(lib, handle, "cuda", x_shape, axes, keepdim, tensor_dtype)
    destroy_handle(lib, handle)



if __name__ == "__main__":
    test_cases = [
        # x_shape, axes, keepdim, dtype
        ((2, 2, 3, 4), (0,), False, torch.float32),
        ((2, 2, 3, 4), (0,), True, torch.float32),
        ((2, 2, 3, 4), (1,), False, torch.float32),
        ((2, 2, 3, 4), (1, 2), False, torch.float32),
        ((2, 2, 3, 4), (1, 3), False, torch.float32),
        ((2, 2, 3, 4), (0, 1, 2, 3), False, torch.float32),
        ((2, 2, 3, 4, 5), (0, 3), False, torch.float32),
        ((64, 64, 64, 64), (0, 2), False, torch.float32),
        
        ((2, 2, 3, 4), (0,), False, torch.float16),
        ((2, 2, 3, 4), (0,), True, torch.float16),
        ((2, 2, 3, 4), (1,), False, torch.float16),
        ((2, 2, 3, 4), (1, 2), False, torch.float16),
        ((2, 2, 3, 4), (1, 3), False, torch.float16),
        ((2, 2, 3, 4), (0, 1, 2, 3), False, torch.float16),
        ((2, 2, 3, 4, 5), (0, 3), False, torch.float16),
        ((64, 64, 64, 64), (0, 2), False, torch.float16),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateReduceMeanDescriptor.restype = c_int32
    lib.infiniopCreateReduceMeanDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopReduceMeanDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        POINTER(ctypes.c_int),
        c_uint64,
    ]
    lib.infiniopReduceMean.restype = c_int32
    lib.infiniopReduceMean.argtypes = [
        infiniopReduceMeanDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyReduceMeanDescriptor.restype = c_int32
    lib.infiniopDestroyReduceMeanDescriptor.argtypes = [
        infiniopReduceMeanDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if not (args.cpu or args.cuda):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
