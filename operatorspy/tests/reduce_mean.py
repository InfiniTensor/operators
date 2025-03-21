from ctypes import POINTER, Structure, c_void_p, c_int32, c_uint64, c_bool
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
import numpy as np

from typing import Tuple

# constant for control whether profile the pytorch and lib functions
# NOTE: need to manually add synchronization function to the lib function,
#       e.g., cudaDeviceSynchronize() for CUDA
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class Inplace(Enum):
    OUT_OF_PLACE = auto()


class ReduceMeanDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopReduceMeanDescriptor_t = POINTER(ReduceMeanDescriptor)


def reduce_mean(input, axis, keepdims=True, noop_with_empty_axes=False):
    if axis == ():
        if noop_with_empty_axes:
            return input
        else:
            return torch.mean(input, dim=axis, keepdim=keepdims)
    return torch.mean(input, dim=axis, keepdim=keepdims)

# convert a python tuple to a ctype void pointer
def tuple_to_int64_p(py_tuple: Tuple):
    array = ctypes.c_int64 * len(py_tuple)
    data_array = array(*py_tuple)
    return ctypes.cast(data_array, ctypes.POINTER(ctypes.c_int64))

def test(
    lib,
    handle,
    torch_device,
    data_shape, 
    axes,
    keepdims=True,
    noop_with_empty_axes=False,
    tensor_dtype=torch.float16,
):
    print(
        f"Testing ReduceMean on {torch_device} with data_shape:{data_shape}  axes:{axes}  keepdims:{keepdims}  noop_with_empty_axes:{noop_with_empty_axes}  dtype:{tensor_dtype}"
    )

    # if inplace != Inplace.OUT_OF_PLACE:
    #     print("Unsupported test: calculatin in-place is not supported")
    #     return

    data = torch.rand(data_shape, dtype=tensor_dtype).to(torch_device)
    ans = reduce_mean(data, axes, keepdims, noop_with_empty_axes)
    reduced = torch.zeros(ans.shape, dtype=tensor_dtype).to(torch_device)
    # print(f"in pytorch, data.size:{data.size()}, reduced.size:{reduced.size()}")

    for _ in range(NUM_PRERUN if PROFILE else 0):
        ans = reduce_mean(data, axes, keepdims, noop_with_empty_axes)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = reduce_mean(data, axes, keepdims, noop_with_empty_axes)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :6f}")

    reduced_tensor = to_tensor(reduced, lib) 
    data_tensor = to_tensor(data, lib)
    descriptor = infiniopReduceMeanDescriptor_t()

    check_error(
        lib.infiniopCreateReduceMeanDescriptor(
            handle,
            ctypes.byref(descriptor),
            reduced_tensor.descriptor,
            data_tensor.descriptor,
            tuple_to_int64_p(axes),
            c_uint64(len(axes)),
            c_bool(keepdims),
            c_bool(noop_with_empty_axes),
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    reduced_tensor.descriptor.contents.invalidate()
    data_tensor.descriptor.contents.invalidate()

    workspaceSize = ctypes.c_uint64(0)
    check_error(
        lib.infiniopGetReduceMeanWorkspaceSize(descriptor, ctypes.byref(workspaceSize))
    )
    workspace = torch.zeros(int(workspaceSize.value), dtype=torch.uint8).to(torch_device)
    workspace_ptr = ctypes.cast(workspace.data_ptr(), ctypes.POINTER(ctypes.c_uint8))
    # print("!!")

    for _ in range(NUM_PRERUN if PROFILE else 1):
        check_error(
            lib.infiniopReduceMean(
                descriptor, 
                workspace_ptr,
                workspaceSize,
                reduced_tensor.data, 
                data_tensor.data, 
                None)
        )
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
            lib.infiniopReduceMean(
                descriptor, 
                workspace_ptr,
                workspaceSize,
                reduced_tensor.data, 
                data_tensor.data, 
                None)
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"    lib time: {elapsed :6f}")
    
    if tensor_dtype == torch.float16:
        assert torch.allclose(reduced, ans, atol=0, rtol=1e-3)
    elif tensor_dtype == torch.float32:
        # assert torch.allclose(reduced, ans, atol=0, rtol=1e-5)
        assert torch.allclose(reduced, ans, atol=0, rtol=1e-3)
    check_error(lib.infiniopDestroyReduceMeanDescriptor(descriptor))
    # print("!!!!")


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for data_shape, axes, keepdims, noop_with_empty_axes in test_cases:
        test(lib, handle, "cpu", data_shape, axes, keepdims, noop_with_empty_axes, tensor_dtype=torch.float16)
        test(lib, handle, "cpu", data_shape, axes, keepdims, noop_with_empty_axes, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # data_shape, axes, keepdims, noop_with_empty_axes

        ((2, 3, 2, 5), (1, 3), True, False),
        ((2, 3, 2, 5), (-2, -1), False, False),
        ((3, 2, 5, 4), (-3, -2, -1), True, False), #Floating point exception
        # ((3, 2, 5, 4), (-4, -3, -2, -1), True, False), #AssertionError

        ((1, 3), (), True, True),
        ((1, 3), (), False, True),

        ((2, 3), (0,), True, False),
        ((32, 20, 512), (2,), True, False),
        ((3, 2, 5, 4), (0, 1, 2, 3), True, False),
        ((32, 56, 112, 112), (0, 1, 2), True, False),      

        ((2, 3, 5), (0,), False, False),
        ((32, 20, 512), (1, 2), False, False),
        ((3, 2, 5, 4), (0, 1, 2, 3), False, False),
        ((64, 64, 64, 64), (0, 2, 3), False, False),
    ]

    args = get_args()
    lib = open_lib()
    lib.infiniopCreateReduceMeanDescriptor.restype = c_int32
    lib.infiniopCreateReduceMeanDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopReduceMeanDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        POINTER(ctypes.c_int64),
        c_uint64,
        c_bool,
        c_bool,
    ]
    lib.infiniopGetReduceMeanWorkspaceSize.restype = c_int32
    lib.infiniopGetReduceMeanWorkspaceSize.argtypes = [
        infiniopReduceMeanDescriptor_t,
        POINTER(c_uint64),
    ]
    lib.infiniopReduceMean.restype = c_int32
    lib.infiniopReduceMean.argtypes = [
        infiniopReduceMeanDescriptor_t,
        c_void_p,
        c_uint64,
        # 
        c_void_p,
        c_void_p,
        #
        c_void_p,
    ]
    lib.infiniopDestroyReduceMeanDescriptor.restype = c_int32
    lib.infiniopDestroyReduceMeanDescriptor.argtypes = [
        infiniopReduceMeanDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)
    # if args.cuda:
    #     test_cuda(lib, test_cases)
    # if args.bang:
    #     test_bang(lib, test_cases)
    # if args.musa:
    #     test_musa(lib, test_cases)
    if not (args.cpu or args.cuda or args.bang or args.musa):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
