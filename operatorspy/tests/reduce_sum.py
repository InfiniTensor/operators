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

class ReduceSumDescriptor(Structure):
    _fields_ = [("device", c_int32)]

infiniopReduceSumDescriptor_t = POINTER(ReduceSumDescriptor)

def reduceSum(x, axis = 0, keepdims = True):
    return torch.sum(x, dim=axis, keepdim=keepdims)

def test(
    lib,
    handle,
    torch_device,
    tensor_shape, 
    axis,
    keepdims,
    tensor_dtype=torch.float16,
    inplace=Inplace.OUT_OF_PLACE,
):
    print(
        f"Testing ReduceSum on {torch_device} with tensor_shape:{tensor_shape} dtype:{tensor_dtype} axis:{axis} keepdim:{keepdims} inplace: {inplace.name}"
    )  
    x = torch.rand(tensor_shape, dtype=tensor_dtype).to(torch_device) * 2 - 1

    out_shape = list(tensor_shape)
    for i in axis:
        if(keepdims==1):
            out_shape[i] = 1
        else:
            out_shape = out_shape[:i] + out_shape[i+1:]
    if(len(axis) == len(tensor_shape)):
        if(keepdims == 0):
            out_shape = [1]

    y = torch.zeros(out_shape, dtype=tensor_dtype).to(torch_device)


    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = reduceSum(x, axis, bool(keepdims))

    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = reduceSum(x, axis, bool(keepdims))
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :6f}")

    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib)
    descriptor = infiniopReduceSumDescriptor_t()

    check_error(
        lib.infiniopCreateReduceSumDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
            (ctypes.c_int * len(axis))(*axis),
            len(axis),
            keepdims,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()

    workspaceSize = ctypes.c_uint64(0)
    check_error(
        lib.infiniopGetReduceSumWorkspaceSize(descriptor, ctypes.byref(workspaceSize))
    )
    workspace = torch.zeros(int(workspaceSize.value), dtype=torch.uint8).to(torch_device)
    workspace_ptr = ctypes.cast(workspace.data_ptr(), ctypes.POINTER(ctypes.c_uint8))

    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(lib.infiniopReduceSum(descriptor, workspace_ptr, workspaceSize, y_tensor.data, x_tensor.data, None))

    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopReduceSum(descriptor, workspace_ptr, workspaceSize, y_tensor.data, x_tensor.data, None)
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"    lib time: {elapsed :6f}")

    if tensor_dtype == torch.float16:
        assert torch.allclose(y, ans, atol=1e-3, rtol=1e-7)
    else:
        assert torch.allclose(y, ans, atol=1e-6, rtol=1e-6)
    check_error(lib.infiniopDestroyReduceSumDescriptor(descriptor))

def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for tensor_shape, axis, keepdims, inplace in test_cases:
        test(lib, handle, "cpu", tensor_shape, axis, keepdims, tensor_dtype=torch.float32, inplace=inplace)
        test(lib, handle, "cpu", tensor_shape, axis, keepdims, tensor_dtype=torch.float16, inplace=inplace)
    destroy_handle(lib, handle)

def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for tensor_shape, axis, keepdims, inplace in test_cases:
        test(lib, handle, "cuda", tensor_shape, axis, keepdims, tensor_dtype=torch.float32, inplace=inplace)
        test(lib, handle, "cuda", tensor_shape, axis, keepdims, tensor_dtype=torch.float16, inplace=inplace)
    destroy_handle(lib, handle)
    
if __name__ == "__main__":
    test_cases = [
        # tensor_shape, inplace
        ((2, 3), [1], 1, Inplace.OUT_OF_PLACE),
        ((2, 3), [0, 1], 0, Inplace.OUT_OF_PLACE),
        ((4, 7), [0], 1, Inplace.OUT_OF_PLACE),
        ((300, 1024), [0], 1, Inplace.OUT_OF_PLACE),
        ((2, 3, 4, 5), [-2, -1], 1, Inplace.OUT_OF_PLACE),
        ((6, 6, 200), [0, 1], 1, Inplace.OUT_OF_PLACE),
        ((6, 2, 7), [0, 2], 1, Inplace.OUT_OF_PLACE),
        ((266, 400), [0], 0, Inplace.OUT_OF_PLACE),
        ((32, 256, 112, 112), [3], 1, Inplace.OUT_OF_PLACE),
        ((32, 288, 112, 112), [3], 0, Inplace.OUT_OF_PLACE),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateReduceSumDescriptor.restype = c_int32
    lib.infiniopCreateReduceSumDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopReduceSumDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        ctypes.POINTER(ctypes.c_int),
        c_int32,
        c_int32,
    ]
    lib.infiniopGetReduceSumWorkspaceSize.restype = c_int32
    lib.infiniopGetReduceSumWorkspaceSize.argtypes = [
        infiniopReduceSumDescriptor_t,
        POINTER(c_uint64),
    ]
    lib.infiniopReduceSum.restype = c_int32
    lib.infiniopReduceSum.argtypes = [
        infiniopReduceSumDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyReduceSumDescriptor.restype = c_int32
    lib.infiniopDestroyReduceSumDescriptor.argtypes = [
        infiniopReduceSumDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)

    if args.cuda:
        test_cuda(lib, test_cases)

    print("\033[92mTest passed!\033[0m")