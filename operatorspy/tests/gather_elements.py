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

class GatherElementsDescriptor(Structure):
    _fields_ = [("device", c_int32)]

infiniopGatherElementsDescriptor_t = POINTER(GatherElementsDescriptor)

def gather(x, indices, axis = 0):
    return torch.gather(x, axis, indices)

def test(
    lib,
    handle,
    torch_device,
    tensor_shape,
    indices_shape, 
    axis,
    tensor_dtype=torch.float16,
    inplace=Inplace.OUT_OF_PLACE,
):
    print(
        f"Testing Gather on {torch_device} with tensor_shape:{tensor_shape} dtype:{tensor_dtype} axis:{axis} inplace: {inplace.name}"
    )
    data = torch.rand(tensor_shape, dtype=tensor_dtype).to(torch_device) * 2 - 1
    dst = torch.empty(indices_shape, dtype=tensor_dtype).to(torch_device) 
    indices = torch.randint(0, tensor_shape[axis], indices_shape, dtype=torch.int64).to(torch_device)

    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = gather(data, indices, axis)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = gather(data, indices, axis)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :6f}")

    data_tensor = to_tensor(data, lib)
    dst_tensor = to_tensor(dst, lib)
    indices_tensor = to_tensor(indices, lib)
    descriptor = infiniopGatherElementsDescriptor_t()

    check_error(
        lib.infiniopCreateGatherElementsDescriptor(
            handle,
            ctypes.byref(descriptor),
            dst_tensor.descriptor,
            data_tensor.descriptor,
            indices_tensor.descriptor,
            axis,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    dst_tensor.descriptor.contents.invalidate()
    data_tensor.descriptor.contents.invalidate()
    indices_tensor.descriptor.contents.invalidate()

    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(lib.infiniopGatherElements(descriptor, dst_tensor.data, data_tensor.data, indices_tensor.data, None))
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopGatherElements(descriptor, dst_tensor.data, data_tensor.data, indices_tensor.data, None)
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"    lib time: {elapsed :6f}")

    assert torch.allclose(dst, ans, atol=0, rtol=0)
    check_error(lib.infiniopDestroyGatherElementsDescriptor(descriptor))

def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for tensor_shape, dst_shape, axis, inplace in test_cases:
        test(lib, handle, "cpu", tensor_shape, dst_shape, axis, tensor_dtype=torch.float32, inplace=inplace)
        test(lib, handle, "cpu", tensor_shape, dst_shape, axis, tensor_dtype=torch.float16, inplace=inplace)
    destroy_handle(lib, handle)

def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for tensor_shape, dst_shape, axis, inplace in test_cases:
        test(lib, handle, "cuda", tensor_shape, dst_shape, axis, tensor_dtype=torch.float32, inplace=inplace)
        test(lib, handle, "cuda", tensor_shape, dst_shape, axis, tensor_dtype=torch.float16, inplace=inplace)
    destroy_handle(lib, handle)

if __name__ == "__main__":
    test_cases = [
        # tensor_shape, inplace
        ((3, 2), (2, 2), 0, Inplace.OUT_OF_PLACE),
        ((3, 2), (3, 2), 1, Inplace.OUT_OF_PLACE),
        ((33, 333, 333), (33, 200, 333), 1, Inplace.OUT_OF_PLACE),
        ((3, 2, 2), (3, 2, 1), -1, Inplace.OUT_OF_PLACE),
        ((2, 3, 4), (2, 2, 4), 1, Inplace.OUT_OF_PLACE),
        ((32, 256, 112, 112), (32, 256, 112, 112), 2, Inplace.OUT_OF_PLACE),

    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateGatherElementsDescriptor.restype = c_int32
    lib.infiniopCreateGatherElementsDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopGatherElementsDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_int32,
    ]
    lib.infiniopGatherElements.restype = c_int32
    lib.infiniopGatherElements.argtypes = [
        infiniopGatherElementsDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyGatherElementsDescriptor.restype = c_int32
    lib.infiniopDestroyGatherElementsDescriptor.argtypes = [
        infiniopGatherElementsDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)

    if args.cuda:
        test_cuda(lib, test_cases)

    print("\033[92mTest passed!\033[0m")