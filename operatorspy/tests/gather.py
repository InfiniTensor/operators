from ctypes import POINTER, Structure, c_int32, c_int64, c_void_p
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


class Inplace(Enum):
    OUT_OF_PLACE = auto()


class GatherDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopGatherDescriptor_t = POINTER(GatherDescriptor)


def gather(data, indices, axis):
    np_input = data.numpy()
    np_indices = indices.numpy()
    np_output = np.take(np_input, np_indices, axis=axis)
    return torch.from_numpy(np_output)

def gather(rank, axis, inputTensor, indexTensor):
    indices = [slice(None)] * rank
    indices[axis] = indexTensor
    outTensor = inputTensor[tuple(indices)]
    return outTensor


def test(
    lib,
    handle,
    torch_device,
    o_shape, 
    data_shape, 
    indices_shape,
    axis=0,
    tensor_dtype=torch.float16,
):
    # if inplace != Inplace.OUT_OF_PLACE:
    #     print("Unsupported test: calculatin in-place is not supported")
    #     return
    # if data_shape != indices_shape: 
    #     print("Unsupported test: broadcasting does not support in-place")
    #     return
    # q+r-1
    # if len(o_shape) != len(indices_shape) + len(data_shape) - 1:
    #     print("Unsupported test: output-shape is not valid")
    #     return

    data = torch.rand(data_shape, dtype=tensor_dtype).to(torch_device)
    indices = torch.randint(0, data.shape[axis], data_shape, device=torch_device).to(torch.int32)
    # ans = gather(data, indices, axis)
    ans = gather(len(data_shape), axis, data, indices)
    # output = torch.zeros(o_shape, dtype=tensor_dtype).to(torch_device) 
    output = torch.zeros(ans.shape, dtype=tensor_dtype, device=torch_device)

    print(
        f"Testing Gather on {torch_device} with output_shape:{ans.shape} data_shape:{data_shape} indices_shape:{indices_shape} axis:{axis} dtype:{tensor_dtype}"
    )
    # print(ans.shape, indices_shape, data_shape)    

    descriptor = infiniopGatherDescriptor_t()
    data_tensor = to_tensor(data, lib)
    indices_tensor = to_tensor(indices, lib)
    o_tensor = to_tensor(output, lib) 

    check_error(
        lib.infiniopCreateGatherDescriptor(
            handle,
            ctypes.byref(descriptor),
            o_tensor.descriptor,
            data_tensor.descriptor,
            indices_tensor.descriptor,
            c_int64(axis),
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    o_tensor.descriptor.contents.invalidate()
    data_tensor.descriptor.contents.invalidate()
    indices_tensor.descriptor.contents.invalidate()

    check_error(
        lib.infiniopGather(descriptor, 
                           o_tensor.data, 
                           data_tensor.data, 
                           indices_tensor.data, 
                           None)
    )
    assert torch.allclose(output, ans, atol=0, rtol=0)
    check_error(lib.infiniopDestroyGatherDescriptor(descriptor))


def get_o_shape(input_shape, indices_shape, axis):
    output_shape = input_shape[:axis] + tuple(indices_shape) + input_shape[axis + 1:]
    return output_shape


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for data_shape, indices_shape, axis, _ in test_cases:
        o_shape = get_o_shape(data_shape, indices_shape, axis)
        test(lib, handle, "cpu", o_shape, data_shape, indices_shape, axis, tensor_dtype=torch.float16)
        test(lib, handle, "cpu", o_shape, data_shape, indices_shape, axis, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # q+r-1
        # data_shape, indices_shape, axis, inplace
        # ((), (), (), Inplace.OUT_OF_PLACE),
        ((3, 3), (2, ), 0, Inplace.OUT_OF_PLACE),
        ((32, 64), (10, 4), 1, Inplace.OUT_OF_PLACE), 
        ((4, 3, 4), (2, 2), 1, Inplace.OUT_OF_PLACE),
        ((32, 20, 64), (2, 3), 2, Inplace.OUT_OF_PLACE),
        ((1, 3, 4, 5), (1, 2), 3, Inplace.OUT_OF_PLACE),
        # ((32, 64, 16, 8), (12, 2), 0, Inplace.OUT_OF_PLACE),
        #Cannot allocate memory
    ]

    args = get_args()
    lib = open_lib()
    lib.infiniopCreateGatherDescriptor.restype = c_int32
    lib.infiniopCreateGatherDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopGatherDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_int64,
    ]
    lib.infiniopGather.restype = c_int32
    lib.infiniopGather.argtypes = [
        infiniopGatherDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyGatherDescriptor.restype = c_int32
    lib.infiniopDestroyGatherDescriptor.argtypes = [
        infiniopGatherDescriptor_t,
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
