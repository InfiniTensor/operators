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
    INPLACE_ = auto()


class ClipDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopClipDescriptor_t = POINTER(ClipDescriptor)

def clip(x, min_value, max_value):
    return torch.clip(x, min=min_value, max=max_value)


def test(
    lib,
    handle,
    torch_device,
    o_shape, 
    i_shape,
    min_value=None,
    max_value=None,
    tensor_dtype=torch.float16,
    inplace=Inplace.OUT_OF_PLACE,
):
    print(
        f"Testing Clip on {torch_device} with o_shape:{o_shape} i_shape:{i_shape} \
        min_value:{min_value} max_value:{max_value} \
        dtype:{tensor_dtype} inplace: {inplace.name}"
    )
    if o_shape != i_shape:
        print("Unsupported test: unmatched shapes for input and output")
        return

    input = torch.rand(i_shape, dtype=tensor_dtype).to(torch_device)
    output = torch.rand(o_shape, dtype=tensor_dtype).to(torch_device) if inplace == Inplace.OUT_OF_PLACE else input

    if min_value != None:
        min_value = torch.tensor(min_value, dtype=tensor_dtype, device=torch_device)
    else:
        min_value = torch.tensor(float("-inf"), dtype=tensor_dtype, device=torch_device)
    if max_value != None:
        max_value = torch.tensor(max_value, dtype=tensor_dtype, device=torch_device)
    else:
        max_value = torch.tensor(float("inf"), dtype=tensor_dtype, device=torch_device)

    descriptor = infiniopClipDescriptor_t()
    i_tensor = to_tensor(input, lib)
    o_tensor = to_tensor(output, lib) if inplace == Inplace.OUT_OF_PLACE else (i_tensor)

    ans = clip(input, min_value, max_value)

    check_error(
        lib.infiniopCreateClipDescriptor(
            handle,
            ctypes.byref(descriptor),
            o_tensor.descriptor,
            i_tensor.descriptor,
            min_value.item(),
            max_value.item(),
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    o_tensor.descriptor.contents.invalidate()
    i_tensor.descriptor.contents.invalidate()

    check_error(
        lib.infiniopClip(descriptor, 
                         o_tensor.data, 
                         i_tensor.data, 
                         None)
    )
    # print(f"  min:{min_value}, max:{max_value}, input:{input}, ans:{ans}, output:{output},")
    assert torch.allclose(output, ans, atol=0, rtol=0)
    check_error(lib.infiniopDestroyClipDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for o_shape, i_shape, min_value, max_value, inplace in test_cases:
        test(lib, handle, "cpu", o_shape, i_shape, min_value, max_value, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "cpu", o_shape, i_shape, min_value, max_value, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # o_shape, i_shape, min, max, inplace
        ((1, 3), (1, 3), None, None, Inplace.OUT_OF_PLACE),
        ((3, 3), (3, 3), 1.0, 2.0, Inplace.OUT_OF_PLACE),
        ((), (), None, None, Inplace.OUT_OF_PLACE),
        ((2, 20, 3), (2, 20, 3), 0., -2.0, Inplace.INPLACE_),

        ((32, 20, 512), (32, 20, 512), -0.3, 0.3, Inplace.INPLACE_),
        ((32, 256, 112, 112), (32, 256, 112, 112), -0.1, 0.1, Inplace.OUT_OF_PLACE),
        # ((32, 150, 5120), (32, 150, 5120), None, None, Inplace.OUT_OF_PLACE),

        ((2, 4, 3), (2, 1, 3), None, None, Inplace.OUT_OF_PLACE),
        ((2, 3, 4, 5), (2, 3, 4, 5), -0.1, 0.1, Inplace.OUT_OF_PLACE),
        ((3, 2, 4, 5), (4, 5), None, None, Inplace.OUT_OF_PLACE),
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
        # 
        c_void_p,
    ]
    lib.infiniopDestroyClipDescriptor.restype = c_int32
    lib.infiniopDestroyClipDescriptor.argtypes = [
        infiniopClipDescriptor_t,
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
