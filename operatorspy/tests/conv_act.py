from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p, c_double
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


class ConvActDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopConvActDescriptor_t = POINTER(ConvActDescriptor)


def convAct(x, w, bias, stride, padding, dilation, mode):
    ndim = len(x.shape) - 2
    conv_func_map = {
        1: F.conv1d,
        2: F.conv2d,
        3: F.conv3d
    }
    activation_func_map = {
        0: lambda x: x,  # Identity 
        1: F.relu,       # ReLU activation
        2: torch.sigmoid # Sigmoid activation
    }

    if ndim not in conv_func_map:
        print("Error: Pytorch -> Unsupported tensor dimension")
        return None
    
    if mode not in activation_func_map:
        print("Error: Unsupported activation mode")
        return None

    # Select the appropriate convolution function
    conv_func = conv_func_map[ndim]

    if PROFILE:
        ans = conv_func(x, w, bias=bias, stride=stride, padding=padding, dilation=dilation)
        torch.cuda.synchronize()
        return activation_func_map[mode](ans)

    ans = conv_func(x, w, bias=bias, stride=stride, padding=padding, dilation=dilation)
    return activation_func_map[mode](ans)


# infer the shape of the output given the inputs for a N-ary convolution
def inferShape(
    x_shape: List[int],
    w_shape: List[int],
    pads: List[int],
    strides: List[int],
    dilations: List[int],
) -> Tuple[int, ...]:
    assert (
        len(x_shape) == len(w_shape) == len(pads) + 2 == len(dilations) + 2 == len(strides) + 2
    ), "x and w should have the same length; pads, strides, and dilatinos should have the same length; the length of pads should be that of x - 2"
    output_dims = [
        math.floor(
            (x_shape[i+2] + 2 * pads[i] - dilations[i] * (w_shape[i+2] - 1) - 1)
            / strides[i]
            + 1
        )
        for i in range(len(pads))
    ]
    return (x_shape[0], w_shape[0]) + tuple(output_dims)


# convert a python tuple to a ctype void pointer
def tuple_to_void_p(py_tuple: Tuple):
    array = ctypes.c_int64 * len(py_tuple)
    data_array = array(*py_tuple)
    return ctypes.cast(data_array, ctypes.c_void_p)


def test(
    lib,
    handle,
    torch_device,
    x_shape,
    w_shape,
    pads,
    strides,
    dilations,
    add_bias,
    mode,
    clip_coef=0.0,
    tensor_dtype=torch.float16,
):
    assert len(pads) == len(strides) == len(dilations)
    print(
        f"Testing ConvAct on {torch_device} with x_shape: {x_shape}, w_shape: {w_shape}, add_bias: {add_bias} b_shape: {w_shape[0]}, pads: {pads}, strides: {strides}, dilations: {dilations}, mode: {mode}, clip_coef: {clip_coef} dtype:{tensor_dtype}"
    )
    x = torch.rand(x_shape, dtype=tensor_dtype).to(torch_device)
    w = torch.rand(w_shape, dtype=tensor_dtype).to(torch_device)
    b = torch.round((torch.rand(w_shape[0], dtype=tensor_dtype).to(torch_device) * 2 - 1) * 1000) / 1000 if add_bias else None
    y = torch.zeros(
        inferShape(x.shape, w.shape, pads, strides, dilations), dtype=tensor_dtype
    ).to(torch_device)

    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = convAct(x, w, b, strides, pads, dilations, mode)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = convAct(x, w, b, strides, pads, dilations, mode)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :6f}")


    x_tensor = to_tensor(x, lib)
    w_tensor = to_tensor(w, lib)
    b_tensor = to_tensor(b, lib) if b is not None else None
    y_tensor = to_tensor(y, lib)
    descriptor = infiniopConvActDescriptor_t()

    check_error(
        lib.infiniopCreateConvActDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
            w_tensor.descriptor,
            b_tensor.descriptor if b_tensor else None,
            tuple_to_void_p(pads),
            tuple_to_void_p(strides),
            tuple_to_void_p(dilations),
            len(pads),
            mode,
            clip_coef,
        )
    )
    workspaceSize = ctypes.c_uint64(0)
    check_error(
        lib.infiniopGetConvActWorkspaceSize(descriptor, ctypes.byref(workspaceSize))
    )
    workspace = torch.zeros(int(workspaceSize.value), dtype=torch.uint8).to(torch_device)
    workspace_ptr = ctypes.cast(workspace.data_ptr(), ctypes.POINTER(ctypes.c_uint8))

    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(
            lib.infiniopConvAct(
                descriptor,
                workspace_ptr,
                workspaceSize,
                y_tensor.data,
                x_tensor.data,
                w_tensor.data,
                b_tensor.data if b_tensor else None,
                None,
            )
        )
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopConvAct(
                    descriptor,
                    workspace_ptr,
                    workspaceSize,
                    y_tensor.data,
                    x_tensor.data,
                    w_tensor.data,
                    b_tensor.data if b_tensor else None,
                    None,
                )
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"    lib time: {elapsed :6f}")
    
    if (tensor_dtype == torch.float16):
        assert torch.allclose(y, ans, atol=1e-5, rtol=1e-2, equal_nan=True)
    else:
        assert torch.allclose(y, ans, atol=1e-7, rtol=1e-3, equal_nan=True)
    check_error(lib.infiniopDestroyConvActDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape, w_shape, pads, strides, dilations, add_bias, mode in test_cases:
        test(lib, handle, "cpu", x_shape, w_shape, pads, strides, dilations, add_bias, mode, tensor_dtype=torch.float16)
        test(lib, handle, "cpu", x_shape, w_shape, pads, strides, dilations, add_bias, mode, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for x_shape, w_shape, pads, strides, dilations, add_bias, mode in test_cases:
        test(lib, handle, "cuda", x_shape, w_shape, pads, strides, dilations, add_bias, mode, tensor_dtype=torch.float16)
        test(lib, handle, "cuda", x_shape, w_shape, pads, strides, dilations, add_bias, mode, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for x_shape, w_shape, pads, strides, dilations, add_bias, mode in test_cases:
        test(lib, handle, "mlu", x_shape, w_shape, pads, strides, dilations, add_bias, mode, tensor_dtype=torch.float16)
        test(lib, handle, "mlu", x_shape, w_shape, pads, strides, dilations, add_bias, mode, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # x_shape, w_shape, pads, strides, dilations, add_bias, activation_mode
        (
            (2, 2, 4),
            (2, 2, 2),
            (0,),
            (1,),
            (1,),
            True,
            0,
        ),
        (
            (32, 3, 4),
            (32, 3, 5),
            (1,),
            (1,),
            (1,),
            False,
            0,
        ),
        (
            (3, 7, 4),
            (7, 7, 2),
            (0,),
            (1,),
            (1,),
            False,
            0,
        ),
        (
            (1, 3, 4, 4),
            (2, 3, 3, 3),
            (1, 1),
            (1, 2),
            (2, 1),
            True,
            1,
        ),
        (
            (32, 3, 128, 128),
            (64, 3, 5, 5),
            (2, 2),
            (2, 2),
            (1, 1),
            True,
            0,
        ),
        (
            (1, 1, 4, 4, 4),
            (1, 1, 5, 5, 5),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            False,
            1,
        ),
        (
            (3, 3, 32, 32, 32),
            (6, 3, 5, 5, 5),
            (3, 2, 2),
            (4, 3, 3),
            (2, 2, 1),
            True,
            0,
        ),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateConvActDescriptor.restype = c_int32
    lib.infiniopCreateConvActDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopConvActDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_uint64,
        c_int32,
        c_double,
    ]
    lib.infiniopConvAct.restype = c_int32
    lib.infiniopConvAct.argtypes = [
        infiniopConvActDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyConvActDescriptor.restype = c_int32
    lib.infiniopDestroyConvActDescriptor.argtypes = [
        infiniopConvActDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)
    if not (args.cpu or args.cuda or args.bang):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
