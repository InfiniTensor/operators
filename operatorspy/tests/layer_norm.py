from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p, c_float
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
    create_workspace,
    check_error,
    rearrange_tensor,
)

from operatorspy.tests.test_utils import get_args
import torch
import torch.nn as nn

class LayerNormDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopLayerNormDescriptor_t = POINTER(LayerNormDescriptor)


def LayerNormFunction(input, scale, bias, eps):
    normlize_shape = scale.shape
    layer_norm = nn.LayerNorm(normlize_shape, elementwise_affine=True, eps = eps)
    layer_norm.weight.data = scale
    layer_norm.bias.data = bias
    return layer_norm.forward(input)


def test(lib, handle, torch_device, x_shape, axis, x_dtype=torch.float16):
    print(
        f"Testing Layernorm on {torch_device} with test_shape:{x_shape}, axis:{axis} ,dtype:{x_dtype}"
    )
    eps = 1e-5
    ndim = len(x_shape)
    normlize_shape = []
    for i in range(axis, ndim):
        normlize_shape.append(x_shape[i])

    x = torch.rand(x_shape, dtype=x_dtype).to(torch_device)
    scale = torch.rand(normlize_shape, dtype=x_dtype).to(torch_device)
    bias = torch.rand(normlize_shape, dtype=x_dtype).to(torch_device)
    y = torch.rand(x_shape, dtype=x_dtype).to(torch_device)
    ans = LayerNormFunction(x, scale, bias, eps)
    x_tensor = to_tensor(x, lib)
    w_tensor = to_tensor(scale, lib)
    b_tensor = to_tensor(bias, lib)
    y_tensor = to_tensor(y, lib)
    descriptor = infiniopLayerNormDescriptor_t()
    check_error(
        lib.infiniopCreateLayerNormDescriptor(
            handle, ctypes.byref(descriptor), x_tensor.descriptor, w_tensor.descriptor, b_tensor.descriptor, y_tensor.descriptor, eps
        )
    )
    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetLayerNormWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = create_workspace(workspace_size.value, torch_device) 
    check_error(
        lib.infiniopLayerNorm(
            descriptor,
            workspace.data_ptr() if workspace is not None else None,
            workspace_size.value,
            x_tensor.data,
            w_tensor.data,
            b_tensor.data,
            y_tensor.data,
            None,
        )
    )
    err = y.reshape(-1,1) - ans.reshape(-1,1)
    print(max(abs(err)))
    assert torch.allclose(y, ans, atol=1e-3, rtol=1e-3)
    check_error(lib.infiniopDestroyLayerNormDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape, axis, x_dtype in test_cases:
        test(lib, handle, "cpu", x_shape, axis, x_dtype)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for x_shape, axis, x_dtype in test_cases:
        test(lib, handle, "cuda", x_shape, axis, x_dtype)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for x_shape, axis, x_dtype in test_cases:
        test(lib, handle, "mlu", x_shape, axis, x_dtype)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # x_shape, axis
        # cnnllayernorm不支持axis=0, cpu torch.layernorm不支持half
        #手写layernorm在float16上精度不足，但是在float32上可以通过测试
        #((32, 20, 512), 0, torch.float16),
        ((32, 20, 512), 1, torch.float16), 
        ((32, 20, 512), 2, torch.float16),

        #((32, 20, 512), 0, torch.float32),
        ((32, 20, 512), 1, torch.float32), 
        ((32, 20, 512), 2, torch.float32), 

    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateLayerNormDescriptor.restype = c_int32
    lib.infiniopCreateLayerNormDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopLayerNormDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_float,
    ]

    lib.infiniopLayerNorm.restype = c_int32
    lib.infiniopLayerNorm.argtypes = [
        infiniopLayerNormDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyLayerNormDescriptor.restype = c_int32
    lib.infiniopDestroyLayerNormDescriptor.argtypes = [
        infiniopLayerNormDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)

    if not (args.cpu or args.cuda or args.bang):
        test_cpu(lib, test_cases)
    print("Test passed!")
