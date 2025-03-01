from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p
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


class SoftmaxDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopSoftmaxDescriptor_t = POINTER(SoftmaxDescriptor)


def softmax(x, axis):
    return torch.softmax(x, axis = axis).to(x.dtype)


def test(lib, handle, torch_device, x_shape, axis, x_dtype=torch.float16):
    print(
        f"Testing Softmax on {torch_device} with x_shape:{x_shape} , axis:{axis} ,dtype:{x_dtype}"
    )
    x = torch.rand(x_shape, dtype=x_dtype).to(torch_device)
    y = torch.rand(x_shape, dtype=x_dtype).to(torch_device)
    ans = softmax(x, axis)
    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib)
    descriptor = infiniopSoftmaxDescriptor_t()
    check_error(
        lib.infiniopCreateSoftmaxDescriptor(
            handle, ctypes.byref(descriptor), x_tensor.descriptor, axis, y_tensor.descriptor
        )
    )
    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetSoftmaxWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = create_workspace(workspace_size.value, torch_device) 
    check_error(
        lib.infiniopSoftmax(
            descriptor,
            workspace.data_ptr() if workspace is not None else None,
            workspace_size.value,
            x_tensor.data,
            y_tensor.data,
            None,
        )
    )
    err = y.reshape(-1,1) - ans.reshape(-1,1)
    print(max(abs(err)))
    assert torch.allclose(y, ans, atol=0, rtol=1e-2)
    check_error(lib.infiniopDestroySoftmaxDescriptor(descriptor))


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
        # 寒武纪芯片的国产CPU可能不支持f16
        ((32, 20, 512), 0, torch.float16),
        ((32, 20, 512), 1, torch.float16), 
        ((32, 20, 512), 2, torch.float16),
        
        ((32, 20, 512), 0, torch.float32),
        ((32, 20, 512), 1, torch.float32), 
        ((32, 20, 512), 2, torch.float32), 
         
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateSoftmaxDescriptor.restype = c_int32
    lib.infiniopCreateSoftmaxDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopSoftmaxDescriptor_t),
        infiniopTensorDescriptor_t,
    ]
    
    lib.infiniopSoftmax.restype = c_int32
    lib.infiniopSoftmax.argtypes = [
        infiniopSoftmaxDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroySoftmaxDescriptor.restype = c_int32
    lib.infiniopDestroySoftmaxDescriptor.argtypes = [
        infiniopSoftmaxDescriptor_t,
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
