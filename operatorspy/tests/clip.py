from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p, c_float
import ctypes
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    CTensor,
    DeviceEnum,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    destroy_handle,
    check_error,
    rearrange_tensor,
    create_workspace,
)

from operatorspy.tests.test_utils import get_args, synchronize_device
import torch

PROFILE = True
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

class ClipDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopClipDescriptor_t = POINTER(ClipDescriptor)

def clip(x, lower_bound, upper_bound):
    return torch.clamp_max(torch.clamp_min(x, lower_bound if lower_bound else torch.finfo(x.dtype).min), upper_bound if upper_bound else torch.finfo(x.dtype).max)


def test(
    lib,
    handle,
    torch_device,
    x_shape,
    lower_bound,
    upper_bound,
    dtype=torch.float16,
):
    print(
        f"Testing Clip on {torch_device} with x_shape:{x_shape} lower_bound:{lower_bound} upper_bound:{upper_bound} dtype:{dtype}"
    )

    x = torch.randn(x_shape, dtype=dtype, device=torch_device)
    ans = clip(x, lower_bound, upper_bound)
    y = torch.zeros(ans.shape, dtype=dtype, device=torch_device)


    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib)
        

    descriptor = infiniopClipDescriptor_t()
    check_error(
        lib.infiniopCreateClipDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
            ctypes.byref(c_float(lower_bound)) if lower_bound else None,
            ctypes.byref(c_float(upper_bound)) if upper_bound else None,
        )
    )
    

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()


    check_error(
        lib.infiniopClip(
            descriptor,
            y_tensor.data,
            x_tensor.data,
            None,
        )
    )

    assert torch.allclose(y, ans, atol=0, rtol=0)
    # ans_ = ans.cpu().numpy().flatten()
    # y_ = y.cpu().numpy().flatten()
    # print(ans_)
    # print(y_)
    # atol = max(abs(ans_ - y_))
    # rtol = atol / max(abs(y_) + 1e-8)

    # print(f"atol: {atol}, rtol: {rtol}")

    if PROFILE:
        for i in range(NUM_PRERUN):
            _ = clip(x, lower_bound, upper_bound)
        synchronize_device(torch_device)
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = clip(x, lower_bound, upper_bound)
        synchronize_device(torch_device)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f" pytorch time: {elapsed * 1000 :6f} ms")
        for i in range(NUM_PRERUN):
            check_error(
                lib.infiniopClip(
                    descriptor,
                    y_tensor.data,
                    x_tensor.data,
                    None,
                )
            )
        synchronize_device(torch_device)
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopClip(
                    descriptor,
                    y_tensor.data,
                    x_tensor.data,
                    None,
                )
            )
        synchronize_device(torch_device)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"     lib time: {elapsed * 1000 :6f} ms")

    check_error(lib.infiniopDestroyClipDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)

    for (
        x_shape,
        lower_bound,
        upper_bound,
        dtype,
    ) in test_cases:
        test(
            lib,
            handle,
            "cpu",
            x_shape,
            lower_bound,
            upper_bound,
            dtype,
        )

    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)

    for (
        x_shape,
        lower_bound,
        upper_bound,
        dtype,
    ) in test_cases:
        test(
            lib,
            handle,
            "cuda",
            x_shape,
            lower_bound,
            upper_bound,
            dtype,
        )

    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # x_shape, lower_bound, upper_bound, test_dtype
        ((2, 2), -0.1, 0.1, torch.float32),
        ((2, 2), 0.1, -0.1, torch.float32),
        # ((2, 2), None, None, torch.float32),
        ((2, 2), None, 0.1, torch.float32),
        ((2, 2), 0.1, None, torch.float32),
        ((2048, 2048), -0.1, 0.1, torch.float32),

        ((2, 2), -0.1, 0.1, torch.float16),
        ((2, 2), 0.1, -0.1, torch.float16),
        # ((2, 2), None, None, torch.float16),
        ((2, 2), None, 0.1, torch.float16),
        ((2, 2), 0.1, None, torch.float16),
        ((2048, 2048), -0.1, 0.1, torch.float16),
    ]
    args = get_args()
    lib = open_lib()

    lib.infiniopCreateClipDescriptor.restype = c_int32
    lib.infiniopCreateClipDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopClipDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        POINTER(c_float),
        POINTER(c_float),
    ]


    lib.infiniopClip.restype = c_int32
    lib.infiniopClip.argtypes = [
        infiniopClipDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyClipDescriptor.restype = c_int32
    lib.infiniopDestroyClipDescriptor.argtypes = [
        infiniopClipDescriptor_t,
    ]

    if args.profile:
        PROFILE = True
    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if not (args.cpu or args.cuda):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
