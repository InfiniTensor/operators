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

class WhereDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopWhereDescriptor_t = POINTER(WhereDescriptor)

def where(condition, x, y):
    return torch.where(condition, x, y)


def test(
    lib,
    handle,
    torch_device,
    x_shape,
    y_shape,
    condition_shape,
    dtype=torch.float16,
):
    print(
        f"Testing Where on {torch_device} with x_shape:{x_shape} y_shape:{y_shape} condition_shape:{condition_shape} dtype:{dtype}"
    )

    x = torch.randn(x_shape, dtype=dtype, device=torch_device)
    y = torch.randn(y_shape, dtype=dtype, device=torch_device)
    condition = torch.randint(0, 2, condition_shape, device=torch_device).to(dtype=torch.uint8)
    ans = where(condition.to(torch.bool), x, y)
    
    output = torch.zeros(ans.shape, dtype=dtype, device=torch_device)
    
    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib)
    condition_tensor = to_tensor(condition, lib)
    output_tensor = to_tensor(output, lib)
        

    descriptor = infiniopWhereDescriptor_t()
    check_error(
        lib.infiniopCreateWhereDescriptor(
            handle,
            ctypes.byref(descriptor),
            output_tensor.descriptor,
            x_tensor.descriptor,
            y_tensor.descriptor,
            condition_tensor.descriptor,
        )
    )
    

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()
    condition_tensor.descriptor.contents.invalidate()
    output_tensor.descriptor.contents.invalidate()


    check_error(
        lib.infiniopWhere(
            descriptor,
            output_tensor.data,
            x_tensor.data,
            y_tensor.data,
            condition_tensor.data,
            None,
        )
    )

    assert torch.allclose(output, ans, atol=0, rtol=0)
    # ans_ = ans.cpu().numpy().flatten()
    # output_ = output.cpu().numpy().flatten()
    # print(ans_)
    # print(output_)
    # atol = max(abs(ans_ - output_))
    # rtol = atol / max(abs(output_) + 1e-8)

    # print(f"atol: {atol}, rtol: {rtol}")

    if PROFILE:
        for i in range(NUM_PRERUN):
            _ = where(condition.to(torch.bool), x, y)
        synchronize_device(torch_device)
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = where(condition.to(torch.bool), x, y)
        synchronize_device(torch_device)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f" pytorch time: {elapsed * 1000 :6f} ms")
        for i in range(NUM_PRERUN):
            check_error(
                lib.infiniopWhere(
                    descriptor,
                    output_tensor.data,
                    x_tensor.data,
                    y_tensor.data,
                    condition_tensor.data,
                    None,
                )
            )
        synchronize_device(torch_device)
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopWhere(
                    descriptor,
                    output_tensor.data,
                    x_tensor.data,
                    y_tensor.data,
                    condition_tensor.data,
                    None,
                )
            )
        synchronize_device(torch_device)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"     lib time: {elapsed * 1000 :6f} ms")

    check_error(lib.infiniopDestroyWhereDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)

    for (
        x_shape,
        y_shape,
        condition_shape,
        dtype,
    ) in test_cases:
        test(
            lib,
            handle,
            "cpu",
            x_shape,
            y_shape,
            condition_shape,
            dtype,
        )

    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)

    for (
        x_shape,
        y_shape,
        condition_shape,
        dtype,
    ) in test_cases:
        test(
            lib,
            handle,
            "cuda",
            x_shape,
            y_shape,
            condition_shape,
            dtype,
        )

    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # x_shape, y_shape, condition_shape, dtype
        ((2, 2), (2, 2), (2, 2), torch.float32),
        ((10,), (10,), (10,), torch.float32),
        ((1,), (2, 2), (2, 2), torch.float32),
        ((2, 2), (1,), (2, 2), torch.float32),
        ((2, 2), (2, 2), (1,), torch.float32),
        ((1, ), (1, ), (2, 2), torch.float32),
        ((1, ), (2, 2), (1, ), torch.float32),
        ((2, 2), (1, ), (1, ), torch.float32),
        ((1024, 1024), (1024, 1024), (1024, 1024), torch.float32),

        ((2, 2), (2, 2), (2, 2), torch.float16),
        ((10,), (10,), (10,), torch.float16),
        ((1,), (2, 2), (2, 2), torch.float16),
        ((2, 2), (1,), (2, 2), torch.float16),
        ((2, 2), (2, 2), (1,), torch.float16),
        ((1, ), (1, ), (2, 2), torch.float16),
        ((1, ), (2, 2), (1, ), torch.float16),
        ((2, 2), (1, ), (1, ), torch.float16),
        ((1024, 1024), (1024, 1024), (1024, 1024), torch.float16),
    ]
    args = get_args()
    lib = open_lib()

    lib.infiniopCreateWhereDescriptor.restype = c_int32
    lib.infiniopCreateWhereDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopWhereDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]


    lib.infiniopWhere.restype = c_int32
    lib.infiniopWhere.argtypes = [
        infiniopWhereDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyWhereDescriptor.restype = c_int32
    lib.infiniopDestroyWhereDescriptor.argtypes = [
        infiniopWhereDescriptor_t,
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
