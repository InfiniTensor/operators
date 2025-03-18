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

class GatherDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopGatherDescriptor_t = POINTER(GatherDescriptor)

def gather(rank, axis, inputTensor, indexTensor):
    indices = [slice(None)] * rank
    indices[axis] = indexTensor
    outTensor = inputTensor[tuple(indices)]
    return outTensor


def test(
    lib,
    handle,
    torch_device,
    input_shape,
    index_shape,
    axis,
    dtype=torch.float16,
):
    print(
        f"Testing Gather on {torch_device} with input_shape:{input_shape} indices_shape:{index_shape} axis:{axis} dtype:{dtype}"
    )

    input = torch.randn(input_shape, dtype=dtype, device=torch_device)
    index = torch.randint(0, input.shape[axis], index_shape, device=torch_device).to(torch.int32)
    ans = gather(len(input_shape), axis, input, index)
    output = torch.zeros(ans.shape, dtype=dtype, device=torch_device)


    input_tensor = to_tensor(input, lib)
    index_tensor = to_tensor(index, lib)
    output_tensor = to_tensor(output, lib)

    descriptor = infiniopGatherDescriptor_t()
    check_error(
        lib.infiniopCreateGatherDescriptor(
            handle,
            ctypes.byref(descriptor),
            output_tensor.descriptor,
            input_tensor.descriptor,
            index_tensor.descriptor,
            axis,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    input_tensor.descriptor.contents.invalidate()
    index_tensor.descriptor.contents.invalidate()
    output_tensor.descriptor.contents.invalidate()


    check_error(
        lib.infiniopGather(
            descriptor,
            output_tensor.data,
            input_tensor.data,
            index_tensor.data,
            None,
        )
    )

    assert torch.allclose(output, ans, atol=0, rtol=0)
    # ans_ = ans.cpu().numpy().flatten()
    # output_ = output.cpu().numpy().flatten()
    # atol = max(abs(ans_ - output_))
    # rtol = atol / max(abs(output_) + 1e-8)

    # print(f"atol: {atol}, rtol: {rtol}")

    if PROFILE:
        for i in range(NUM_PRERUN):
            _ = gather(len(input_shape), axis, input, index)
        synchronize_device(torch_device)
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = gather(len(input_shape), axis, input, index)
        synchronize_device(torch_device)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f" pytorch time: {elapsed * 1000 :6f} ms")
        for i in range(NUM_PRERUN):
            check_error(
                lib.infiniopGather(
                    descriptor,
                    output_tensor.data,
                    input_tensor.data,
                    index_tensor.data,
                    None,
                )
            )
        synchronize_device(torch_device)
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopGather(
                    descriptor,
                    output_tensor.data,
                    input_tensor.data,
                    index_tensor.data,
                    None,
                )
            )
        synchronize_device(torch_device)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"     lib time: {elapsed * 1000 :6f} ms")

    check_error(lib.infiniopDestroyGatherDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)

    for (
        input_shape,
        index_shape,
        axis,
        dtype,
    ) in test_cases:
        test(
            lib,
            handle,
            "cpu",
            input_shape,
            index_shape,
            axis,
            dtype,
        )

    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)

    for (
        input_shape,
        index_shape,
        axis,
        dtype,
    ) in test_cases:
        test(
            lib,
            handle,
            "cuda",
            input_shape,
            index_shape,
            axis,
            dtype,
        )

    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # input_shape , index_shape, axis, test_dtype
        ((64, 64), (64, 64), 0, torch.float32),
        ((64, 64), (64, 64), 1, torch.float32),
        ((8, 8, 8, 8, 8), (8, 8), 0, torch.float32),
        ((8, 8, 8, 8, 8), (8, 8), 2, torch.float32),
        ((1024, 1024, 1024), (1, ), 1, torch.float32),
        ((2048, 2048), (128, 128), 0, torch.float32),
        ((2048, 2048), (128, 128), 1, torch.float32),

        ((64, 64), (64, 64), 0, torch.float16),
        ((64, 64), (64, 64), 1, torch.float16),
        ((8, 8, 8, 8, 8), (8, 8), 0, torch.float16),
        ((8, 8, 8, 8, 8), (8, 8), 2, torch.float16),
        ((1024, 1024, 1024), (1, ), 1, torch.float16),
        ((2048, 2048), (128, 128), 0, torch.float16),
        ((2048, 2048), (128, 128), 1, torch.float16),
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
        c_uint64,
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

    if args.profile:
        PROFILE = True
    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if not (args.cpu or args.cuda):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
