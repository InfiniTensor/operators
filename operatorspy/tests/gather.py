import sys
import os
import ctypes
import numpy as np
import torch

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

from ctypes import POINTER, Structure, c_int32, c_void_p

class GatherDescriptor(Structure):
    _fields_ = [("device", c_int32)]

infiniopGatherDescriptor_t = POINTER(GatherDescriptor)


def gather(data, axis, indices):
    # 计算输出形状：data.shape[:axis] + indices.shape + data.shape[axis+1:]
    out_shape = list(data.shape[:axis]) + list(indices.shape) + list(data.shape[axis+1:])
    # 生成输出张量各维度坐标的网格
    grids = torch.meshgrid(*[torch.arange(s, device=data.device) for s in out_shape], indexing='ij')
    # 为 gather 生成高级索引列表
    index_list = []
    # 对于 data 的前 axis 维度，对应网格的前 axis 个
    for i in range(axis):
        index_list.append(grids[i])
    # 对于 data 第 axis 维，使用 indices，需要先将其扩展到输出形状
    new_shape = [1] * axis + list(indices.shape) + [1] * (data.dim() - axis - 1)
    indices_expanded = indices.view(new_shape).expand(out_shape)
    index_list.append(indices_expanded)
    # 对于 data 后续维度，映射到网格中对应的位置
    for i in range(axis, data.dim() - 1):
        index_list.append(grids[i + len(indices.shape)])
    return data[tuple(index_list)]


def test(lib, handle, torch_device, data_shape, indices_shape, axis, tensor_dtype=torch.float16):
    print(f"Testing Gather on {torch_device} with data_shape:{data_shape}, indices_shape:{indices_shape}, axis:{axis}, dtype:{tensor_dtype}")
    data = torch.rand(data_shape, dtype=tensor_dtype).to(torch_device)
    indices = torch.randint(0, data_shape[axis], indices_shape, dtype=torch.int64).to(torch_device)
    output = torch.empty(data.shape[:axis] + indices.shape + data.shape[axis+1:], dtype=tensor_dtype).to(torch_device)
    ans = gather(data, axis, indices)


    data_tensor = to_tensor(data, lib)
    indices_tensor = to_tensor(indices, lib)
    output_tensor = to_tensor(output, lib)

    descriptor = infiniopGatherDescriptor_t()
    check_error(lib.infiniopCreateGatherDescriptor(handle,
                ctypes.byref(descriptor),
                output_tensor.descriptor,
                data_tensor.descriptor,
                indices_tensor.descriptor,
                ctypes.c_int64(axis)))

    data_tensor.descriptor.contents.invalidate()
    indices_tensor.descriptor.contents.invalidate()
    output_tensor.descriptor.contents.invalidate()

    check_error(lib.infiniopGather(descriptor,
                output_tensor.data,
                data_tensor.data,
                indices_tensor.data,
                None))
    assert torch.allclose(output, ans, atol=0, rtol=0)
    check_error(lib.infiniopDestroyGatherDescriptor(descriptor))

def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for data_shape, indices_shape, axis in test_cases:
        test(lib, handle, "cpu", data_shape, indices_shape, axis, tensor_dtype=torch.float16)
        test(lib, handle, "cpu", data_shape, indices_shape, axis, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)

if __name__ == "__main__":
    test_cases = [
        # (data_shape, indices_shape, axis)
        ((3, 4), (2,), 0),
        ((3, 4), (3,), 1),
        ((2, 3, 4), (2,), 1),
        ((3, 2), (2, 2), 0),
        ((3, 3), (1, 2), 1),
    ]
    from operatorspy.tests.test_utils import get_args
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateGatherDescriptor.restype = ctypes.c_int32
    lib.infiniopCreateGatherDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopGatherDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        ctypes.c_int64,
    ]
    lib.infiniopGather.restype = ctypes.c_int32
    lib.infiniopGather.argtypes = [
        infiniopGatherDescriptor_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.infiniopDestroyGatherDescriptor.restype = ctypes.c_int32
    lib.infiniopDestroyGatherDescriptor.argtypes = [
        infiniopGatherDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)
    if not args.cpu:
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
