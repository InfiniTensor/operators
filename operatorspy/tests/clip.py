from ctypes import POINTER, Structure, c_int32, c_void_p
import ctypes
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    CTensor,
    DeviceEnum,
    open_lib,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    to_tensor,
    create_handle,
    destroy_handle,
    check_error,
)

from operatorspy.tests.test_utils import get_args
from enum import Enum, auto
import torch

class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE = auto()

class ClipDescriptor(Structure):
    _fields_ = [("device", c_int32)]

infiniopClipDescriptor_t = POINTER(ClipDescriptor)

def clip(x, min, max):
    return torch.clamp(x, min, max)

def test(
    lib,
    handle,
    torch_device,
    input_shape,
    min_val,
    max_val,
    tensor_dtype=torch.float16,
    inplace=Inplace.OUT_OF_PLACE,
):
    print(
        f"Testing Clip on {torch_device} with input_shape:{input_shape} min:{min_val} max:{max_val} dtype:{tensor_dtype} inplace: {inplace.name}"
    )
    
    # 随机生成输入张量
    input_data= torch.rand(input_shape, dtype=tensor_dtype).to(torch_device)
    # 随机生成输出张量
    output_data = torch.rand(input_shape, dtype=tensor_dtype).to(torch_device) if inplace == Inplace.OUT_OF_PLACE else input_data
    # 将 min 和 max 转换为标量张量
    min = torch.tensor(min_val, dtype=tensor_dtype).to(torch_device) if min_val is not None else None
    max = torch.tensor(max_val, dtype=tensor_dtype).to(torch_device) if max_val is not None else None

    # 计算正确的结果（pytorch 的 clamp 不支持 min 和 max 均为空，跳过这种情况）
    if (min_val is not None) or (max_val is not None):
        ans = clip(input_data, min, max)

    # 将 torch 张量转换为 infiniop 张量
    input_tensor = to_tensor(input_data, lib)
    output_tensor = to_tensor(output_data, lib) if inplace == Inplace.OUT_OF_PLACE else input_tensor
    min_tensor = to_tensor(min, lib) if min is not None else None
    max_tensor = to_tensor(max, lib) if max is not None else None

    # 创建 clip 算子描述符的指针
    descriptor = infiniopClipDescriptor_t()
    # 创建 clip 算子描述符
    check_error(
        lib.infiniopCreateClipDescriptor(
            handle,
            ctypes.byref(descriptor),
            output_tensor.descriptor,
            input_tensor.descriptor,
            min_tensor.descriptor if min_tensor is not None else None,
            max_tensor.descriptor if max_tensor is not None else None,
        )
    )

    # 标记输入和输出张量的形状和步长信息为无效，检查错误的实现
    input_tensor.descriptor.contents.invalidate()
    output_tensor.descriptor.contents.invalidate()

    # 执行 clip 算子
    check_error(
        lib.infiniopClip(
            descriptor,
            output_tensor.data,
            input_tensor.data,
            min_tensor.data if min_tensor is not None else None,
            max_tensor.data if max_tensor is not None else None,
            None
        )
    )

    # 检查结果是否正确（pytorch 的 clamp 不支持 min 和 max 均为空，跳过这种情况的比较，动态库支持这种操作）
    if (min_val is not None) or (max_val is not None):
        assert torch.allclose(output_data, ans, atol=0, rtol=0)

    # 销毁 clip 算子描述符
    check_error(lib.infiniopDestroyClipDescriptor(descriptor))

def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for input_shape, min, max, inplace in test_cases:
        # 测试数据类型为 float16
        test(lib, handle, "cpu", input_shape, min, max, tensor_dtype=torch.float16, inplace=inplace)
        # 测试数据类型为 float32
        test(lib, handle, "cpu", input_shape, min, max, tensor_dtype=torch.float32, inplace=inplace) 
    destroy_handle(lib, handle)

if __name__ == "__main__":
    test_cases = [
        # input_shape, min, max, inplace
        ((), 0, 1, Inplace.OUT_OF_PLACE), # 测试传入标量时的正确性
        ((), 0.4, 0.6, Inplace.OUT_OF_PLACE), # 传入标量时可以正确执行
        ((1, 3), 0.4, 0.6, Inplace.OUT_OF_PLACE), # 传入标量时可以正确执行
        ((4, 3), 0.3, 0.6, Inplace.OUT_OF_PLACE), # min 和 max 均非空时可以正确执行
        ((4, 3), None, 0.6, Inplace.OUT_OF_PLACE), # min 为空时可以正确执行
        ((4, 3), 0.3, None, Inplace.OUT_OF_PLACE), # max 为空时可以正确执行
        ((4, 3), None, None, Inplace.OUT_OF_PLACE), # min 和 max 均为空时可以正确执行
        ((32, 20, 512), 0.4, 0.6, Inplace.OUT_OF_PLACE), # 较多元素的张量
        ((32, 20, 512), 0.4, 0.6, Inplace.INPLACE), # 较多元素的张量可以原地执行
        ((2, 3, 4, 5, 6), 0.4, 0.6, Inplace.OUT_OF_PLACE) # 较多维度的张量
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateClipDescriptor.restype = c_int32
    lib.infiniopCreateClipDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopClipDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopClip.restype = c_int32
    lib.infiniopClip.argtypes = [
        infiniopClipDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyClipDescriptor.restype = c_int32
    lib.infiniopDestroyClipDescriptor.argtypes = [
        infiniopClipDescriptor_t,
    ]
    if args.cpu:
        test_cpu(lib, test_cases)
    if not (args.cpu):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")