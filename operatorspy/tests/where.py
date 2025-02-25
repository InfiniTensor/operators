from ctypes import POINTER, Structure, c_int32, c_void_p
import ctypes
import sys
import os

# 将当前文件 ../../ 添加到环境变量中，这样就能使用 operatorspy 模块了
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    # operatorspy/devices.py 中的类型
    DeviceEnum,
    # operatorspy/operators.py 中的函数和类型
    open_lib,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    # 下面是 operatorspy/utils.py 中的函数
    to_tensor,
    create_handle,
    destroy_handle,
    check_error,
)

from operatorspy.tests.test_utils import get_args
from enum import Enum, auto
import torch

class WhereDescriptor(Structure):
    """定义与 C 语言中的 WhereCpuDescriptor 类型相同的结构体
    """
    _fields_ = [("device", c_int32)]


# infiniopWhereDescriptor_t 类型是 C 语言中指向 WhereDescriptor 类型的指针
infiniopWhereDescriptor_t = POINTER(WhereDescriptor)

def where(x, y, z):
    """调用 torch 中的 where 操作符执行 where 运算
    Args:
        x: 第一个操作数
        y: 第二个操作数
        z: 第三个操作数
    Returns:
        计算结果
    """
    return torch.where(x, y, z)

def test(
    lib,
    handle,
    torch_device,
    output_shape,
    x_shape,
    y_shape,
    tensor_dtype=torch.float16
):
    """使用所加载的动态库，执行实际的算子操作，并与 torch 计算的结果进行比较
    
    Args:
        lib: 要测试的动态库
        handle: 动态库的句柄
        torch_device: torch 计算的设备
        output_shape: 输出张量的形状
        x_shape: 第一个操作数张量的形状
        y_shape: 第二个操作数张量的形状
        tensor_dtype: 张量的数据类型
    """
    print(
        f"Testing Add on {torch_device} with output_shape:{output_shape} x_shape:{x_shape} y_shape:{y_shape} dtype:{tensor_dtype}"
    )
    # 生成随机的输入数据
    x_data = torch.rand(x_shape, dtype=tensor_dtype).to(torch_device)
    y_data = torch.rand(y_shape, dtype=tensor_dtype).to(torch_device)
    output_data = torch.rand(output_shape, dtype=tensor_dtype).to(torch_device)
    condition_data = torch.randint(0, 2, output_shape, dtype=torch.uint8).to(torch_device)

    # 使用 torch 计算的正确结果
    ans = where(condition_data.bool(), x_data, y_data)

    # 将 torch 张量转换为 infiniop 张量
    x_tensor = to_tensor(x_data, lib)
    y_tensor = to_tensor(y_data, lib)
    output_tensor = to_tensor(output_data, lib)
    condition_tensor = to_tensor(condition_data, lib)

    # 创建 where 算子描述符
    descriptor = infiniopWhereDescriptor_t()
    check_error(
        lib.infiniopCreateWhereDescriptor(
            handle,
            ctypes.byref(descriptor),
            output_tensor.descriptor,
            condition_tensor.descriptor,
            x_tensor.descriptor,
            y_tensor.descriptor,
        )
    )

    # 将输入和输出张量的相关信息置为无效，以防止运算时直接使用这些信息
    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()
    condition_tensor.descriptor.contents.invalidate()
    output_tensor.descriptor.contents.invalidate()
    # 执行 where 运算
    check_error(
        lib.infiniopWhere(
            descriptor,
            output_tensor.data,
            condition_tensor.data,
            x_tensor.data,
            y_tensor.data,
            None
        )
    )
    # 比较计算结果
    assert torch.allclose(output_data, ans, atol=0, rtol=0)
    check_error(lib.infiniopDestroyWhereDescriptor(descriptor))

def test_cpu(lib, test_cases):
    """测试 CPU 上的给定算子
    Args:
        lib: 要测试的 CPU 算子动态库
        test_cases: 要执行的测试用例
    """
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for output_shape, x_shape, y_shape in test_cases:
        # 测试数据类型为 float16 的数据
        test(lib, handle, "cpu", output_shape, x_shape, y_shape, tensor_dtype=torch.float16)  
        # 测试数据类型为 float32 的数据
        test(lib, handle, "cpu", output_shape, x_shape, y_shape, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)

if __name__ == "__main__":
    test_cases = [
        # output_shape, x_shape, y_shape
        ((1, 3), (1, 3), (1, 3)),
        ((), (), ()), # 都为空时为标量
        ((3, 3), (3, 3), (3, 3)),
        ((2, 20, 3), (2, 1, 3), (2, 20, 3)), # 广播测试
        ((32, 20, 512), (32, 20, 512), (32, 20, 512)), # 较多数据
        ((32, 256, 112, 112), (32, 256, 112, 112), (32, 256, 112, 112)), # 更多数据
        ((32, 256, 112, 112), (32, 256, 112, 1), (32, 256, 112, 112)), # 较多数据的广播
        ((2, 4, 3), (2, 1, 3), (4, 3)),
        ((2, 3, 4, 5), (2, 3, 4, 5), (5,)), # 广播多个维度
        ((3, 2, 4, 5), (4, 5), (3, 2, 1, 1)), # 广播多个维度
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
    if args.cpu:
        test_cpu(lib, test_cases)
    if not (args.cpu):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")