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
import torch


class GatherDescriptor(Structure):
    """定义与 C 语言中的 GatherDescriptor 类型相同的结构体
    """
    _fields_ = [("device", c_int32)]


# infiniopGatherDescriptor_t 类型是 C 语言中指向 GatherDescriptor 类型的指针
infiniopGatherDescriptor_t = POINTER(GatherDescriptor)

def gather(data, axis, index):
    """使用 pytorch 实现 gather 操作
    Args:
        data: 要获取的数据
        axis: 合并的维度
        index: 索引张量
    Returns:
        计算结果
    """
    # 计算合并后的输出张量形状
    output_ndim = data.ndim + index.ndim - 1
    output_shape = [0 for i in range(output_ndim)]
    axis = axis if axis >= 0 else axis + data.ndim
    for i in range(output_ndim):
        if i < axis:
            output_shape[i] = data.shape[i]
        elif i >= axis + index.ndim:
            output_shape[i] = data.shape[i - index.ndim + 1]
        else:
            output_shape[i] = index.shape[i - axis]

    # 初始化输出张量
    output = torch.zeros(output_shape, dtype=data.dtype)
    # 计算输出张量的元素个数
    output_elements = output.numel()
    # 初始化遍历输出张量的索引
    output_indices = [0 for i in range(output_ndim)]
    # 遍历输出张量的每个元素
    for i in range(output_elements):
        # 以 axis 分界从 output_indices 中分别提取 index 和 data 的索引
        index_indices = [0 for i in range(index.ndim)]
        data_indices = [0 for i in range(data.ndim)]
        for j in range(output_ndim):
            if j < axis:
                data_indices[j] = output_indices[j]
            elif j >= axis + index.ndim:
                data_indices[j - index.ndim + 1] = output_indices[j]
            else:
                index_indices[j - axis] = output_indices[j]
        # 从 index 中获取索引值，并赋值给 data_indices 的对应位置
        index_value = index[tuple(index_indices)]
        data_indices[axis] = index_value.item()
        # 从 data 中获取数据值
        data_value = data[tuple(data_indices)]
        # 将数据值赋值给输出张量
        output[tuple(output_indices)] = data_value

        # print("索引", tuple(output_indices), "元素", output[tuple(output_indices)])
        # 递增 output_indices 中的索引
        for j in range(output_ndim - 1, -1, -1):
            output_indices[j] += 1
            if output_indices[j] < output_shape[j]:
                break
            output_indices[j] = 0
    
    return output
        
def test(
        lib,
        handle,
        torch_device,
        output_shape,
        data_shape,
        index_shape,
        axis,
        data_dtype
):
    """使用所加载的动态库，执行 gather 操作，并与 pytorch 计算的结果进行比较
    
    Args:
        lib: 动态库
        handle: 句柄
        torch_device: pytorch 设备
        output_shape: 输出张量的形状
        data_shape: 输入张量的形状
        index_shape: 索引张量的形状
        axis: 合并的维度
        data_dtype: 输入张量的数据类型
    """
    print(
        f"Testing Add on {torch_device} with output_shape:{output_shape} data_shape:{data_shape} index_shape:{index_shape} axis: {axis} data_dtype:{data_dtype}"
    )
    # 生成随机数据
    data = torch.rand(data_shape, dtype=data_dtype).to(torch_device)
    output = torch.zeros(output_shape, dtype=data_dtype).to(torch_device)
    # index_shape 为标量时，直接作为索引
    if (type(index_shape) != tuple):
        index = torch.tensor(index_shape, dtype=torch.int32).to(torch_device)
    else:
        index = torch.randint(-data_shape[axis], data_shape[axis], index_shape, dtype=torch.int32).to(torch_device)
    
    # 使用 pytorch 计算正确结果
    ans = gather(data, axis, index)

    # 将 pytorch 张量转换为 infiniop 张量
    data_tensor = to_tensor(data, lib)
    output_tensor = to_tensor(output, lib)
    index_tensor = to_tensor(index, lib)
    descriptor = infiniopGatherDescriptor_t()

    check_error(
        lib.infiniopCreateGatherDescriptor(
            handle,
            ctypes.byref(descriptor),
            output_tensor.descriptor,
            data_tensor.descriptor,
            index_tensor.descriptor,            
            axis,
        )
    )

    # 置空所有张量的相关信息，避免在实际运算时直接使用这些信息
    data_tensor.descriptor.contents.invalidate()
    index_tensor.descriptor.contents.invalidate()
    output_tensor.descriptor.contents.invalidate()
    check_error(
        lib.infiniopGather(descriptor, output_tensor.data, data_tensor.data, index_tensor.data, None)
    )
    
    # 比较输出结果
    assert torch.allclose(output, ans, atol=0, rtol=0)

    # 销毁算子描述符
    check_error(lib.infiniopDestroyGatherDescriptor(descriptor))
    

def test_cpu(lib, test_cases):
    """测试 CPU 上的给定算子
    Args:
        lib: 要测试的 CPU 算子动态库
        test_cases: 要执行的测试用例
    """
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for output_shape, data_shape, index_shape, axis in test_cases:
        # 测试数据类型为 float16 的数据
        test(lib, handle, "cpu", output_shape, data_shape, index_shape, axis, data_dtype=torch.float16)
        # 测试数据类型为 float32 的数据
        test(lib, handle, "cpu", output_shape, data_shape, index_shape, axis, data_dtype=torch.float32)
    destroy_handle(lib, handle)

if __name__ == "__main__":
    test_cases = [
        # output_shape, data_shape, index_shape, axis
        ((5, 6, 3, 4), (2, 3, 4), (5, 6), 0),
        ((2, 5, 6, 4), (2, 3, 4), (5, 6), 1),
        ((2, 3, 5, 6), (2, 3, 4), (5, 6), 2),
        ((5, 6, 3, 4), (2, 3, 4), (5, 6), -3), # axis 为负的情况
        ((2, 5, 6, 4), (2, 3, 4), (5, 6), -2),
        ((2, 3, 5, 6), (2, 3, 4), (5, 6), -1),
        ((7, 8, 4, 5, 6), (3, 4, 5, 6), (7, 8), 0), # 较大尺寸的张量
        ((3, 7, 8, 5, 6), (3, 4, 5, 6), (7, 8), 1),
        ((3, 4, 7, 8, 6), (3, 4, 5, 6), (7, 8), 2),
        ((3, 4, 5, 7, 8), (3, 4, 5, 6), (7, 8), 3),
        ((200, 32, 32, 7, 8), (200, 32, 32, 3), (7, 8), 3),
        ((3, 4), (2, 3, 4), 0, 0), # index 为标量
        ((3, 4), (2, 3, 4), 1, 0),
        ((2, 4), (2, 3, 4), 0, 1),
        ((2, 4), (2, 3, 4), 1, 1),
        ((2, 4), (2, 3, 4), 2, 1),
        ((2, 3), (2, 3, 4), 0, 2),
        ((2, 3), (2, 3, 4), 1, 2),
        ((2, 3), (2, 3, 4), 2, 2),
        ((2, 3), (2, 3, 4), 3, 2),
        ((2, 3), (2, 3, 4), -1, 2), # index 为负的标量
        ((2, 3), (2, 3, 4), -2, 2),
        ((2, 3), (2, 3, 4), -3, 2),
        ((2, 3), (2, 3, 4), -4, 2)
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
        c_int32,
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
    lib.infiniopDestroyGatherDescriptor.argtypes = [infiniopGatherDescriptor_t]
    if args.cpu:
        test_cpu(lib, test_cases)
    if not (args.cpu):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
