from ctypes import POINTER, Structure, c_int32, c_void_p, c_int64, c_size_t
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
    check_error,
)

from operatorspy.tests.test_utils import get_args
import torch

class ReduceMinDescriptor(Structure):
    _fields_ = [("device", c_int32)]

infiniopReduceMinDescriptor_t = POINTER(ReduceMinDescriptor)

def reduce_min(data, dim, keepdim):
    return torch.amin(data, dim=dim, keepdim=keepdim)

def test(lib, handle, device, reduced_shape, data_shape, axes, axes_size, keepdims, noop_with_empty_axes, tensor_dtype):
    print(
        f"Testing ReduceMin on {device} with reduced_shape:{reduced_shape} data_shape:{data_shape} axes:{axes} " \
        f"axes_size:{axes_size} keepdims:{keepdims} noop_with_empty_axes:{noop_with_empty_axes} dtype:{tensor_dtype}"
    )

    # 生成随机数据
    data = torch.randn(data_shape, dtype=tensor_dtype).to(device)
    reduced = torch.randn(reduced_shape, dtype=tensor_dtype).to(device)

    # 调用 pytorch 的函数获取实际结果
    if isinstance(data_shape, tuple) and data_shape == (0,):
        # 如果产生的是空集合，ans 应该为无穷小或最小值
        ans = torch.tensor(float("inf"), dtype=tensor_dtype).to(device)
    else: 
        if (axes is None) and (noop_with_empty_axes == 1):
            # axes 为空，且 noop_with_empty_axes 为 1，应该返回原数组
            ans = data
        else:
            # 其他情况都直接调用 pytorch 的函数返回结果
            ans = reduce_min(data,
                             dim=axes, 
                             keepdim=False if (keepdims==0) else True)
    assert ans.shape == reduced.shape

    # 将 pytorch 的数据转换为 tensor
    data_tensor = to_tensor(data, lib)
    reduced_tensor = to_tensor(reduced, lib)
    axes_ptr = None
    if axes is not None:
        axes_tensor = torch.tensor(axes, dtype=torch.int64).to(device)
        axes_ptr = ctypes.cast(axes_tensor.data_ptr(), ctypes.POINTER(ctypes.c_int64))

    # 创建 descriptor，C 语言不支持默认参数，因此在传入参数时根据参数是否为 None 来决定传入的值
    descriptor = infiniopReduceMinDescriptor_t()
    check_error(
        lib.infiniopCreateReduceMinDescriptor(
            handle,
            ctypes.byref(descriptor),
            reduced_tensor.descriptor,
            data_tensor.descriptor,
            axes_ptr,
            axes_size,
            keepdims if keepdims is not None else 1,
            noop_with_empty_axes if noop_with_empty_axes is not None else 0,
        )
    )

    # 置空参数的相关信息
    data_tensor.descriptor.contents.invalidate()
    reduced_tensor.descriptor.contents.invalidate()

    # 调用 infiniop 的函数
    check_error(
        lib.infiniopReduceMin(
            descriptor,
            reduced_tensor.data,
            data_tensor.data,
            axes_ptr,
            None
        )
    )
    if tensor_dtype == torch.float16:
        assert torch.allclose(reduced, ans, atol=0, rtol=1e-3)
    elif tensor_dtype == torch.float32:
        assert torch.allclose(reduced, ans, atol=0, rtol=1e-5)

    # 销毁 descriptor
    check_error(lib.infiniopDestroyReduceMinDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for reduced_shape, data_shape, axes, axes_size, keepdims, noop_with_empty_axes in test_cases:
        test(lib, handle, "cpu", reduced_shape, data_shape, axes, axes_size, keepdims, noop_with_empty_axes, tensor_dtype=torch.float16)
        test(lib, handle, "cpu", reduced_shape, data_shape, axes, axes_size, keepdims, noop_with_empty_axes, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)

if __name__ == "__main__":
    test_cases = [
        # reduced_shape, data_shape, axes, axes_size, keepdims, noop_with_empty_axes
        ((1, 3, 4), (2, 3, 4), [0], 1, 1, 0), # 测试 keepdims
        ((1, 3, 4), (2, 3, 4), [-3], 1, 1, 0),

        ((3, 4), (2, 3, 4), [0], 1, 0, 0),
        ((3, 4), (2, 3, 4), [-3], 1, 0, 0),

        ((2, 1, 4), (2, 3, 4), [1], 1, 1, 0),
        ((2, 1, 4), (2, 3, 4), [-2], 1, 1, 0),

        ((2, 4), (2, 3, 4), [1], 1, 0, 0),
        ((2, 4), (2, 3, 4), [-2], 1, 0, 0),

        ((2, 3, 1), (2, 3, 4), [2], 1, 1, 0),
        ((2, 3, 1), (2, 3, 4), [-1], 1, 1, 0),

        ((2, 3), (2, 3, 4), [2], 1, 0, 0),
        ((2, 3), (2, 3, 4), [-1], 1, 0, 0),

        ((2, 3, 1), (2, 3, 4), [2], 1, 1, 1), # 指定了 axes 时，noop 不生效
        ((2, 3), (2, 3, 4), [2], 1, 0, 1), # 指定了 axes 时，noop 不生效

        ((1, 1, 4), (2, 3, 4), [0, 1], 2, 1, 0),
        ((4), (2, 3, 4), [0, 1], 2, 0, 0),

        ((2, 1, 1), (2, 3, 4), [1, 2], 2, 1, 0),
        ((2), (2, 3, 4), [1, 2], 2, 0, 0),

        ((1, 1, 1, 5, 6), (2, 3, 4, 5, 6), [0, 1, 2], 3, 1, 0),
        ((5, 6), (2, 3, 4, 5, 6), [0, 1, 2], 3, 0, 0),
        ((2, 1, 1, 1, 6), (2, 3, 4, 5, 6), [1, 2, 3], 3, 1, 0),
        ((2, 6), (2, 3, 4, 5, 6), [1, 2, 3], 3, 0, 0),
        ((2, 3, 1, 1, 1), (2, 3, 4, 5, 6), [2, 3, 4], 3, 1, 0),
        ((2, 3), (2, 3, 4, 5, 6), [2, 3, 4], 3, 0, 0),
        ((1, 3, 1, 5, 1), (2, 3, 4, 5, 6), [0, 2, 4], 3, 1, 0),
        ((3, 5), (2, 3, 4, 5, 6), [0, 2, 4], 3, 0, 0),

        ((2, 1, 4), (2, 3, 4), [1], 1, None, None), # axes 非空，keepdims 默认为 1
        ((1, 1, 1), (2, 3, 4), None, 0, None, None), # axes 为空，keepdims 默认为 1，noop 默认为 0
        ((1, 1, 1), (2, 3, 4), None, 0, None, 0), # axes 为空，keepdims 默认为 1，noop 设置为 0
        ((2, 3, 4), (2, 3, 4), None, 0, None, 1), # axes 为空，keepdims 默认为 1，noop 设置为 1
        ((), (2, 3, 4), None, 0, 0, None), # axes 为空，keepdims 为 0，noop 默认为 0
        ((), (2, 3, 4), None, 0, 0, 0), # axes 为空，keepdims 为 0，noop 设置为 0
        ((2, 3, 4), (2, 3, 4), None, 0, 0, 1), # axes 为空，keepdims 为 0，noop 设置为 1，应该返回原数组
        ((1, 1, 1), (2, 3, 4), None, 0, 1, None), # axes 为空，keepdims 为 1，noop 默认为 0
        ((1, 1, 1), (2, 3, 4), None, 0, 1, 0), # axes 为空，keepdims 为 1，noop 设置为 0
        ((2, 3, 4), (2, 3, 4), None, 0, 1, 1), # axes 为空，keepdims 为 1，noop 设置为 1，应该返回原数组

        ((), (), [], 0, 1, 0), # 输入标量
        ((), (), [], 0, 0, 0),
        ((), (), None, 0, 1, 0),
        ((), (), None, 0, 1, 1),

        ((), (0,), [], 0, 1, 0), # 空集合返回负无穷
        ((), (0,), [], 0, 0, 0),
        ((), (0,), None, 0, 1, 0),
        ((), (0,), None, 0, 1, 1),

        ((32, 1, 1, 224), (32, 3, 224, 224), [1, 2], 2, 1, 0),
        ((32, 224), (32, 3, 224, 224), [1, 2], 2, 0, 0),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateReduceMinDescriptor.restype = c_int32
    lib.infiniopCreateReduceMinDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopReduceMinDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        POINTER(c_int64),
        c_size_t,
        c_int32,
        c_int32,
    ]
    lib.infiniopReduceMin.restype = c_int32
    lib.infiniopReduceMin.argtypes = [
        infiniopReduceMinDescriptor_t,
        c_void_p,
        c_void_p,
        POINTER(c_int64),
        c_void_p,
    ]
    lib.infiniopDestroyReduceMinDescriptor.restype = c_int32
    lib.infiniopDestroyReduceMinDescriptor.argtypes = [infiniopReduceMinDescriptor_t]

    if args.cpu:
        test_cpu(lib, test_cases)
    if not (args.cpu):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")