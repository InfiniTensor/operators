from ctypes import POINTER, Structure, c_int32, c_void_p, c_uint64, c_bool
import ctypes
import sys
import os
import time

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
from typing import Tuple
import numpy as np

PROFILE = True
NUM_PRERUN = 1
NUM_ITERATIONS = 50

class ReducemaxDescriptor(Structure):
    _fields_ = [("device", c_int32)]

infiniopReducemaxDescriptor_t = POINTER(ReducemaxDescriptor)

def reduce_max(input, axis, noop_with_empty_axes, keepdims=True):
    if axis == None:
        if noop_with_empty_axes:
            return input
        else:
            return torch.amax(input, dim=axis, keepdim=keepdims)
    return torch.amax(input, dim=axis, keepdim=keepdims)

def inferShape(x_shape, axis, noop_with_empty_axes, keepdims=False):
    if axis == None:
        if noop_with_empty_axes:
            return x_shape
        else:
            if keepdims:
                return tuple([1] * len(x_shape))
            else:
                return tuple([])
    assert len(axis) <= len(x_shape), "axis out of range"
    output_shape = []
    axis = [a if a >= 0 else a + len(x_shape) for a in axis]  # 更新 axis 列表中的值
    for a in axis:
        assert 0 <= a <= len(x_shape) - 1, "axis out of range"
    for i, s in enumerate(x_shape):
        if i in axis and keepdims:
            output_shape.append(1)
        elif i in axis and not keepdims:
            continue
        else:
            output_shape.append(s)
            
    print(f"output_shape = {output_shape}")
    return tuple(output_shape)

def tuple_to_void_p(py_tuple: Tuple):
    array = ctypes.c_int64 * len(py_tuple)
    data_array = array(*py_tuple)
    return ctypes.cast(data_array, ctypes.c_void_p)

def test(
    lib,
    handle,
    torch_device,
    x_shape,
    axes,
    dynamic_axes,
    noop_with_empty_axes=False,
    keepdims=True,
    tensor_dtype=torch.float16
):
    print(
        f"Testing reducemax on {torch_device} with x_shape:{x_shape} dtype:{tensor_dtype}"
    )
    x = torch.randn(x_shape, dtype=tensor_dtype, device=torch_device)
    print(f"y_shape = {inferShape(x_shape, axes if dynamic_axes == None else dynamic_axes, noop_with_empty_axes, keepdims)}")
    y = torch.full(inferShape(x_shape, axes if dynamic_axes == None else dynamic_axes, noop_with_empty_axes, keepdims), float('-inf'), dtype=tensor_dtype, device=torch_device)
    print(f"y_shape = {y.shape}")
    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = reduce_max(x, axes if dynamic_axes == None else dynamic_axes, noop_with_empty_axes, keepdims)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = reduce_max(x, axes if dynamic_axes == None else dynamic_axes, noop_with_empty_axes, keepdims)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :10f}")
    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib)
    descriptor = infiniopReducemaxDescriptor_t()
    axe = tuple_to_void_p(axes) if axes != None else None
    lenth_axes = c_uint64(len(axes)) if axes != None else c_uint64(0)
    dynamic_axes_parm = tuple_to_void_p(dynamic_axes) if dynamic_axes != None else None
    lenth_dynamic_axes = c_uint64(len(dynamic_axes)) if dynamic_axes != None else c_uint64(0)
    check_error(
        lib.infiniopCreateReducemaxDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
            axe,
            lenth_axes,
            c_bool(keepdims),
            c_bool(noop_with_empty_axes),
        )
    )
    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()
    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(
            lib.infiniopReducemax(
                descriptor,
                y_tensor.data,
                x_tensor.data,
                dynamic_axes_parm,
                lenth_dynamic_axes,
                None,
            )
        )
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopReducemax(
                    descriptor,
                    y_tensor.data,
                    x_tensor.data,
                    dynamic_axes_parm,
                    lenth_dynamic_axes,
                    None,
                )
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"lib time: {elapsed :10f}")
    # print(f"input : {x}")
    # print(f"custom op output:{y}")
    # print(f"pytorch output:{ans}")
    check_error(lib.infiniopDestroyReducemaxDescriptor(descriptor))
    assert torch.allclose(y, ans, atol=0, rtol=1e-3)

def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape, axes, noop_with_empty_axes, keepdims, dynamic_axes, tensor_dtype in test_cases:
        test(lib, handle, "cpu", x_shape, axes, dynamic_axes, noop_with_empty_axes, keepdims, tensor_dtype=tensor_dtype)
        print("\n")
        #test(lib, handle, "cpu", x_shape, axes, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)

def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for x_shape, axes, noop_with_empty_axes, keepdims, dynamic_axes, tensor_dtype in test_cases:
        test(lib, handle, "cuda", x_shape, axes, dynamic_axes, noop_with_empty_axes, keepdims, tensor_dtype=tensor_dtype)
        print("\n")
    destroy_handle(lib, handle)

if __name__ == "__main__":
    test_cases = [
        # dynamic calc test eg
        # ((2, 3, 4, 5), [0, 2], False, True, None),
        # ((2, 3, 4, 5), [0, 2], False, True, None),
        # #(input_shape, axis, noop_with_empty_axes, keepdims, dynamic_axes)
        # ((2, 10, 24, 10), [0, 2], False, True, None),
        # # stride = 
        # ((2, 10, 24, 10), [0, 1], False, True, None),
        # ((2, 10, 24, 10), [2, 3], False , True, None),
        # ((2, 10, 24, 10), [0, 1, 2, 3], False, True, None),
        # # validate attribute noop_with_empty_axes and keepdims
        # ((2, 10, 24, 10), None, True, True, None),
        # ((2, 10, 24, 10), None, True, False, None),
        # ((2, 10, 24, 10), None, False, True, None),
        # ((2, 10, 24, 10), None, False, False, None),
        # ((2, 3, 4), [0, 1], False, False, None),
        #((2, 10, 24, 10), [], True),
        #((4,), [0], False, False, None, torch.float32),
        ((1000, 300), [0, 1], False, False, None, torch.float16),
        ((50, 3), [0, 1], False, False, None, torch.float16),
        ((1000, 300), [0, 1], False, False, None, torch.float16),
        ((2000, 200, 50), [0, 1], False, True, None, torch.float32),
        ((1000, 200, 500), [0, 1], False, True, None, torch.float16),
        ((1000, 200, 50), [0, 1], False, True, None, torch.float32),
        ((20, 3, 4, 5), [0, 2], False, False, None, torch.float32),
        ((20, 30, 40, 5), [0, 2, 3], False, False, None, torch.float32),
        ((200, 3, 40, 5), [0, 3], False, False, None, torch.float32),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateReducemaxDescriptor.restype = c_int32
    lib.infiniopCreateReducemaxDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopReducemaxDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_void_p,
        c_uint64,
        c_bool,
        c_bool
    ]
    lib.infiniopReducemax.restype = c_int32
    lib.infiniopReducemax.argtypes = [
        infiniopReducemaxDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_uint64,
        c_void_p,
    ]
    lib.infiniopDestroyReducemaxDescriptor.restype = c_int32
    lib.infiniopDestroyReducemaxDescriptor.argtypes = [infiniopReducemaxDescriptor_t]
    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    print("All tests passed!")