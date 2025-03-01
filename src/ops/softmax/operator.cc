#include "../utils.h"
#include "operators.h"
#include "ops/softmax/softmax.h"

#ifdef ENABLE_CPU
#include "cpu/softmax_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/common_cuda.h"
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/softmax.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "../../devices/bang/bang_handle.h"
#include "bang/softmax_bang.h"
#include "bang/softmax_cnnl.h"
#endif


__C infiniopStatus_t infiniopCreateSoftmaxDescriptor(
    infiniopHandle_t handle,
    infiniopSoftmaxDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t input_desc, int axis, infiniopTensorDescriptor_t output_desc) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateSoftmaxDescriptor(handle, (SoftmaxCpuDescriptor_t *) desc_ptr, input_desc, axis, output_desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateSoftmaxDescriptor((CudaHandle_t) handle, (SoftmaxCudaDescriptor_t *) desc_ptr, input_desc, axis, output_desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            //return bangCreateSoftmaxDescriptor((BangHandle_t) handle, (SoftmaxBangDescriptor_t *) desc_ptr, input_desc, axis, output_desc);
            return cnnlCreateSoftmaxDescriptor((BangHandle_t) handle, (SoftmaxCnnlDescriptor_t *) desc_ptr, input_desc, axis, output_desc);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}
__C infiniopStatus_t infiniopGetSoftmaxWorkspaceSize(infiniopSoftmaxDescriptor_t desc, uint64_t *size) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuGetSoftmaxWorkspaceSize((SoftmaxCpuDescriptor_t) desc, size);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaGetSoftmaxWorkspaceSize((SoftmaxCudaDescriptor_t) desc, size);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            //return bangGetSoftmaxWorkspaceSize((SoftmaxBangDescriptor_t) desc, size);
            return cnnlGetSoftmaxWorkspaceSize((SoftmaxCnnlDescriptor_t) desc, size);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopSoftmax(infiniopSoftmaxDescriptor_t desc, void *workspace,
                                          uint64_t workspace_size, void const *input, void *output, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuSoftmax((SoftmaxCpuDescriptor_t) desc, workspace, workspace_size, input, output, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaSoftmax((SoftmaxCudaDescriptor_t) desc, workspace, workspace_size, input, output, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            //return bangSoftmax((SoftmaxBangDescriptor_t) desc, workspace, workspace_size, input, output, stream);
            return cnnlSoftmax((SoftmaxCnnlDescriptor_t) desc, workspace, workspace_size, input, output, stream);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroySoftmaxDescriptor(infiniopSoftmaxDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroySoftmaxDescriptor((SoftmaxCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroySoftmaxDescriptor((SoftmaxCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            //return bangDestroySoftmaxDescriptor((SoftmaxBangDescriptor_t) desc);
            return cnnlDestroySoftmaxDescriptor((SoftmaxCnnlDescriptor_t) desc);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}
