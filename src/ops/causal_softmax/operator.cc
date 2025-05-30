#include "../utils.h"
#include "operators.h"
#include "ops/causal_softmax/causal_softmax.h"

#ifdef ENABLE_CPU
#include "cpu/causal_softmax_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/common_cuda.h"
#include "cuda/causal_softmax.cuh"
#include "../../devices/cuda/cuda_handle.h"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "../../devices/bang/bang_handle.h"
#include "bang/causal_softmax_bang.h"
#include "bang/causal_softmax_cnnl.h"
#endif
#ifdef ENABLE_ASCEND_NPU
#include "ascend/causal_softmax_aclnn.h"
#endif
#ifdef ENABLE_METAX_GPU
#include "maca/causal_softmax_maca.h"
#endif
#ifdef ENABLE_MTHREADS_GPU
#include "musa/causal_softmax_musa.h"
#include "../../devices/musa/common_musa.h"
#endif

__C infiniopStatus_t infiniopCreateCausalSoftmaxDescriptor(
    infiniopHandle_t handle,
    infiniopCausalSoftmaxDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateCausalSoftmaxDescriptor(handle, (CausalSoftmaxCpuDescriptor_t *) desc_ptr, y_desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateCausalSoftmaxDescriptor((CudaHandle_t)handle, (CausalSoftmaxCudaDescriptor_t *) desc_ptr, y_desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            // return bangCreateCausalSoftmaxDescriptor((BangHandle_t) handle, (CausalSoftmaxBangDescriptor_t *) desc_ptr, y_desc);
            return cnnlCreateCausalSoftmaxDescriptor((BangHandle_t) handle, (CausalSoftmaxCnnlDescriptor_t *) desc_ptr, y_desc);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnCreateCausalSoftmaxDescriptor((AscendHandle_t) handle, (CausalSoftmaxAclnnDescriptor_t *) desc_ptr, y_desc);
        }
#endif
#ifdef ENABLE_METAX_GPU
        case DevMetaxGpu: {
            return macaCreateCausalSoftmaxDescriptor((MacaHandle_t) handle, (CausalSoftmaxMacaDescriptor_t *) desc_ptr, y_desc);
        }
#endif
#ifdef ENABLE_MTHREADS_GPU
        case DevMthreadsGpu: {
            return musaCreateCausalSoftmaxDescriptor((MusaHandle_t) handle, (CausalSoftmaxMusaDescriptor_t *) desc_ptr, y_desc);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopGetCausalSoftmaxWorkspaceSize(infiniopCausalSoftmaxDescriptor_t desc, uint64_t *size) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuGetCausalSoftmaxWorkspaceSize((CausalSoftmaxCpuDescriptor_t) desc, size);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaGetCausalSoftmaxWorkspaceSize((CausalSoftmaxCudaDescriptor_t) desc, size);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            // return bangGetCausalSoftmaxWorkspaceSize((CausalSoftmaxBangDescriptor_t) desc, size);
            return cnnlGetCausalSoftmaxWorkspaceSize((CausalSoftmaxCnnlDescriptor_t) desc, size);
        }

#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnGetCausalSoftmaxWorkspaceSize((CausalSoftmaxAclnnDescriptor_t) desc, size);
        }
#endif
#ifdef ENABLE_METAX_GPU
        case DevMetaxGpu: {
            return macaGetCausalSoftmaxWorkspaceSize((CausalSoftmaxMacaDescriptor_t) desc, size);
        }
#endif
#ifdef ENABLE_MTHREADS_GPU
        case DevMthreadsGpu: {
            return musaGetCausalSoftmaxWorkspaceSize((CausalSoftmaxMusaDescriptor_t) desc, size);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopCausalSoftmax(infiniopCausalSoftmaxDescriptor_t desc, void *workspace, uint64_t workspace_size, void *data, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCausalSoftmax((CausalSoftmaxCpuDescriptor_t) desc, workspace, workspace_size, data, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCausalSoftmax((CausalSoftmaxCudaDescriptor_t) desc, workspace, workspace_size, data, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            // return bangCausalSoftmax((CausalSoftmaxBangDescriptor_t) desc, workspace, workspace_size, data, stream);
            return cnnlCausalSoftmax((CausalSoftmaxCnnlDescriptor_t) desc, workspace, workspace_size, data, stream);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnCausalSoftmax((CausalSoftmaxAclnnDescriptor_t) desc, workspace, workspace_size, data, stream);
        }
#endif
#ifdef ENABLE_METAX_GPU
        case DevMetaxGpu: {
            return macaCausalSoftmax((CausalSoftmaxMacaDescriptor_t) desc, workspace, workspace_size, data, stream);
        }
#endif
#ifdef ENABLE_MTHREADS_GPU
        case DevMthreadsGpu: {
            return musaCausalSoftmax((CausalSoftmaxMusaDescriptor_t) desc, workspace, workspace_size, data, stream);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyCausalSoftmaxDescriptor(infiniopCausalSoftmaxDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyCausalSoftmaxDescriptor((CausalSoftmaxCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyCausalSoftmaxDescriptor((CausalSoftmaxCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            // return bangDestroyCausalSoftmaxDescriptor((CausalSoftmaxBangDescriptor_t) desc);
            return cnnlDestroyCausalSoftmaxDescriptor((CausalSoftmaxCnnlDescriptor_t) desc);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnDestroyCausalSoftmaxDescriptor((CausalSoftmaxAclnnDescriptor_t) desc);
        }
#endif
#ifdef ENABLE_METAX_GPU
        case DevMetaxGpu: {
            return macaDestroyCausalSoftmaxDescriptor((CausalSoftmaxMacaDescriptor_t) desc);
        }
#endif
#ifdef ENABLE_MTHREADS_GPU
        case DevMthreadsGpu:
            return musaDestroyCausalSoftmaxDescriptor((CausalSoftmaxMusaDescriptor_t) desc);
#endif
    }
    return STATUS_BAD_DEVICE;
}
