//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "XPUPreCompiledWorkload.hpp"

#include <xpu/XPUBackendUtils.hpp>

#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>

#include <Profiling.hpp>

#include <backendsCommon/TensorHandle.hpp>

#include <boost/polymorphic_cast.hpp>

namespace armnn
{

XPUPreCompiledWorkload::XPUPreCompiledWorkload(const PreCompiledQueueDescriptor& descriptor,
                                                     const WorkloadInfo& info)
    : BaseWorkload<PreCompiledQueueDescriptor>(descriptor, info)
    , m_PreCompiledObject(static_cast<const XPUPreCompiledObject*>(descriptor.m_PreCompiledObject))
{
    // Check that the workload is holdind a pointer to a valid pre-compiled object
    if (m_PreCompiledObject == nullptr)
    {
        throw InvalidArgumentException("XPUPreCompiledWorkload requires a valid pre-compiled object");
    }
}

void XPUPreCompiledWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT("XPU", "XPUPreCompiledWorkload_Execute");

    // This example pre-compiled workload does an element-wise addition on the given inputs using the
    // method stored in the pre-compiled object (to mock a computation done on the xpu backend)

    // Get the input/output buffers
    const float* inputData0 = GetInputTensorData<PreCompiledQueueDescriptor, float>(0, m_Data);
    const float* inputData1 = GetInputTensorData<PreCompiledQueueDescriptor, float>(1, m_Data);
    float* outputData       = GetOutputTensorData<PreCompiledQueueDescriptor, float>(0, m_Data);

    // Get the number of elements
    const TensorInfo& info = XpuGetTensorInfo(m_Data.m_Inputs[0]);
    unsigned int numElements = info.GetNumElements();

    // Do the work
    m_PreCompiledObject->DoElementwiseAddition(inputData0, inputData1, outputData, numElements);
}

} // namespace armnn
