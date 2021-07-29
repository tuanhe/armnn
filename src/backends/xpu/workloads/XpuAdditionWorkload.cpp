//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "XPUAdditionWorkload.hpp"

#include <xpu/XPUBackendUtils.hpp>

#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>

#include <Profiling.hpp>

#include <backendsCommon/TensorHandle.hpp>

#include <boost/polymorphic_cast.hpp>

namespace armnn
{

XPUAdditionWorkload::XPUAdditionWorkload(const AdditionQueueDescriptor& descriptor, const WorkloadInfo& info)
    : BaseWorkload(descriptor, info)
{}

void XPUAdditionWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT("XPU", "XPUAdditionalWorkload_Execute");

    const TensorInfo& info = XpuGetTensorInfo(m_Data.m_Inputs[0]);
    unsigned int num = info.GetNumElements();

    const float* inputData0 = GetInputTensorData<AdditionQueueDescriptor, float>(0, m_Data);
    const float* inputData1 = GetInputTensorData<AdditionQueueDescriptor,float>(1, m_Data);
    float* outputData       = GetOutputTensorData<AdditionQueueDescriptor,float>(0, m_Data);

    for (unsigned int i = 0; i < num; ++i)
    {
        outputData[i] = inputData0[i] + inputData1[i];
    }
}

} // namespace armnn
