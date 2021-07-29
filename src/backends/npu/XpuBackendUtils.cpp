//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "XPUBackendUtils.hpp"

using namespace armnn;

const TensorInfo& XpuGetTensorInfo(const ITensorHandle* tensorHandle)
{
    // We know that this workloads use CpuTensorHandles only, so this cast is legitimate.
    const ConstTensorHandle* cpuTensorHandle =
        boost::polymorphic_downcast<const ConstTensorHandle*>(tensorHandle);

    return cpuTensorHandle->GetTensorInfo();
}
