//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

#include <xpu/XpuPreCompiledObject.hpp>

#include <armnn/TypesUtils.hpp>

namespace armnn
{

class XPUPreCompiledWorkload : public BaseWorkload<PreCompiledQueueDescriptor>
{
public:
    XPUPreCompiledWorkload(const PreCompiledQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    const XPUPreCompiledObject* m_PreCompiledObject;
};

} // namespace armnn
