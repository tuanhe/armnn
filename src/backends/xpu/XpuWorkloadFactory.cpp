//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "XPUWorkloadFactory.hpp"

#include "workloads/XPUAdditionWorkload.hpp"
#include "workloads/XPUPreCompiledWorkload.hpp"

#include <backendsCommon/TensorHandle.hpp>
#include <backendsCommon/MemCopyWorkload.hpp>

#include <Layer.hpp>

#include <boost/log/trivial.hpp>

namespace armnn
{

namespace
{

static const BackendId s_Id{"XPU"};

} // Anonymous namespace

XPUWorkloadFactory::XPUWorkloadFactory()
{
}

const BackendId& XPUWorkloadFactory::GetBackendId() const
{
    return s_Id;
}

bool XPUWorkloadFactory::IsLayerSupported(const Layer& layer,
                                             Optional<DataType> dataType,
                                             std::string& outReasonIfUnsupported)
{
    return IWorkloadFactory::IsLayerSupported(s_Id, layer, dataType, outReasonIfUnsupported);
}

std::unique_ptr<ITensorHandle> XPUWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo, bool) const
{
    return std::make_unique<ScopedTensorHandle>(tensorInfo);
}

std::unique_ptr<ITensorHandle> XPUWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                         DataLayout dataLayout,
						                                                 bool isMemoryManaged) const
{
    IgnoreUnused(dataLayout);
    IgnoreUnused(isMemoryManaged);
    return std::make_unique<ScopedTensorHandle>(tensorInfo);
}

std::unique_ptr<IWorkload> XPUWorkloadFactory::CreateInput(const InputQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const
{
    if (info.m_InputTensorInfos.empty() )
    {
        throw InvalidArgumentException("XPUWorkloadFactory::CreateInput: Input cannot be zero length");
    }
    if (info.m_OutputTensorInfos.empty())
    {
        throw InvalidArgumentException("XPUWorkloadFactory::CreateInput: Output cannot be zero length");
    }

    if (info.m_InputTensorInfos[0].GetNumBytes() != info.m_OutputTensorInfos[0].GetNumBytes())
    {
        throw InvalidArgumentException("XPUWorkloadFactory::CreateInput: "
                                       "data input and output differ in byte count.");
    }

    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> XPUWorkloadFactory::CreateOutput(const OutputQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info) const
{
    if (info.m_InputTensorInfos.empty() )
    {
        throw InvalidArgumentException("XPUWorkloadFactory::CreateOutput: Input cannot be zero length");
    }
    if (info.m_OutputTensorInfos.empty())
    {
        throw InvalidArgumentException("XPUWorkloadFactory::CreateOutput: Output cannot be zero length");
    }
    if (info.m_InputTensorInfos[0].GetNumBytes() != info.m_OutputTensorInfos[0].GetNumBytes())
    {
        throw InvalidArgumentException("XPUWorkloadFactory::CreateOutput: "
                                       "data input and output differ in byte count.");
    }

    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> XPUWorkloadFactory::CreateAddition(const AdditionQueueDescriptor& descriptor,
                                                                        const WorkloadInfo& info) const
{
    return std::make_unique<XPUAdditionWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> XPUWorkloadFactory::CreatePreCompiled(const PreCompiledQueueDescriptor& descriptor,
                                                                    const WorkloadInfo& info) const
{
    return std::make_unique<XPUPreCompiledWorkload>(descriptor, info);
}

} // namespace armnn
