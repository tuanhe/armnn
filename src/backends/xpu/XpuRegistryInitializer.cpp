//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "XpuBackend.hpp"

#include <armnn/BackendRegistry.hpp>

namespace
{

using namespace armnn;

static BackendRegistry::StaticRegistryInitializer g_RegisterHelper
{
    BackendRegistryInstance(),
    XPUBackend::GetIdStatic(),
    []()
    {
        return IBackendInternalUniquePtr(new XPUBackend());
    }
};


} // Anonymous namespace
