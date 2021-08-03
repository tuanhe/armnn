//
// Copyright © tuanhe. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "XpuLayerBridge.hpp"

namespace armnn
{

XpuLayerBridge& XpuLayerBridge::GetBridge()
{
    static XpuLayerBridge bridge;
    return bridge;
}

LayerFunctionPtr XpuLayerBridge::GetLayerFunction(LayerType type)
{
    std::map<LayerType, LayerFunctionPtr>::iterator iter = m_LayerMap.find(type);
    if(iter == m_LayerMap.end())
        return nullptr;
    return iter->second;
}

void XpuLayerBridge::RegisterLayer(LayerType type, LayerFunctionPtr fn)
{
    m_LayerMap.insert(std::pair<LayerType, LayerFunctionPtr>(type, fn));
}

} // namespace armnn
