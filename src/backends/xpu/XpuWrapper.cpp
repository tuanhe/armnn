//
// Copyright © tuanhe. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "XpuWrapper.hpp"

namespace armnn
{

XpuLib::XpuLib()
{
}

bool XpuLib::Init()
{
    uint32_t m_InstanceId = 0;
    m_Graph = std::make_shared<aipubt::Graph>("aipu_graph_" + std::to_string(m_InstanceId),
                                                aipubt::DataLayout::DataLayout_NHWC);
    m_Gbuilder = new aipubt::GraphBuilder(m_Graph.get());
    //set target
    //Load driver
    return true;
}

bool XpuLib::Compile()
{
    //1, quantize 
    Quantize();
    //2, build
    m_Gbuilder->build_aipu();
    //3, Get the section
    GetSectionInfo();

    return true;
}

bool XpuLib::GetSectionInfo()
{
    auto ro = m_Gbuilder->get_ro();
    auto text = m_Gbuilder->get_build_text();
    auto constants = m_Gbuilder->get_constants();
    auto descriptor = m_Gbuilder->get_descriptor();

    armnn::IgnoreUnused(ro);
    armnn::IgnoreUnused(text);
    armnn::IgnoreUnused(constants);
    armnn::IgnoreUnused(descriptor);
    return true;
}

bool XpuLib::Execute()
{
    return true;
}

bool XpuLib::Quantize()
{
    //aipuaqt::QuantizeAsymToSym quantize(graph.get());
    //quantize.asym_to_sym();
    //quantize.requantize();
    return true;
}

} // namespace armnn
