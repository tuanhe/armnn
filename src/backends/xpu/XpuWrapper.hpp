//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <functional>
#include <map>

#include <Layer.hpp>
#include <SubgraphView.hpp>

#include "gcompiler_api.h"

namespace armnn
{

typedef struct _section
{

}Section;

typedef struct _CompiledNetworkSection
{
    std::unique_ptr<Section> text = nullptr;
    std::unique_ptr<Section> rodata = nullptr;
    std::unique_ptr<Section> data = nullptr;
    std::unique_ptr<Section> stack = nullptr;
    std::unique_ptr<Section> featuremap = nullptr;
    std::unique_ptr<Section> sram = nullptr;
} CompiledNetworkSection;

class XpuLib
{
    public:
        XpuLib();
        ~XpuLib() = default;
        bool Init();
        bool Compile();
        bool Execute();

        
    private:
        bool Quantize();
        bool GetSectionInfo();
        
    private:
        aipubt::GraphBuilder* m_Gbuilder;
        std::shared_ptr<aipubt::Graph> m_Graph;
};

} // namespace armnn
