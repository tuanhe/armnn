#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

AUX_SOURCE_DIRECTORY(.    armnnXPUBackendUnitTests_source_files)
list(APPEND armnnXPUBackendUnitTests_sources ${armnnXPUBackendUnitTests_source_files})

add_library(armnnXPUBackendUnitTests OBJECT ${armnnXPUBackendUnitTests_sources})
target_include_directories(armnnXPUBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnXPUBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnXPUBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnXPUBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnXPUBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)
