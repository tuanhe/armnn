#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

AUX_SOURCE_DIRECTORY(.    armnnXPUBackendWorkloads_source_files)
list(APPEND armnnXPUBackendWorkloads_sources ${armnnXPUBackendWorkloads_source_files})

add_library(armnnXPUBackendWorkloads OBJECT ${armnnXPUBackendWorkloads_sources})
target_include_directories(armnnXPUBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnXPUBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnXPUBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnXPUBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnXPUBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)
