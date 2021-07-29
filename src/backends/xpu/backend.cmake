#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

add_subdirectory(${PROJECT_SOURCE_DIR}/src/backends/xpu)
list(APPEND armnnLibraries armnnXPUBackend)
list(APPEND armnnLibraries armnnXPUBackendWorkloads)
list(APPEND armnnUnitTestLibraries armnnXPUBackendUnitTests)
