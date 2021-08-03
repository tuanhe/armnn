#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#
AUX_SOURCE_DIRECTORY(.    XPUBackend_sources)
list(APPEND armnnXPUBackend_sources ${XPUBackend_sources})

add_library(armnnXPUBackend OBJECT ${armnnXPUBackend_sources})
target_include_directories(armnnXPUBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnXPUBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnXPUBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnXPUBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnXPUBackend PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)

add_subdirectory(workloads)

if(BUILD_UNIT_TESTS)
    add_subdirectory(test)
endif()

if(COMPILER_IS_GNU_LIKE)
    set_source_files_properties(${armnnXPUBackend_sources} PROPERTIES COMPILE_FLAGS
    "-Wno-unused-variable -Wno-unused-parameter -Wno-conversion -Wno-sign-conversion \
    -Wno-old-style-cast -Wno-ignored-qualifiers -Wno-error=deprecated-declarations -Wno-unused-function")
endif()
