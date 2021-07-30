#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

# File needed to wrap the existing backend into a dynamic one
list(APPEND armnnCustomDynamicBackend_sources
    CustomDynamicBackend.cpp
    CustomDynamicBackend.hpp
)

# Set the backend source path
set(CustomBackendPath ${PROJECT_SOURCE_DIR}/src/backends/custom)

# Source files of the backend, taken directly from the source tree
#file(GLOB CustomBackendBaseFiles ${CustomBackendPath}/*.cpp)
#file(GLOB CustomBackendWorloadFiles ${CustomBackendPath}/workloads/*.cpp)
AUX_SOURCE_DIRECTORY(${CustomBackendPath} CustomBackendBaseFiles)
AUX_SOURCE_DIRECTORY(${CustomBackendPath}/workloads CustomBackendWorloadFiles)
set(CustomBackendFiles ${CustomBackendBaseFiles} ${CustomBackendWorloadFiles})

# Remove the file that contains the static backend registration
list(REMOVE_ITEM CustomBackendFiles ${CustomBackendPath}/CustomRegistryInitializer.cpp)

# Create the shared object
add_library(Arm_Custom_backend MODULE ${armnnCustomDynamicBackend_sources} ${CustomBackendFiles})
target_include_directories(Arm_Custom_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(Arm_Custom_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(Arm_Custom_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(Arm_Custom_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(Arm_Custom_backend PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)
set_target_properties(Arm_Custom_backend PROPERTIES PREFIX "")
target_link_libraries(Arm_Custom_backend armnn)
