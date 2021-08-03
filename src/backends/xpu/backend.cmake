#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#
#
# Copyright © tuanhe. All rights reserved.
# SPDX-License-Identifier: MIT
#
if(XPU_SUPPORT)
    set(XPU_SDK_ROOT ${SET_YOUR_SDK_DIR})
    #set(XPU_SDK_ROOT ./)
    message(STATUS "SDK root : ${XPU_SDK_ROOT}")
    if(NOT DEFINED XPU_SDK_ROOT)
        message(FATAL_ERROR  "XPU_SDK_ROOT is not set while you enable XPU_SUPPORT")  
    endif()

    set(HEADER_FILE gcompiler_api.h)
    # Add the support library
    find_path(SUPPORT_LIBRARY_INCLUDE_DIR 
              ${HEADER_FILE}
              HINTS ${XPU_SDK_ROOT}/include)
    if(NOT SUPPORT_LIBRARY_INCLUDE_DIR)
        message(WARNING "XPU support head file (${HEADER_FILE}) not found")
    else()
        message(STATUS "XPU support head file located at: ${XPU_SUPPORT_LIBRARY}")
        include_directories(${SUPPORT_LIBRARY_INCLUDE_DIR})
    endif()

    set(LIB libaipubuildtool.so)
    find_library(XPU_SUPPORT_LIBRARY
                 NAMES ${LIB}
                 HINTS ${XPU_SDK_ROOT}/lib)
    if(NOT XPU_SUPPORT_LIBRARY)
        message(WARNING "Custom support library (${LIB}) not found")
    else()
        message(STATUS "Custom support library located at: ${XPU_SUPPORT_LIBRARY}")
        link_libraries(${XPU_SUPPORT_LIBRARY})
    endif()

    add_subdirectory(${PROJECT_SOURCE_DIR}/src/backends/xpu)
    list(APPEND armnnLibraries armnnXPUBackend)
    list(APPEND armnnLibraries armnnXPUBackendWorkloads)
    
    if(BUILD_UNIT_TESTS)
        list(APPEND armnnUnitTestLibraries armnnXPUBackendUnitTests)
    endif()

endif()