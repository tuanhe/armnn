#
# Copyright © 2017 ARM Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

# BACKEND_SOURCES contains the list of files to be included
# in the Android build and it is picked up by the Android.mk
# file in the root of ArmNN

BACKEND_SOURCES := \
        XPUBackend.cpp \
        XPULayerSupport.cpp \
        XPUWorkloadFactory.cpp \
        workloads/XPUAdditionWorkload.cpp \

# BACKEND_TEST_SOURCES contains the list of files to be included
# in the Android unit test build (armnn-tests) and it is picked
# up by the Android.mk file in the root of ArmNN

BACKEND_TEST_SOURCES := \
         test/XPUCreateWorkloadTests.cpp \
         test/XPUEndToEndTests.cpp
