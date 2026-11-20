# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

include(FetchContent)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

FetchContent_Declare(cxxopts
    URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.3.1.tar.gz
    URL_HASH SHA256=3bfc70542c521d4b55a46429d808178916a579b28d048bd8c727ee76c39e2072)
FetchContent_MakeAvailable(cxxopts)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

set(TARGET_NAME continuous_batching_staged_poc)
add_executable(${TARGET_NAME} ${TARGET_NAME}.cpp)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::genai cxxopts::cxxopts)

set_target_properties(${TARGET_NAME} PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION samples_bin/
        COMPONENT tools_bin
        EXCLUDE_FROM_ALL)
