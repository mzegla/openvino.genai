# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

include(FetchContent)

if(NOT TARGET dr_libs)
    FetchContent_Declare(dr_libs
        URL https://github.com/mackron/dr_libs/archive/da35f9d6c7374a95353fd1df1d394d44ab66cf01.tar.gz
        URL_HASH SHA256=2704d347f480ca1bc92233fb01747e4550cc8031735b6ea62ca9990ebb8851ae)
    FetchContent_MakeAvailable(dr_libs)
endif()

add_executable(whisper_speech_recognition whisper_speech_recognition.cpp audio_utils.cpp)
target_link_libraries(whisper_speech_recognition PRIVATE openvino::genai)
target_include_directories(whisper_speech_recognition PRIVATE "$<BUILD_INTERFACE:${dr_libs_SOURCE_DIR}>")
set_target_properties(whisper_speech_recognition PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)
target_compile_features(whisper_speech_recognition PRIVATE cxx_std_11)

install(TARGETS whisper_speech_recognition
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)

add_executable(whisper_dynamic_batching_benchmark whisper_dynamic_batching_benchmark.cpp audio_utils.cpp)
target_link_libraries(whisper_dynamic_batching_benchmark PRIVATE openvino::genai)
target_include_directories(whisper_dynamic_batching_benchmark PRIVATE "$<BUILD_INTERFACE:${dr_libs_SOURCE_DIR}>")
set_target_properties(whisper_dynamic_batching_benchmark PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)
target_compile_features(whisper_dynamic_batching_benchmark PRIVATE cxx_std_17)

install(TARGETS whisper_dynamic_batching_benchmark
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)

add_executable(pa_experimental_test pa_experimental_test.cpp)
target_link_libraries(pa_experimental_test PRIVATE openvino::genai)
set_target_properties(pa_experimental_test PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)
target_compile_features(pa_experimental_test PRIVATE cxx_std_11)

install(TARGETS pa_experimental_test
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
