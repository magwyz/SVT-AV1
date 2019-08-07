#
# Copyright(c) 2019 Intel Corporation
# SPDX - License - Identifier: BSD - 2 - Clause - Patent
#
# Common/Codec Directory CMakeLists.txt

# Include Encoder Subdirectories
include_directories(${PROJECT_SOURCE_DIR}/third_party/fastfeat/)

file(GLOB all_files
    "*.h"
    "*.c")

add_library(FASTFEAT OBJECT ${all_files})
