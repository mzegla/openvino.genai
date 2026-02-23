// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "openvino/core/model.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {
namespace utils {

/**
 * @brief Applies gather-before-matmul transformation to the model.
 * 
 * This transformation optimizes matrix multiplication operations by applying
 * gather operations before the matmul, which can improve performance for
 * PagedAttention-based models.
 * 
 * @param model Pointer to the ov::Model to be transformed.
 */
OPENVINO_GENAI_EXPORTS void apply_gather_before_matmul_transformation(std::shared_ptr<ov::Model> model);

}  // namespace utils
}  // namespace genai
}  // namespace ov
