// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <openvino/openvino.hpp>

#include "openvino/genai/scheduler_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov::genai {
struct PipelineMetrics { 
    // All requests as viewed by the pipeline
    size_t requests = 0;
    // Requests scheduled for processing
    size_t scheduled_requests = 0;
    // Percentage of KV cache usage
    float cache_usage = 0.0;
};

class OPENVINO_GENAI_EXPORTS ContinuousBatchingPipeline {
    class Impl;
    std::shared_ptr<Impl> m_impl;

public:
    ContinuousBatchingPipeline(const std::string& models_path,
                               const SchedulerConfig& scheduler_config,
                               const std::string& device = "CPU",
                               const ov::AnyMap& plugin_config = {});

    std::shared_ptr<ov::genai::Tokenizer> get_tokenizer();

    ov::genai::GenerationConfig get_config() const;

    PipelineMetrics get_metrics() const;

    std::vector<std::string> get_model_configuration();

    GenerationHandle add_request(uint64_t request_id, std::string prompt, ov::genai::GenerationConfig sampling_params);

    void step();

    bool has_non_finished_requests();

    // more high level interface, which can process multiple prompts in continuous batching manner
    std::vector<GenerationResult> generate(const std::vector<std::string>& prompts, std::vector<ov::genai::GenerationConfig> sampling_params);
};
}
