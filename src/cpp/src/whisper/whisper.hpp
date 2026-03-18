// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>

#include "context_tokens.hpp"
#include "models/decoder.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/whisper_generation_config.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "sampling/sampler.hpp"
#include "whisper/config.hpp"
#include "whisper/feature_extractor.hpp"
#include "whisper/models.hpp"
#include "circular_buffer_queue.hpp"

namespace ov {
namespace genai {

struct Segment {
    float m_start;
    float m_end;
    std::vector<int64_t> m_tokens;
};

struct WhisperGenerateResult {
    std::vector<int64_t> output_tokens;
    std::optional<std::vector<Segment>> segments = std::nullopt;
    std::optional<std::vector<WhisperWordTiming>> words = std::nullopt;
    WhisperPerfMetrics perf_metrics;
};

WhisperGenerateResult whisper_generate(const ov::genai::WhisperGenerationConfig& config,
                                       const ov::genai::WhisperConfig& model_config,
                                       const WhisperContextTokens& context_tokens,
                                       const RawSpeechInput& raw_speech,
                                       ov::InferRequest& encoder,
                                       std::shared_ptr<WhisperDecoder> decoder,
                                       WhisperFeatureExtractor& feature_extractor,
                                       const std::shared_ptr<StreamerBase> streamer,
                                       Sampler& sampler,
                                       Tokenizer& tokenizer);


class SpeechEncoder {
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_encoder_pool;
    mutable std::mutex m_decoder_mutex;  // serialize prepare_sot_tokens (decoder not pooled)
    std::shared_ptr<WhisperDecoder> m_decoder;
    WhisperFeatureExtractor m_feature_extractor;
    WhisperConfig m_model_config;
    double m_last_encoder_infer_ms = 0.0;
    double m_last_feature_extract_ms = 0.0;
    double m_last_sot_tokens_ms = 0.0;
public:
    double get_last_encoder_infer_ms()    const { return m_last_encoder_infer_ms; }
    double get_last_feature_extract_ms()  const { return m_last_feature_extract_ms; }
    double get_last_sot_tokens_ms()        const { return m_last_sot_tokens_ms; }
    SpeechEncoder(const std::filesystem::path& model_path,
                  const std::string& device,
                  const ov::AnyMap& properties);

    // Encode raw speech into hidden states for each data frame in order
    std::vector<std::pair<ov::Tensor, ov::Tensor>> encode(const RawSpeechInput& raw_speech,
                                  const WhisperContextTokens& context_tokens,
                                  const ov::genai::WhisperGenerationConfig& config);
    };

}  // namespace genai
}  // namespace ov
