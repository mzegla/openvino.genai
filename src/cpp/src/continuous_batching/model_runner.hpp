// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdlib>
#include <set>
#include <sstream>

#include <openvino/runtime/infer_request.hpp>

#include "visual_language/embedding_model.hpp"
#include "sequence_group.hpp"
#include "continuous_batching/scheduler.hpp"
#include "continuous_batching/timer.hpp"

#include "continuous_batching/attention_output.hpp"
#include "continuous_batching/cache_eviction.hpp"
#include "continuous_batching/cross_kv_cache.hpp"

namespace ov::genai {

inline std::string get_paged_attention_score_output_for_decoder_layer(size_t decoder_layer_id) {
    std::stringstream ss;
    ss << "scores." << decoder_layer_id;
    return ss.str();
}

inline std::string get_adaptive_rkv_diversity_score_output_for_decoder_layer(size_t decoder_layer_id) {
    std::stringstream ss;
    ss << "adaptive_rkv_diversity." << decoder_layer_id;
    return ss.str();
}

/**
 * @brief Bitwise flags for hidden state handling in ModelRunner, used in certain speculative decoding, e.g eagle series.
 *
 * The HiddenStateFlags enumeration defines bitwise flags used to control the behavior of hidden state handling in the model runner.
 * Each flag represents a specific mode or capability related to hidden state export, import, or internal processing.
 *
 * Usage:
 *   - Flags can be combined using bitwise OR to enable multiple behaviors simultaneously.
 *   - Helper methods (e.g., enable_hidden_state_export) set or clear these flags.
 *   - Use bitwise AND to check if a flag is enabled.
 *
 * Enum values:
 *   - HS_NONE:    No hidden state operations are enabled (default).
 *   - HS_EXPORT:  Enables exporting hidden states from the model for draft model useage.
 *   - HS_IMPORT:  Enables importing hidden states into the model for a valid draft model forward.
 *   - HS_INTERNAL: Enables internal handling of hidden states for draft model forward.
 */

enum HiddenStateFlags : uint8_t {
    HS_NONE      = 0,
    HS_EXPORT    = 1 << 0,
    HS_IMPORT    = 1 << 1,
    HS_INTERNAL  = 1 << 2
};

/**
 * @brief Uniquely identifies a sequence within a group for hidden state mapping.
 *
 * The SequenceKey struct is used as a composite key for mapping hidden state ranges.
 * It combines a request ID and a grouped sequence ID to uniquely identify a sequence within a batch or group.
 *
 * Members:
 *   - request_id: The unique identifier for the request.
 *   - grouped_sequence_id: The identifier for the sequence within its group.
 *
 * Comparison:
 *   - operator< is defined to allow use as a key in std::map or std::set.
 */

struct SequenceKey {
    size_t request_id{};
    size_t grouped_sequence_id{};
    bool operator<(const SequenceKey& other) const {
        return std::tie(request_id, grouped_sequence_id) <
            std::tie(other.request_id, other.grouped_sequence_id);
    }
};

/**
 * @brief Represents the range of tokens for which hidden states are stored or processed.
 *
 * The HiddenStateRange struct defines a contiguous range of tokens in a sequence,
 * used to indicate which part of the sequence's hidden states are relevant in a batch processing context.
 *
 * Members:
 *   - start_token_idx: The starting index of the token range.
 *   - length: The number of tokens in the range.
 */

struct HiddenStateRange {
    size_t start_token_idx{};
    size_t length{};
};

/**
 * @brief Runs the LLM infer request, parsing the continuous batching scheduler output into proper inputs in terms of OV API (e.g. token input IDs,
 * KV cache block indices etc.) and returning the logit scores for the next token to be generated for each of the currently scheduled sequences.
 */
class ModelRunner {
    ov::InferRequest m_request;
    AttentionScoresForEachSubsequence m_last_attention_scores;
    BlockDiversityForEachSubsequence m_last_block_diversities;
    size_t m_block_size;
    size_t m_num_decoder_layers;
    bool m_collect_attention_scores;
    bool m_is_use_per_layer_cache_control;

    bool m_is_use_rotation_inputs;
    std::vector<std::map<size_t, std::vector<size_t>>> m_rotated_block_logical_indices_per_sequence_for_each_layer;
    std::vector<ov::Tensor> m_cache_rotation_deltas_for_each_layer;
    ov::Tensor m_cache_rotation_trig_lut;

    bool m_is_aggregate_attention_scores;

    bool m_is_use_xattention_inputs;

    bool m_is_use_adaptive_rkv;
    // A model to compute token embeddings.
    // Input shape: [N, conversation length].
    // Output shape: [1, conversation length, hidden_size].
    EmbeddingsModel::Ptr m_embedding;
    uint8_t m_hidden_state_flags = HS_NONE;
    // a container which uses sequence group id and request id as key to store hidden states
    std::map<SequenceKey, HiddenStateRange> m_sequence_hidden_state_mapping;
    std::unordered_map<size_t, ov::Tensor> m_initial_hidden_states; // shape: [N, seq_len, hidden_size]

    std::shared_ptr<InputsEmbedder> m_inputs_embedder;

    // Cached pre-allocated tensors to avoid CPU->GPU copy
    ov::Tensor m_cached_input_ids;
    ov::Tensor m_cached_inputs_embeds;
    ov::Tensor m_cached_position_ids;
    ov::Tensor m_cached_past_lens;
    ov::Tensor m_cached_subsequence_begins;
    ov::Tensor m_cached_block_indices_begins;
    ov::Tensor m_cached_max_context_len;
    ov::Tensor m_cached_score_aggregation_window;
    ov::Tensor m_cached_token_type_ids;

    // Per-forward-call timing — overwritten each forward(), read by pipeline_impl via
    // get_last_timings() to accumulate into PipelineMetrics for the profiling summary.
    double m_last_assembly_us = 0.0;  ///< μs: building the encoder_hidden_states batch tensor
    double m_last_infer_us    = 0.0;  ///< μs: m_request.infer() wall time

    // Cross-attention K/V slot cache (Option A). Non-owning pointer set by pipeline_impl
    // after CrossKVCache is constructed. Null when CROSS_KV_CACHE=0 or not a Whisper model.
    CrossKVCache* m_cross_kv_cache = nullptr;

    // True when the compiled decoder model still has 'encoder_hidden_states' as an input.
    // Checked once on the slow path to avoid re-scanning on every call.
    bool m_has_encoder_hidden_states = true;
    bool m_enc_hs_checked = false;
    // True when the compiled model has a 'cache_position' input (Whisper-specific).
    // Shares semantics with position_ids and is set from the same data each forward.
    bool m_has_cache_position = false;
    bool m_cache_position_checked = false;
public:
    /**
     * Constructs the ModelRunner.
     * @param request The ov::InferRequest for the LLM to be inferred in the continuous batching mode.
     * @param num_decoder_layers Number of decoder attention layers in the LLM corresponding to the request.
     * @param collect_attention_scores If true, then after each `forward` call the ModelRunner will collect and make
     * available the per-token attention scores for each decoder layer, so that these can be used in per-step cache
     * optimizations (such as cache eviction algorithm).
     * @param is_use_per_layer_cache_control If true, then the runner will pass cache control input tensors to the model
     * on a per-attention layer basis.
     * @param is_use_rotation_inputs If true, then the runner will pass cache rotation input tensors to the model
     * on a per-attention layer basis.
     * @param is_aggregate_attention_scores If true, then the runner will pass the input tensors containing per-sequence
     * score aggregation window sizes to the model as requested by the scheduler.
     * on a per-attention layer basis.
     * @param is_use_xattention_inputs If true, then the runner will pass the input tensors containing XAttention
     * configuration per-sequence on a per-attention layer basis.
     */
    ModelRunner(ov::InferRequest request,
                size_t block_size,
                size_t num_decoder_layers = 1,
                bool collect_attention_scores = false,
                bool is_use_per_layer_cache_control = false,
                bool is_use_rotation_inputs = false,
                bool is_aggregate_attention_scores = false,
                bool is_use_xattention_inputs = false,
                bool m_is_use_adaptive_rkv_inputs = false)
        : m_request(std::move(request)),
          m_block_size(block_size),
          m_num_decoder_layers(num_decoder_layers),
          m_collect_attention_scores(collect_attention_scores),
          m_is_use_per_layer_cache_control(is_use_per_layer_cache_control),
          m_is_use_rotation_inputs(is_use_rotation_inputs),
          m_rotated_block_logical_indices_per_sequence_for_each_layer(num_decoder_layers),
          m_is_aggregate_attention_scores(is_aggregate_attention_scores),
          m_is_use_xattention_inputs(is_use_xattention_inputs),
          m_is_use_adaptive_rkv(m_is_use_adaptive_rkv_inputs) {
        OPENVINO_ASSERT(m_num_decoder_layers != 0, "num_decoder_layers must be non-zero");
        _reset_cache_rotation_coefficients();
    }

    /**
     * @return The ov::InferRequest this ModelRunner is handling.
     */
    ov::InferRequest get_infer_request() {
        return m_request;
    }

    void enable_hidden_state_export(bool on)   { on ? m_hidden_state_flags |= HS_EXPORT   : m_hidden_state_flags &= ~HS_EXPORT; }
    void enable_hidden_state_import(bool on)   { on ? m_hidden_state_flags |= HS_IMPORT   : m_hidden_state_flags &= ~HS_IMPORT; }
    void enable_hidden_state_internal(bool on) { on ? m_hidden_state_flags |= HS_INTERNAL : m_hidden_state_flags &= ~HS_INTERNAL; }

    void set_inputs_embedder(const std::shared_ptr<InputsEmbedder>& inputs_embedder) {
        m_inputs_embedder = inputs_embedder;
        m_embedding = inputs_embedder->get_embedding_model();
    }

    /**
     * @return A map of sequence IDs to vectors of ov::Tensor per-token attention scores. Each vector element is associated with its own
     * decoder layer, in order of their execution in the model. Each ov::Tensor has a shape of {N_k}, where N_k is the length of
     * a sequence with ID k processed during the previous `forward` call.
     */
    const AttentionScoresForEachSubsequence& get_last_attention_scores() const {
        return m_last_attention_scores;
    }

    const BlockDiversityForEachSubsequence& get_last_block_diversities() const {
        return m_last_block_diversities;
    }

    /// Timing breakdown from the most recent forward() call.
    struct LastTimings { double assembly_us; double infer_us; };
    LastTimings get_last_timings() const { return {m_last_assembly_us, m_last_infer_us}; }

    /// Attach the cross-KV cache (called by pipeline_impl after CrossKVCache is constructed).
    void set_cross_kv_cache(CrossKVCache* cache) { m_cross_kv_cache = cache; }

    void set_cache_rotation_trig_lut(ov::Tensor&& rotation_trig_lut) {
        m_cache_rotation_trig_lut = std::move(rotation_trig_lut);
    }

    void set_cache_rotation_data(std::vector<std::map<size_t, std::vector<size_t>>>&&
                                     rotated_logical_block_indices_per_sequence_for_each_layer,
                                 std::vector<ov::Tensor>&& rotation_deltas_for_each_layer) {
        m_rotated_block_logical_indices_per_sequence_for_each_layer =
            std::move(rotated_logical_block_indices_per_sequence_for_each_layer);
        m_cache_rotation_deltas_for_each_layer = std::move(rotation_deltas_for_each_layer);
    }

    void set_initial_hidden_state(uint64_t request_id, const ov::Tensor& hidden_state) {
        m_initial_hidden_states[request_id] = hidden_state;
    }

    /**
     * Runs the forward inference call on the underlying LLM's ov::InferRequest, scheduling for inferencing tokens for given sequences
     * taking into account the supplied scheduler output struct.
     * @param sequence_groups A vector of pointers to sequence groups to be processed during this `forward` call
     * @param scheduler_output The scheduler output struct with information on the specifics of the token scheduling during this forward call
     * @return An ov::Tensor with next-token logit scores for each sequence processed during this `forward` call.
     */
    ov::Tensor forward(const std::vector<SequenceGroup::Ptr> & sequence_groups, const Scheduler::Output& scheduler_output) {
        m_sequence_hidden_state_mapping.clear();
        size_t num_sequence_groups = scheduler_output.m_scheduled_sequence_groups_ids.size();

        size_t batch_size_in_sequences = 0;
        size_t total_num_tokens = 0, total_num_blocks = 0;
        size_t max_context_len_val = 0;
        size_t hidden_size = 0;
        bool have_token_type_ids = false;
        OPENVINO_ASSERT(sequence_groups.size() > 0);
        auto sequence_group_type = sequence_groups[0]->get_sequence_group_type();
        if (sequence_group_type == SequenceGroupType::EMBEDDINGS) {
            hidden_size = sequence_groups[0]->get_hidden_size();
        }

        // compute aggregated values
        for (size_t i = 0; i < num_sequence_groups; ++i) {
            size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
            SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id];
            size_t num_sequences = sequence_group->num_running_seqs();
            batch_size_in_sequences += num_sequences;
            total_num_tokens += sequence_group->get_num_scheduled_tokens() * num_sequences;
            total_num_blocks += sequence_group->get_num_blocks() * num_sequences;
            max_context_len_val = std::max(max_context_len_val, sequence_group->get_context_len());
        }

        // Check what shape the model expects for input_ids (may be 1D or 2D depending on model)
        ov::Shape input_ids_shape = {total_num_tokens};
        try {
            auto input_ids_param = m_request.get_compiled_model().input("input_ids");
            auto expected_shape = input_ids_param.get_partial_shape();
            if (expected_shape.rank().is_static() && expected_shape.rank().get_length() == 2) {
                // Model expects 2D, create as [1, total_num_tokens]
                input_ids_shape = {1, total_num_tokens};
                if (cb_verbose()) std::cout << "Model expects 2D input_ids, will create [1, " << total_num_tokens << "]\n";
            }
        } catch (const ov::Exception&) {
            // If input_ids not found, try decoder_input_ids
            try {
                auto input_ids_param = m_request.get_compiled_model().input("decoder_input_ids");
                auto expected_shape = input_ids_param.get_partial_shape();
                if (expected_shape.rank().is_static() && expected_shape.rank().get_length() == 2) {
                    input_ids_shape = {1, total_num_tokens};
                    if (cb_verbose()) std::cout << "Model expects 2D decoder_input_ids, will create [1, " << total_num_tokens << "]\n";
                }
            } catch (const ov::Exception&) {
                // Default to 1D
            }
        }

        // Use cached pre-allocated tensors instead of creating new ones
        ov::Tensor input_ids = _get_or_resize_tensor(m_cached_input_ids, "input_ids", input_ids_shape, ov::element::i64);
        ov::Tensor inputs_embeds = _get_or_resize_tensor(m_cached_inputs_embeds, "inputs_embeds",
            {total_num_tokens, hidden_size}, ov::element::f32);
        // PA specific parameters
        ov::Tensor past_lens = _get_or_resize_tensor(m_cached_past_lens, "past_lens",
            {batch_size_in_sequences}, ov::element::i32);
        ov::Tensor subsequence_begins = _get_or_resize_tensor(m_cached_subsequence_begins, "subsequence_begins",
            {batch_size_in_sequences + 1}, ov::element::i32);
        ov::Tensor block_indices_begins = _get_or_resize_tensor(m_cached_block_indices_begins, "block_indices_begins",
            {batch_size_in_sequences + 1}, ov::element::i32);
        ov::Tensor max_context_len = _get_or_resize_tensor(m_cached_max_context_len, "max_context_len",
            {}, ov::element::i32);
        ov::Tensor token_type_ids = _get_or_resize_tensor(m_cached_token_type_ids, "token_type_ids",
            {1, total_num_tokens}, ov::element::i64);
        ov::Tensor score_aggregation_window = _get_or_resize_tensor(m_cached_score_aggregation_window, "score_aggregation_window",
            {batch_size_in_sequences}, ov::element::i32);

        ov::Tensor hidden_state_input = _prepare_hidden_state_input(total_num_tokens, hidden_size);
        float* hidden_state_data = nullptr;
        if (hidden_state_input) {
            hidden_state_data = hidden_state_input.data<float>();
        }

        ov::Tensor generated_ids_embeds;
        float *generated_ids_embeds_data = nullptr;

        max_context_len.data<int32_t>()[0] = max_context_len_val;

        // get raw pointers to copy to
        float *inputs_embeds_data = nullptr;
        int64_t *input_ids_data = nullptr;
        int64_t *token_type_ids_data = nullptr;

        ov::Tensor position_ids;
        if (sequence_group_type == SequenceGroupType::EMBEDDINGS) {
            inputs_embeds_data = inputs_embeds.data<float>();
            token_type_ids_data = token_type_ids.data<int64_t>();
            auto position_ids_elem = sequence_groups[0]->get_running_sequences()[0]->get_position_ids_list();
            ov::Shape position_ids_shape = position_ids_elem[0].get_shape();
            if (position_ids_shape.size() == 3) {
                position_ids_shape[2] = total_num_tokens;
            }
            else {
                position_ids_shape = {total_num_tokens};
            }
            position_ids = _get_or_resize_tensor(m_cached_position_ids, "position_ids", position_ids_shape, ov::element::i64);
        } else if (sequence_group_type == SequenceGroupType::TOKENS) {
            input_ids_data = input_ids.data<int64_t>();
            position_ids = _get_or_resize_tensor(m_cached_position_ids, "position_ids", {total_num_tokens}, ov::element::i64);
        }

        int64_t
            * position_ids_data = position_ids.data<int64_t>();

        int32_t
            * past_lens_data = past_lens.data<int32_t>(),
            * subsequence_begins_data = subsequence_begins.data<int32_t>(),
            * block_indices_begins_data = block_indices_begins.data<int32_t>(),
            * score_aggregation_window_data = score_aggregation_window.data<int32_t>();

        // sub-sequence data starts with 0
        subsequence_begins_data[0] = 0;
        block_indices_begins_data[0] = 0;

        bool matmul_gathering_is_available = false;
        size_t gathering_current_index = 0;
        std::vector<int64_t> gather_indices_values;
        try {
            std::ignore = m_request.get_tensor("sampled_tokens_indices");
            matmul_gathering_is_available = true;
        } catch (const ov::Exception&) {}

        size_t current_token_idx = 0;
        std::map<size_t, std::set<size_t>> seq_id_to_skipped_blocks_map;
        size_t position_ids_idx = 0;
        
        for (size_t i = 0; i < num_sequence_groups; ++i) {
            size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
            SequenceGroup::Ptr sequence_group = sequence_groups[seq_group_id];
            std::vector<Sequence::Ptr> running_sequences = sequence_group->get_running_sequences();
            size_t num_running_sequences = running_sequences.size();
            size_t num_scheduled_tokens = sequence_group->get_num_scheduled_tokens();
            size_t group_position_id = sequence_group->get_num_processed_tokens();
            size_t prompt_len = sequence_group->get_prompt_len();

            // Next variables are only for sliced matmul case
            size_t output_seq_len = 0;
            const bool echo_output = sequence_group->get_sampling_parameters().echo;
            const bool sampling_is_required = sequence_group->requires_sampling();
            const size_t tokens_to_sample_per_sequence = 1 + sequence_group->get_num_tokens_to_validate();

            for (size_t seq_idx = 0; seq_idx < num_running_sequences; ++seq_idx) {
                // compute token_type_ids for current sequence
                if (sequence_group_type == SequenceGroupType::EMBEDDINGS) {
                    if (auto token_type_ids = sequence_group->get_token_type_ids()) {
                        have_token_type_ids = true;
                        OPENVINO_ASSERT(token_type_ids->size() >= prompt_len, "Token type IDs size is smaller than prompt_len");
                        for (size_t i = 0; i < num_scheduled_tokens; ++i) {
                            token_type_ids_data[i] = (i < prompt_len ? (*token_type_ids)[i] : 0);
                        }
                    }
                }

                output_seq_len = 0;
                Sequence::CPtr sequence = running_sequences[seq_idx];
                if (_is_hs_export()) {
                    size_t start_token_idx = current_token_idx;
                    size_t sequence_length = num_scheduled_tokens;

                    SequenceKey key{sequence_group->get_request_id(), sequence->get_grouped_id()};
                    m_sequence_hidden_state_mapping[key] = HiddenStateRange{start_token_idx, sequence_length};
                }
                if (_is_hs_import()) {
                    auto it = m_initial_hidden_states.find(sequence_group->get_request_id());
                    OPENVINO_ASSERT(it != m_initial_hidden_states.end() && it->second.get_size() > 0,
                                    "Missing initial hidden state for Eagle3 draft model inference.");
                    const auto& stored_hidden_state = it->second;
                    auto stored_shape = stored_hidden_state.get_shape();
                    OPENVINO_ASSERT(stored_shape.size() > 0, "Unexpected hidden state shape for Eagle3 draft model inference.");
                    size_t stored_seq_len = stored_shape[0];
                    size_t stored_hidden_size = stored_shape[stored_shape.size() - 1];

                    OPENVINO_ASSERT(stored_hidden_size == hidden_size, "Target state hidden size does not match the expected size for Eagle3 draft model inference.");
                    OPENVINO_ASSERT(stored_seq_len == total_num_tokens, "Target state sequence length does not match the expected length for Eagle3 draft model inference.");

                    // fill the draft model hidden state input with the target hidden state
                    hidden_state_input = stored_hidden_state;
                } else if (_is_hs_internal()) {
                    // fill hidden_state_data with m_hidden_states
                    if (hidden_state_data) {
                        OPENVINO_ASSERT(num_scheduled_tokens == 1, "unexpected num_scheduled_tokens in speculative drafting stage in eagle3 mode");
                        std::memset(hidden_state_data + current_token_idx * hidden_size,
                                    0,
                                    num_scheduled_tokens * hidden_size * sizeof(float));
                        auto hidden_state = running_sequences[seq_idx]->get_hidden_state();
                        if (hidden_state.get_size() > 0) {
                            auto shape = hidden_state.get_shape();
                            if (shape.size() >= 2 && shape[shape.size() - 1] == hidden_size) {
                                size_t seq_len = shape[0];
                                size_t copy_length = std::min(seq_len, num_scheduled_tokens);

                                size_t src_start_idx = seq_len >= copy_length ? seq_len - copy_length : 0;
                                auto target_shape = ov::Shape{num_scheduled_tokens, 1, hidden_size};
                                ov::Tensor target_base(ov::element::f32, target_shape, hidden_state_data + current_token_idx * hidden_size);
                                _copy_roi_between_tensors(hidden_state, src_start_idx, copy_length, target_base, 0);
                            }
                        }
                    }
                }
                for (size_t token_id = 0, position_id = group_position_id; token_id < num_scheduled_tokens; ++token_id, ++position_id, ++gathering_current_index) {
                    // compute token for current sequence
                    if (sequence_group_type == SequenceGroupType::TOKENS) {
                        int64_t token_value = position_id < prompt_len ?
                            sequence_group->get_prompt_ids()[position_id] :
                            sequence->get_generated_ids()[position_id - prompt_len];
                        input_ids_data[token_id] = token_value;
                        position_ids_data[position_ids_idx] = position_id;
                    } else if (sequence_group_type == SequenceGroupType::EMBEDDINGS) {
                        const auto& generated_embeds = sequence->get_generated_ids_embeds();
                        const float* src = position_id < prompt_len ? sequence_group->get_input_embeds()[position_id].data() :  generated_embeds[position_id - prompt_len].data();
                        std::copy_n(src, hidden_size, inputs_embeds_data + token_id * hidden_size);
                        const auto& position_ids_elem = sequence->get_position_ids_list()[position_id];
                        const auto [begin, end] = Sequence::get_position_ids_elem_coordinates(position_ids_elem.get_shape(), position_ids_idx, false);

                        ov::Tensor dst_roi(position_ids, begin, end);
                        position_ids_elem.copy_to(dst_roi);
                    } else {
                        OPENVINO_THROW("Unknown model inputs type.");
                    }

                    // Check if token gathering is required for the entire sequence group
                    if (matmul_gathering_is_available && (sampling_is_required || echo_output)) {
                        // Determine if the current token should be gathered
                        if (echo_output ||
                            // Skip gathering for prompt tokens
                            group_position_id + token_id >= prompt_len - 1 &&
                            // Gather only the last scheduled token or 1 + num_tokens_to_validate tokens for SD
                            // In SD, tokens_to_sample_per_sequence may exceed num_scheduled_tokens
                            token_id + tokens_to_sample_per_sequence >= num_scheduled_tokens) {
                            gather_indices_values.push_back(gathering_current_index);
                            output_seq_len++;
                        }
                    }
                    position_ids_idx++;
                }

                size_t num_blocks = sequence_group->get_num_logical_blocks();
                size_t expected_kv_cache_size = sequence_group->get_num_processed_tokens() - sequence_group->get_num_evicted_tokens();
                size_t num_past_blocks_to_ignore = 0;

                if (scheduler_output.m_apply_sparse_attention_mask) {
                    auto it = scheduler_output.m_sparse_attention_skipped_logical_blocks.find(sequence->get_id());
                    if (it != scheduler_output.m_sparse_attention_skipped_logical_blocks.end()) {
                        seq_id_to_skipped_blocks_map[sequence->get_id()] = it->second;
                        num_past_blocks_to_ignore = seq_id_to_skipped_blocks_map[sequence->get_id()].size();
                    }
                }

                OPENVINO_ASSERT(num_blocks >= num_past_blocks_to_ignore);
                size_t num_blocks_utilized = num_blocks - num_past_blocks_to_ignore;

                past_lens_data[0] = expected_kv_cache_size - num_past_blocks_to_ignore * m_block_size;

                subsequence_begins_data[1] = subsequence_begins_data[0] + num_scheduled_tokens;

                block_indices_begins_data[1] = block_indices_begins_data[0] + num_blocks_utilized;

                // Diagnostic: print PA inputs for every scheduled sequence
                if (cb_verbose()) std::cout << "[PA_INPUTS] req=" << sequence_group->get_request_id()
                          << " seq_id=" << sequence->get_id()
                          << " sched_tok=" << num_scheduled_tokens
                          << " past_lens=" << past_lens_data[0]
                          << " subseq_begin=[" << subsequence_begins_data[0] << "," << subsequence_begins_data[1] << ")"
                          << " blk_begin=[" << block_indices_begins_data[0] << "," << block_indices_begins_data[1] << ")"
                          << " num_blocks=" << num_blocks_utilized
                          << " processed_tok=" << sequence_group->get_num_processed_tokens()
                          << " prompt_len=" << sequence_group->get_prompt_len()
                          << " requires_sampling=" << sequence_group->requires_sampling()
                          << "\n";

                // apply strides to shift to a next sequence
                if (sequence_group_type == SequenceGroupType::TOKENS) {
                    input_ids_data += num_scheduled_tokens;
                } else if (sequence_group_type == SequenceGroupType::EMBEDDINGS) {
                    inputs_embeds_data += num_scheduled_tokens * hidden_size;
                    if (have_token_type_ids)
                        token_type_ids_data += num_scheduled_tokens;
                }

                if (m_is_aggregate_attention_scores) {
                    size_t seq_id = sequence->get_id();
                    auto it = scheduler_output.m_score_aggregation_windows.find(seq_id);
                    if (it != scheduler_output.m_score_aggregation_windows.end()) {
                        *score_aggregation_window_data = it->second; // the prompt has reached the SnapKV window, either fully or partially
                    }
                    else {
                        // either the prompt has not reached the SnapKV window yet (in which case we will disregard the scores anyway),
                        // or the sequence is in the generation stage already
                        *score_aggregation_window_data = 1;
                    }
                }
                current_token_idx += num_scheduled_tokens;
                past_lens_data += 1;
                subsequence_begins_data += 1;
                block_indices_begins_data += 1;
                score_aggregation_window_data += 1;
            }
            sequence_group->set_output_seq_len(matmul_gathering_is_available ? output_seq_len : num_scheduled_tokens);
        }
        
        // Note: A ireq will pre-allocate a USM for each model's input. For tensor optimization, we cache pre-allocated USM gotten from a ireq for these tensors.
        // Since these tensors(except score_aggregation_window) are gotten from a ireq, there's no need to set them again.
        // Score_aggregation_window might be not managed through the cached tensor system in some case as it is created unconditionally, and need to be set to a ireq.
        // To align these tensors' behavior, set each tensor when it is not cached.

        // Build cross-attention K/V inputs for segmented cross-attention.
        //
        // FAST PATH (CROSS_KV_CACHE=1, default): K/V projections were run once at
        // request admission and are cached in CrossKVCache. Gather the per-request
        // slots into [N, n_heads, T_enc, head_dim] and set 64 parameters.
        // Cost: N × n_layers × 2 × slot_bytes gather + 64 zero-copy set_tensor calls.
        //
        // SLOW PATH (CROSS_KV_CACHE=0 or cache unavailable): assemble
        // [N, T_enc, D] from per-request encoder hidden states and let the
        // decoder re-run all K/V projections inside infer(). This is the original
        // behaviour (2256 ms at N=5).
        {
            const auto t_asm_start = std::chrono::steady_clock::now();

            // ── sub-timer lambdas ─────────────────────────────────────────────
            using ClockT = std::chrono::steady_clock;
            auto us_since = [](ClockT::time_point t0) {
                return std::chrono::duration<double, std::micro>(ClockT::now() - t0).count();
            };

            // Sub-step accumulators (printed when assembly finishes)
            double t_build_ids_us   = 0, t_has_slot_us  = 0,
                   t_gather_us      = 0, t_set_tensor_us = 0,
                   t_slow_path_us   = 0;
            bool   did_set_tensor   = false;
            bool   used_fast_path   = false;

            // ── 1. Build batch_req_ids ────────────────────────────────────────
            auto t0 = ClockT::now();
            std::vector<uint64_t> batch_req_ids;
            batch_req_ids.reserve(num_sequence_groups);
            for (size_t i = 0; i < num_sequence_groups; ++i) {
                size_t sg_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
                batch_req_ids.push_back(sequence_groups[sg_id]->get_request_id());
            }
            t_build_ids_us = us_since(t0);

            if (cb_verbose()) {
                std::cout << "[ENC_BATCH] scheduled_ids=[";
                for (size_t i = 0; i < batch_req_ids.size(); ++i) {
                    std::cout << "req" << batch_req_ids[i];
                    if (i + 1 < batch_req_ids.size()) std::cout << ",";
                }
                std::cout << "]\n";
            }

            // ── 2. has_slot check ─────────────────────────────────────────────
            t0 = ClockT::now();
            bool all_have_slots = m_cross_kv_cache != nullptr &&
                std::all_of(batch_req_ids.begin(), batch_req_ids.end(),
                    [this](uint64_t rid) { return m_cross_kv_cache->has_slot(rid); });
            t_has_slot_us = us_since(t0);

            if (all_have_slots) {
                used_fast_path = true;

                // ── 3. gather_batch ───────────────────────────────────────────
                t0 = ClockT::now();
                bool need_set_tensor = false;
                const auto& batch_kv = m_cross_kv_cache->gather_batch(batch_req_ids, need_set_tensor);
                t_gather_us = us_since(t0);

                // ── 4. set_tensor (only on N-change) ─────────────────────────
                if (need_set_tensor) {
                    did_set_tensor = true;
                    t0 = ClockT::now();
                    for (size_t l = 0; l < m_cross_kv_cache->n_layers(); ++l) {
                        m_request.set_tensor("cross_kv_K_" + std::to_string(l), batch_kv[l][0]);
                        m_request.set_tensor("cross_kv_V_" + std::to_string(l), batch_kv[l][1]);
                    }
                    t_set_tensor_us = us_since(t0);
                }

                // ── 5. encoder_hidden_states ──────────────────────────────────
                // encoder_hidden_states is removed from the decoder model by graph surgery
                // (extract_cross_attn_projector step 5 calls remove_parameter(enc_param)).
                // Nothing to set here — cross-KV cache tensors above are the only decoder
                // inputs carrying encoder context.

                if (cb_verbose())
                    std::cout << "[ENC_BATCH] FAST PATH: "
                              << (need_set_tensor ? "set_tensor (N changed)" : "memcpy or cached")
                              << " N=" << batch_req_ids.size() << "\n";
            } else {
                // ── SLOW PATH ─────────────────────────────────────────────────
                t0 = ClockT::now();
                std::vector<ov::Tensor> enc_states;
                enc_states.reserve(num_sequence_groups);
                for (size_t i = 0; i < num_sequence_groups; ++i) {
                    size_t sg_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
                    auto opt = sequence_groups[sg_id]->get_encoder_hidden_state();
                    if (opt.has_value())
                        enc_states.push_back(opt.value());
                }
                if (!enc_states.empty()) {
                    const auto& ref = enc_states[0];
                    OPENVINO_ASSERT(ref.get_shape().size() == 3,
                        "encoder_hidden_state must be rank-3 [1, T_enc, D]");
                    const size_t T_enc = ref.get_shape()[1];
                    const size_t D     = ref.get_shape()[2];
                    const size_t N     = enc_states.size();
                    if (cb_verbose())
                        std::cout << "[ENC_STATE] SLOW PATH N=" << N
                                  << " type=" << ref.get_element_type()
                                  << " shape=[" << N << "," << T_enc << "," << D << "]\n";
                    const size_t bytes_per_seq = ref.get_byte_size();
                    ov::Tensor batched(ref.get_element_type(), {N, T_enc, D});
                    uint8_t* dst = static_cast<uint8_t*>(batched.data());
                    for (size_t n = 0; n < N; ++n) {
                        OPENVINO_ASSERT(enc_states[n].get_shape()[1] == T_enc &&
                                       enc_states[n].get_shape()[2] == D,
                            "encoder_hidden_state shape mismatch across scheduled sequence groups");
                        std::memcpy(dst + n * bytes_per_seq,
                                    static_cast<const uint8_t*>(enc_states[n].data()),
                                    bytes_per_seq);
                    }
                    if (!m_enc_hs_checked) {
                        m_has_encoder_hidden_states = false;
                        for (const auto& inp : m_request.get_compiled_model().inputs()) {
                            for (const auto& n : inp.get_names())
                                if (n.find("encoder_hidden_states") != std::string::npos) {
                                    m_has_encoder_hidden_states = true; break;
                                }
                            if (m_has_encoder_hidden_states) break;
                        }
                        m_enc_hs_checked = true;
                        if (!m_has_encoder_hidden_states)
                            std::cout << "[ModelRunner] encoder_hidden_states not in compiled model — slow path disabled\n";
                    }
                    if (m_has_encoder_hidden_states)
                        m_request.set_tensor("encoder_hidden_states", batched);
                }
                t_slow_path_us = us_since(t0);
            }

            m_last_assembly_us = std::chrono::duration<double, std::micro>(
                ClockT::now() - t_asm_start).count();

            // ── Always-on sub-step breakdown (printed unconditionally so we can ──
            // ── diagnose the 1050ms overhead without verbose flood)             ──
            // Accumulate into per-step detail string for the step summary line.
            // Only print when total assembly exceeds 1ms to avoid spamming PROMPT steps.
            if (m_last_assembly_us > 1000.0) {
                const double total = m_last_assembly_us;
                std::cout << "[ASM_DETAIL] total=" << total << "us"
                          << "  build_ids=" << t_build_ids_us << "us"
                          << "  has_slot="  << t_has_slot_us  << "us"
                          << "  path=" << (used_fast_path ? "FAST" : "SLOW");
                if (used_fast_path) {
                    std::cout << "  gather="     << t_gather_us     << "us"
                              << "  set_tensor=" << t_set_tensor_us << "us"
                              << " (called=" << (did_set_tensor ? "yes" : "no") << ")";
                } else {
                    std::cout << "  slow_path=" << t_slow_path_us << "us";
                }
                std::cout << "  N=" << batch_req_ids.size() << "\n";
            }
        }
        
        // For input_ids, always set it (don't rely on cache) to handle potential rank mismatches
        if (sequence_group_type == SequenceGroupType::TOKENS) {
            m_request.set_tensor("input_ids", input_ids);
        }
        else if (sequence_group_type == SequenceGroupType::EMBEDDINGS) {
            if (!m_cached_inputs_embeds) {
                m_request.set_tensor("inputs_embeds", inputs_embeds);
            }
            if (have_token_type_ids && !m_cached_token_type_ids) {
                m_request.set_tensor("token_type_ids", token_type_ids);
            }
        }
        if (hidden_state_input && hidden_state_input.get_size() > 0) {
            m_request.set_tensor("hidden_states", hidden_state_input);
        }
        if (position_ids.get_shape().size() == 3) {
            // flatten positions ids for 3D position ids case
            position_ids.set_shape({ov::shape_size(position_ids.get_shape())});
        }
        
        // typical LLM parameters
        if (!m_cached_position_ids) {
            m_request.set_tensor("position_ids", position_ids);
        }
        // Whisper-specific: cache_position has the same semantics as position_ids.
        // The CB model_runner never set it, leaving it as the empty {0} reset tensor
        // which caused all downstream Q/K/V shapes to be zero-dim → PA never executed.
        if (!m_cache_position_checked) {
            m_has_cache_position = false;
            for (const auto& inp : m_request.get_compiled_model().inputs()) {
                for (const auto& n : inp.get_names())
                    if (n == "cache_position") { m_has_cache_position = true; break; }
                if (m_has_cache_position) break;
            }
            m_cache_position_checked = true;
        }
        if (m_has_cache_position) {
            m_request.set_tensor("cache_position", position_ids);
        }
        // PA specific parameters
        if (!m_cached_past_lens) {
            m_request.set_tensor("past_lens", past_lens);
        }
        if (!m_cached_subsequence_begins) {
            m_request.set_tensor("subsequence_begins", subsequence_begins);
        }

        _set_block_indices(sequence_groups, scheduler_output, total_num_blocks, seq_id_to_skipped_blocks_map);

        if (!m_cached_block_indices_begins) {
            m_request.set_tensor("block_indices_begins", block_indices_begins);
        }
        if (!m_cached_max_context_len) {
            m_request.set_tensor("max_context_len", max_context_len);
        }
        if (m_is_use_rotation_inputs) {
            m_request.set_tensor("rotation_trig_lut", m_cache_rotation_trig_lut);
            _set_cache_rotation_coefficients(sequence_groups, scheduler_output);
        }

        if (m_is_use_xattention_inputs) {
            _set_xattention_tensors(sequence_groups, scheduler_output, batch_size_in_sequences);
        }

        if (m_is_use_adaptive_rkv) {
            _set_adaptive_rkv_tensors(sequence_groups, scheduler_output, batch_size_in_sequences);
        }

        if (matmul_gathering_is_available) {
            // use pre-allocated tensor for gather_indices as well
            ov::Tensor gather_indices = m_request.get_tensor("sampled_tokens_indices");
            gather_indices.set_shape({gather_indices_values.size()});
            if (gather_indices.get_element_type() == ov::element::i32) {
                // Model expects int32 but gather_indices_values is int64 - must convert
                int32_t* dst = gather_indices.data<int32_t>();
                for (size_t k = 0; k < gather_indices_values.size(); ++k) {
                    dst[k] = static_cast<int32_t>(gather_indices_values[k]);
                }
            } else {
                std::memcpy(gather_indices.data(), gather_indices_values.data(), gather_indices_values.size() * sizeof(int64_t));
            }
        }

        if (m_is_aggregate_attention_scores && !m_cached_score_aggregation_window) {
            m_request.set_tensor("score_aggregation_window", score_aggregation_window);
        }

        {
            // Confirm encoder_hidden_states batch matches scheduled N each step.
            if (cb_verbose()) {
                try {
                    auto enc_t = m_request.get_tensor("encoder_hidden_states");
                    std::cout << "[DBG infer] encoder_hidden_states shape: " << enc_t.get_shape() << "\n";
                } catch (...) {}
            }

            static ManualTimer timer("pure generate inference");
            timer.start();
            // Sentinel test: pre-fill logits with 42.0 to detect if OV writes to this tensor
            try {
                ov::Tensor logits_pre = m_request.get_tensor("logits");
                if (logits_pre.get_size() > 0 && logits_pre.get_element_type() == ov::element::f32)
                    std::fill(logits_pre.data<float>(), logits_pre.data<float>() + logits_pre.get_size(), 42.f);
            } catch (...) {}
            try {
                const auto t_infer_start = std::chrono::steady_clock::now();
                m_request.infer();
                m_last_infer_us = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - t_infer_start).count();
                timer.end();

                
            } catch (const std::exception& e) {
                timer.end();
                std::cerr << "\n!!! INFERENCE FAILED WITH EXCEPTION !!!\n";
                std::cerr << "Exception type: " << typeid(e).name() << "\n";
                std::cerr << "Exception message: " << e.what() << "\n";
                std::cerr << "This error occurred during model inference.\n";
                
                // Try to get profiling info to see which operation was executing when it failed
                try {
                    auto profiling_info = m_request.get_profiling_info();
                    if (!profiling_info.empty()) {
                        std::cerr << "\nOperations executed before failure (last 10):\n";
                        size_t start_idx = profiling_info.size() > 10 ? profiling_info.size() - 10 : 0;
                        for (size_t i = start_idx; i < profiling_info.size(); ++i) {
                            const auto& info = profiling_info[i];
                            std::cerr << "  [" << i << "] " << info.node_name << " (" << info.node_type << ")\n";
                        }
                    }
                } catch (...) {}
                
                throw;  // Re-throw to propagate the error
            } catch (...) {
                timer.end();
                std::cerr << "\n!!! INFERENCE FAILED WITH UNKNOWN EXCEPTION !!!\n";
                std::cerr << "An unknown exception occurred during model inference.\n";
                throw;  // Re-throw to propagate the error
            }
        }

        if (m_collect_attention_scores) {
            _collect_attention_scores(sequence_groups, scheduler_output);
        }

        if (m_is_use_adaptive_rkv) {
            _collect_adaptive_rkv_block_diversities(sequence_groups, scheduler_output);
        }

        _reset_cache_rotation_coefficients();

        if (_is_hs_export()) {
            m_hidden_states = m_request.get_tensor("last_hidden_state");
            for (size_t i = 0; i < num_sequence_groups; ++i) {
                size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
                SequenceGroup::Ptr sequence_group = sequence_groups[seq_group_id];
                std::vector<Sequence::Ptr> running_sequences = sequence_group->get_running_sequences();
                for (size_t seq_idx = 0; seq_idx < running_sequences.size(); ++seq_idx) {
                    Sequence::Ptr sequence = running_sequences[seq_idx];
                    sequence->update_hidden_state(
                        _get_hidden_state(sequence_group->get_request_id(), sequence->get_grouped_id()));
                }
            }
        }
        // return logits
        auto logits_tensor = m_request.get_tensor("logits");

        if (cb_verbose()) {
        std::cout << "=== ModelRunner::forward END ===\n";
        
        std::cout << "DEBUG: Extracting logits tensor...\n";
        std::cout << "  Logits shape: [";
        for (size_t i = 0; i < logits_tensor.get_shape().size(); ++i) {
            std::cout << logits_tensor.get_shape()[i];
            if (i < logits_tensor.get_shape().size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "  Logits element type: " << logits_tensor.get_element_type() << "\n";
        
        if (logits_tensor.get_size() > 0) {
            const float* logits_data = logits_tensor.data<const float>();
            
            if (logits_tensor.get_shape().size() == 3) {
                size_t batch = logits_tensor.get_shape()[0];
                size_t seq_len = logits_tensor.get_shape()[1];
                size_t vocab_size = logits_tensor.get_shape()[2];
                
                // Check if ALL values are zero
                bool all_zero = true;
                for (size_t i = 0; i < logits_tensor.get_size() && all_zero; ++i) {
                    if (logits_data[i] != 0.0f) all_zero = false;
                }
                std::cout << "  ALL values zero: " << (all_zero ? "YES" : "NO") << "\n";
                
                // Print max logit for EACH token position
                for (size_t s = 0; s < seq_len; ++s) {
                    const float* pos_logits = logits_data + s * vocab_size;
                    auto max_it = std::max_element(pos_logits, pos_logits + vocab_size);
                    size_t max_idx = max_it - pos_logits;
                    float max_val = *max_it;
                    // Check if this position is all zeros
                    bool pos_all_zero = true;
                    for (size_t v = 0; v < std::min<size_t>(100, vocab_size); ++v) {
                        if (pos_logits[v] != 0.0f) { pos_all_zero = false; break; }
                    }
                    std::cout << "  Position[" << s << "]: argmax=" << max_idx 
                              << ", max_logit=" << max_val
                              << ", all_zero=" << (pos_all_zero ? "YES" : "NO")
                              << ", first5=[" << pos_logits[0] << "," << pos_logits[1] << "," 
                              << pos_logits[2] << "," << pos_logits[3] << "," << pos_logits[4] << "]\n";
                }
            }
        }
        
        // Inspect cross-attention states (ReadValueWithSubgraph outputs)
        {
            auto& outputs = m_request.get_compiled_model().outputs();
            std::cout << "  Model has " << outputs.size() << " outputs total\n";
            bool found_nonzero_output = false;
            for (const auto& out : outputs) {
                const std::string& name = out.get_any_name();
                // Skip logits, check everything else
                if (name == "logits") continue;
                try {
                    auto t = m_request.get_tensor(name);
                    if (t.get_size() > 0 && t.get_element_type() == ov::element::f32) {
                        const float* d = t.data<const float>();
                        bool has_nonzero = false;
                        for (size_t i = 0; i < std::min<size_t>(50, t.get_size()); ++i) {
                            if (d[i] != 0.0f) { has_nonzero = true; break; }
                        }
                        if (has_nonzero) {
                            found_nonzero_output = true;
                            // Helper: per-position stats, reports absmax/L2/first-val
                            auto print_pos_stats = [&](const float* base, size_t H, size_t stride, size_t N, const std::string& tag) {
                                for (size_t p = 0; p < N; ++p) {
                                    const float* ptr = base + p * stride;
                                    float absmax = 0.f, sum2 = 0.f;
                                    for (size_t h = 0; h < H; ++h) {
                                        absmax = std::max(absmax, std::abs(ptr[h]));
                                        sum2 += ptr[h] * ptr[h];
                                    }
                                    std::cout << "    " << tag << " pos[" << p << "]: absmax=" << absmax
                                              << " L2=" << std::sqrt(sum2)
                                              << " first3=[" << ptr[0] << "," << ptr[1] << "," << ptr[2] << "]\n";
                                }
                            };
                            const auto& sh = t.get_shape();
                            // DEBUG_lm_input: [1, N, H]   — final LayerNorm output
                            // DEBUG_after_xattn_LX: [1, N, H] or [N, H] — after cross-attn residual
                            if ((name == "DEBUG_lm_input" || name.find("DEBUG_after_xattn_L") == 0) && sh.size() >= 2) {
                                size_t H = sh.back();
                                size_t N = sh.size() == 3 ? sh[1] : sh[0];
                                // For shape [1,N,H]: d starts at batch 0, so offset 0
                                if (N > 1) {  // only print when co-batching (avoids solo-step noise)
                                    std::cout << "  Output '" << name << "' shape=" << sh << "\n";
                                    print_pos_stats(d, H, H, N, name);
                                }
                            }
                            // DEBUG_xattn_LX: [1, n_heads, N, head_dim] — cross-attn SDPA output
                            else if (name.find("DEBUG_xattn_L") == 0 && sh.size() == 4) {
                                size_t n_heads   = sh[1];
                                size_t N         = sh[2];
                                size_t head_dim  = sh[3];
                                if (N > 1) {  // only print when co-batching
                                    std::cout << "  Output '" << name << "' shape=" << sh << "\n";
                                    // Per-position: gather non-contiguous elements [0, h, p, d]
                                    for (size_t p = 0; p < N; ++p) {
                                        float absmax = 0.f, sum2 = 0.f;
                                        for (size_t h = 0; h < n_heads; ++h) {
                                            const float* ptr = d + h * N * head_dim + p * head_dim;
                                            for (size_t dd = 0; dd < head_dim; ++dd) {
                                                absmax = std::max(absmax, std::abs(ptr[dd]));
                                                sum2 += ptr[dd] * ptr[dd];
                                            }
                                        }
                                        // first 3 values from head 0
                                        const float* h0 = d + p * head_dim;
                                        std::cout << "    " << name << " pos[" << p << "]: absmax=" << absmax
                                                  << " L2=" << std::sqrt(sum2)
                                                  << " h0_first3=[" << h0[0] << "," << h0[1] << "," << h0[2] << "]\n";
                                    }
                                }
                            }
                        }
                    }
                } catch (...) {}
            }
            if (!found_nonzero_output) {
                std::cout << "  WARNING: All model outputs are zero!\n";
            }
        }
        std::cout << "\n";
        } // end cb_verbose() block
        
        return logits_tensor;
    }

    void append_embeddings(const std::vector<SequenceGroup::Ptr> & sequence_groups, const Scheduler::Output& scheduler_output) {
        size_t num_sequence_groups = scheduler_output.m_scheduled_sequence_groups_ids.size();
        size_t num_generated_ids_without_embeddings = 0;
        OPENVINO_ASSERT(sequence_groups.size() > 0);

        // compute aggregated values
        for (size_t i = 0; i < num_sequence_groups; ++i) {
            size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
            SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id];
            size_t num_sequences = sequence_group->num_running_seqs();
            OPENVINO_ASSERT(sequence_group->get_sequence_group_type() == SequenceGroupType::EMBEDDINGS);
            for (auto seq: sequence_group->get_running_sequences()) {
                num_generated_ids_without_embeddings += seq->get_generated_len() - seq->get_generated_ids_embeds().size();
            }
        }
        size_t hidden_size = sequence_groups[0]->get_hidden_size();

        ov::Tensor generated_ids_embeds;
        float *generated_ids_embeds_data = nullptr;

        ov::Tensor generated_ids = ov::Tensor(ov::element::i64, {1, num_generated_ids_without_embeddings});

        int64_t *generated_ids_data = generated_ids.data<int64_t>();
        size_t pos = 0;
        for (size_t i = 0; i < num_sequence_groups; ++i) {
            size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
            SequenceGroup::Ptr sequence_group = sequence_groups[seq_group_id];
            for (auto seq: sequence_group->get_running_sequences()) {
                const auto& generated_ids = seq->get_generated_ids();
                for (size_t token_idx = seq->get_generated_ids_embeds().size(); token_idx < generated_ids.size(); token_idx++) {
                    generated_ids_data[pos] = generated_ids[token_idx];
                    pos++;

                    size_t position_id = token_idx + sequence_group->get_prompt_len();
                    auto new_position_ids = m_inputs_embedder->get_generation_phase_position_ids(1, position_id, seq->get_rope_delta()).first;
                    seq->append_position_ids(new_position_ids);
                }
            }
        }
        if (pos > 0) {
            CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
            EmbeddingsRequest& req = embeddings_request_guard.get();
            generated_ids_embeds = m_embedding->infer(req, generated_ids);
            generated_ids_embeds_data = generated_ids_embeds.data<float>();
            size_t embeds_pos = 0;
            for (size_t i = 0; i < num_sequence_groups; ++i) {
                size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
                SequenceGroup::Ptr sequence_group = sequence_groups[seq_group_id];
                for (auto seq: sequence_group->get_running_sequences()) {
                    auto generated_ids = seq->get_generated_ids();
                    size_t new_embeds_count = seq->get_generated_len() - seq->get_generated_ids_embeds().size();
                    ov::Coordinate start{0, embeds_pos, 0};
                    ov::Coordinate end{1, embeds_pos + new_embeds_count, hidden_size};
                    ov::Tensor embedding(generated_ids_embeds, start, end);
                    seq->append_generated_ids_embeds(embedding);
                    embeds_pos += new_embeds_count;
                }
            }
        }
    }

private:
    ov::Tensor m_hidden_states;

    // Hidden state flags and helpers
    bool _is_hs_export()   const { return m_hidden_state_flags & HS_EXPORT; }
    bool _is_hs_import()   const { return m_hidden_state_flags & HS_IMPORT; }
    bool _is_hs_internal() const { return m_hidden_state_flags & HS_INTERNAL; }

    /**
     * @brief Retrieves a slice of the hidden state tensor corresponding to a specific request and sequence group.
     *
     * This method looks up the hidden state mapping for the given request and sequence group IDs.
     * If the mapping exists and the hidden states tensor is available, it returns a sub-tensor (region of interest)
     * representing the hidden state for the specified sequence. If the mapping does not exist or the hidden states
     * tensor is empty, an empty tensor is returned.
     *
     * @param request_id        The unique identifier for the request.
     * @param seq_grouped_id    The identifier for the sequence group within the request.
     * @return ov::Tensor       The tensor slice representing the hidden state for the specified sequence,
     *                          or an empty tensor if not found.
     */
    ov::Tensor _get_hidden_state(uint64_t request_id, uint64_t seq_grouped_id) const {
        if (m_hidden_states.get_size() == 0) {
            return ov::Tensor();
        }

        SequenceKey key{request_id, seq_grouped_id};
        const auto it = m_sequence_hidden_state_mapping.find(key);
        if (it == m_sequence_hidden_state_mapping.end()) {
            return ov::Tensor();
        }

        size_t start_idx = it->second.start_token_idx;
        size_t length = it->second.length;

        auto shape = m_hidden_states.get_shape();
        OPENVINO_ASSERT(shape.size() >= 2,
                        "Hidden states tensor rank is less than 2.");

        auto [start_coord, end_coord] = ov::genai::utils::make_roi(shape, 0, start_idx, start_idx + length);
        return ov::Tensor(m_hidden_states, start_coord, end_coord);
    }

    /**
     * @brief Prepares and returns a hidden state input tensor for the model.
     *
     * This function checks if hidden state input is required based on the internal flags.
     * If required, it determines the hidden size from the initial hidden states if not already provided.
     * It then creates and returns a tensor of shape [total_num_tokens, 1, hidden_size] initialized to zeros.
     *
     * @param total_num_tokens The total number of tokens for which the hidden state tensor is to be created.
     * @param hidden_size [in/out] The size of the hidden state. If set to 0, it will be inferred from the initial hidden states.
     * @return ov::Tensor The prepared hidden state tensor, or an empty tensor if not applicable.
     */
    ov::Tensor _prepare_hidden_state_input(size_t total_num_tokens,
                                           size_t& hidden_size /*in/out*/) {
        if (!(m_hidden_state_flags & (HS_IMPORT | HS_INTERNAL))) {
            return {};
        }

        if (hidden_size == 0) {
            for (const auto& kv : m_initial_hidden_states) {
                const auto& initial_hidden_states = kv.second;
                auto hidden_states_shape = initial_hidden_states.get_shape();
                OPENVINO_ASSERT(initial_hidden_states && hidden_states_shape.size() >= 2,
                                "Initial hidden states tensor rank is less than 2.");
                hidden_size = hidden_states_shape.back();
                break;
            }
        }
        if (hidden_size == 0) {
            return {};
        }

        ov::Tensor hs(ov::element::f32, {total_num_tokens, 1, hidden_size});
        std::memset(hs.data<float>(), 0, hs.get_byte_size());
        return hs;
    }

    // Common helper to copy a contiguous slice (first-dim range) from src to dst using ROI tensors.
    // src_start_idx: start index along src first dimension
    // copy_length: number of elements along first dim to copy
    // dst_base: destination base tensor (may be full buffer or a wrapper around a raw pointer)
    // dst_first_dim_start: start index in first dimension of dst_base where copy should be placed
    static void _copy_roi_between_tensors(const ov::Tensor& src,
                                         size_t src_start_idx,
                                         size_t copy_length,
                                         const ov::Tensor& dst_base,
                                         size_t dst_first_dim_start) {
        if (copy_length == 0) {
            return;
        }

        // prepare source ROI coords
        const auto src_shape = src.get_shape();
        OPENVINO_ASSERT(!src_shape.empty(), "source tensor rank is zero");
        auto [src_start, src_end] = ov::genai::utils::make_roi(src_shape, 0, src_start_idx, src_start_idx + copy_length);
        ov::Tensor src_roi(src, src_start, src_end);

        // prepare destination ROI coords
        const auto dst_shape = dst_base.get_shape();
        OPENVINO_ASSERT(!dst_shape.empty(), "destination tensor rank is zero");
        auto [tgt_start, tgt_end] = ov::genai::utils::make_roi(dst_shape, 0, dst_first_dim_start, dst_first_dim_start + copy_length);
        ov::Tensor tgt_roi(dst_base, tgt_start, tgt_end);

        // bulk copy
        src_roi.copy_to(tgt_roi);
    }

    ov::Tensor _get_or_resize_tensor(ov::Tensor& cached_tensor, 
                                   const std::string& tensor_name,
                                   const ov::Shape& required_shape,
                                   ov::element::Type element_type) {
       // Determine actual tensor name (it might differ from requested, e.g., decoder_input_ids vs input_ids)
       std::string actual_tensor_name = tensor_name;
       std::vector<std::string> names_to_try = {tensor_name};
       if (tensor_name == "input_ids") {
           names_to_try.push_back("decoder_input_ids");
       }
       
       // Always find the actual tensor name in the model, even if cached
       bool tensor_found = false;
       for (const auto& name : names_to_try) {
           try {
               auto test_input = m_request.get_compiled_model().input(name);
               actual_tensor_name = name;
               tensor_found = true;
               break;
           } catch (const ov::Exception&) {
               // Try next name
           }
       }
       
       if (!cached_tensor) {
            // If cached tensor is not initialized, try to get the tensor from the m_request.
            if (tensor_found) {
                try {
                    cached_tensor = m_request.get_tensor(actual_tensor_name);
                    if (cb_verbose()) std::cout << "Got tensor from request: " << actual_tensor_name << " (requested as " << tensor_name << ")\n";
                } catch (const ov::Exception&) {
                    if (cb_verbose()) std::cout << "Failed to get tensor " << actual_tensor_name << " from request\n";
                    return ov::Tensor(element_type, required_shape);
                }
            } else {
                // Fall back to default construction methods when exception occurs.
                // For example, score_aggregation_window may not be used by a model but a Tensor is required for following operation.
                if (cb_verbose()) std::cout << "Fall back initiated for tensor: " << tensor_name << "\n";
                return ov::Tensor(element_type, required_shape);
            }
       }
       
       // Get the target tensor's partial shape to check its rank
       ov::PartialShape target_shape;
       try {
           // Get the input info from the compiled model using the actual tensor name
           target_shape = m_request.get_compiled_model().input(actual_tensor_name).get_partial_shape();
           if (cb_verbose()) std::cout << "Parameter " << actual_tensor_name << " has shape: " << target_shape << "\n";
       } catch (...) {
           // If we can't get shape info, proceed with direct set_shape
           target_shape = ov::PartialShape::dynamic();
       }
       
       ov::Shape shape_to_set = required_shape;
       
       if (cb_verbose()) std::cout << "Setting shape for " << actual_tensor_name << ": required=" << required_shape 
                << ", target_expects=" << target_shape << ", shape_to_set=" << shape_to_set << "\n";
       
       if (cached_tensor.get_shape() != shape_to_set) {
            try {
                cached_tensor.set_shape(shape_to_set);
            } catch (const ov::Exception& e) {
                OPENVINO_THROW("set_shape failed for tensor: ", tensor_name, ". Error: ", e.what());
            }
        }
        
        return cached_tensor;
    }

    // Fills indices for sequences in the order defined by scheduler_output
    void _fill_indices_from_block_tables(
        const std::vector<std::string>& dst_tensor_names,
        const std::vector<SequenceGroup::Ptr>& sequence_groups,
        const Scheduler::Output& scheduler_output,
        const std::vector<std::map<size_t, std::vector<size_t>>>& seq_id_to_select_logical_idx_maps) {
        OPENVINO_ASSERT(seq_id_to_select_logical_idx_maps.size() == dst_tensor_names.size() ||
                        (dst_tensor_names.size() == 1 && !m_is_use_per_layer_cache_control) ||
                        seq_id_to_select_logical_idx_maps.empty());
        bool is_fill_all = seq_id_to_select_logical_idx_maps.empty();
        if (cb_verbose()) {
            std::cout << "\n--- _fill_indices_from_block_tables called ---\n";
            std::cout << "Block tables in scheduler_output for sequences: ";
            for (const auto& entry : scheduler_output.m_block_tables) {
                std::cout << entry.first << " ";
            }
            std::cout << "\n";
            std::cout << "seq_id_to_select_logical_idx_maps.size()=" << seq_id_to_select_logical_idx_maps.size() << "\n";
            std::cout << "dst_tensor_names.size()=" << dst_tensor_names.size() << "\n";
            std::cout << "is_fill_all=" << (is_fill_all ? "true" : "false") << "\n";
        }
        size_t num_sequence_groups = scheduler_output.m_scheduled_sequence_groups_ids.size();
        std::vector<size_t> filled_blocks_per_layer(dst_tensor_names.size(), 0);


        for (size_t layer_idx = 0; layer_idx < dst_tensor_names.size(); layer_idx++) {
            auto input_tensor = m_request.get_tensor(dst_tensor_names[layer_idx]);
            auto block_indices_data = input_tensor.data<int32_t>();
            for (size_t i = 0; i < num_sequence_groups; ++i) {
                size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
                SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id];
                std::vector<Sequence::CPtr> running_sequences = sequence_group->get_running_sequences();
                size_t num_running_sequences = running_sequences.size();

                for (size_t i = 0; i < num_running_sequences; ++i) {
                    Sequence::CPtr sequence = running_sequences[i];
                    size_t seq_id = sequence->get_id();
                    if (cb_verbose()) {
                        std::cout << "  Filling blocks for seq_id=" << seq_id << " (layer " << layer_idx << ")";
                        const auto& kv_blocks = scheduler_output.m_block_tables.at(seq_id);
                        std::cout << " - found " << kv_blocks.size() << " layers in block table\n";
                    }

                    const auto& kv_blocks = scheduler_output.m_block_tables.at(seq_id);

                    if (is_fill_all) {
                        if (cb_verbose()) std::cout << "    Taking is_fill_all path\n";
                        size_t num_blocks = sequence_group->get_num_logical_blocks();
                        if (cb_verbose()) std::cout << "    num_blocks=" << num_blocks << ", filling for layer " << layer_idx << "\n";
                        for (size_t block_id = 0; block_id < num_blocks; ++block_id) {
                            // In case no cache eviction is requested, all per-layer block tables are expected to be
                            // identical at all times
                            int32_t physical_block_idx = kv_blocks[layer_idx][block_id]->get_index();
                            if (cb_verbose()) std::cout << "      block_id=" << block_id << " -> physical_block_idx=" << physical_block_idx << "\n";
                            block_indices_data[block_id] = physical_block_idx;
                        }
                        block_indices_data += num_blocks;
                        filled_blocks_per_layer[layer_idx] += num_blocks;
                    } else {
                        if (cb_verbose()) std::cout << "    Taking selective fill path (NOT is_fill_all)\n";
                        auto seq_id_to_select_logical_idx_map = seq_id_to_select_logical_idx_maps[layer_idx];
                        if (cb_verbose()) std::cout << "    seq_id_to_select_logical_idx_map.size()=" << seq_id_to_select_logical_idx_map.size() << "\n";
                        if (seq_id_to_select_logical_idx_map.find(seq_id) == seq_id_to_select_logical_idx_map.end()) {
                            if (cb_verbose()) std::cout << "    seq_id " << seq_id << " NOT found in map for layer " << layer_idx << ", skipping\n";
                            continue;  // sequence not being present in layer-specific map means it should be skipped entirely
                        }

                        if (cb_verbose()) std::cout << "    seq_id " << seq_id << " found in map\n";
                        const auto& select_logical_idxs = seq_id_to_select_logical_idx_maps[layer_idx].at(seq_id);
                        if (cb_verbose()) std::cout << "    select_logical_idxs.size()=" << select_logical_idxs.size() << "\n";
                        const auto& block_table = kv_blocks[layer_idx];
                        size_t block_table_size = block_table.size();
                        if (cb_verbose()) std::cout << "    block_table_size=" << block_table_size << "\n";
                        for (size_t block_id = 0; block_id < select_logical_idxs.size(); ++block_id) {
                            size_t logical_block_idx = select_logical_idxs[block_id];
                            OPENVINO_ASSERT(logical_block_idx < block_table_size);
                            int32_t physical_block_idx = block_table[logical_block_idx]->get_index();
                            if (cb_verbose()) std::cout << "      block_id=" << block_id << ", logical_block_idx=" << logical_block_idx 
                                      << " -> physical_block_idx=" << physical_block_idx << "\n";
                            block_indices_data[block_id] = physical_block_idx;
                        }
                        block_indices_data += select_logical_idxs.size();
                        filled_blocks_per_layer[layer_idx] += select_logical_idxs.size();
                    }
                }
            }
        }
        if (cb_verbose()) {
            std::cout << "\n--- Verification after filling ---\n";
            for (size_t layer_idx = 0; layer_idx < dst_tensor_names.size(); layer_idx++) {
                const auto& target_tensor_name = dst_tensor_names[layer_idx];
                size_t tensor_size = m_request.get_tensor(target_tensor_name).get_size();
                size_t last_filled_element_idx = filled_blocks_per_layer[layer_idx];
                std::cout << "  " << target_tensor_name << " tensor_size=" << tensor_size 
                          << ", filled=" << last_filled_element_idx << "\n";
                auto tensor = m_request.get_tensor(target_tensor_name);
                auto* data = tensor.data<int32_t>();
                std::cout << "  " << target_tensor_name << " actual data: ";
                for (size_t i = 0; i < std::min<size_t>(10, tensor_size); ++i) {
                    std::cout << data[i] << " ";
                }
                std::cout << "\n";
            }
        }
        for (size_t layer_idx = 0; layer_idx < dst_tensor_names.size(); layer_idx++) {
            const auto& target_tensor_name = dst_tensor_names[layer_idx];
            size_t tensor_size = m_request.get_tensor(target_tensor_name).get_size();
            size_t last_filled_element_idx = filled_blocks_per_layer[layer_idx];
            OPENVINO_ASSERT(tensor_size == last_filled_element_idx, "did not fill tensor ", target_tensor_name, " completely, tensor size in elements ", tensor_size, ", last filled idx ", last_filled_element_idx);
        }
    }

    // Fills indices for sequences in the order defined by seq_id_to_select_logical_idx_maps
    // (i.e. ascending for an ordered map)
    void _fill_select_indices_from_block_tables(
        const std::vector<std::string>& dst_tensor_names,
        const Scheduler::Output& scheduler_output,
        const std::vector<std::map<size_t, std::vector<size_t>>>& seq_id_to_select_logical_idx_maps) {
        OPENVINO_ASSERT(seq_id_to_select_logical_idx_maps.size() == dst_tensor_names.size() ||
                        (dst_tensor_names.size() == 1 && !m_is_use_per_layer_cache_control) ||
                        seq_id_to_select_logical_idx_maps.empty());
        std::vector<size_t> filled_blocks_per_layer(dst_tensor_names.size(), 0);

        for (size_t layer_idx = 0; layer_idx < dst_tensor_names.size(); layer_idx++) {
            auto input_tensor = m_request.get_tensor(dst_tensor_names[layer_idx]);
            auto block_indices_data = input_tensor.data<int32_t>();
            for (const auto& kv : seq_id_to_select_logical_idx_maps[layer_idx]) {
                size_t seq_id = kv.first;
                const auto& select_logical_idxs = kv.second;

                const auto& kv_blocks = scheduler_output.m_block_tables.at(seq_id);
                const auto& block_table = kv_blocks[layer_idx];
                size_t block_table_size = block_table.size();
                for (size_t block_id = 0; block_id < select_logical_idxs.size(); ++block_id) {
                    size_t logical_block_idx = select_logical_idxs[block_id];
                    OPENVINO_ASSERT(logical_block_idx < block_table_size);
                    block_indices_data[block_id] = block_table[logical_block_idx]->get_index();
                }
                block_indices_data += select_logical_idxs.size();
                filled_blocks_per_layer[layer_idx] += select_logical_idxs.size();
            }
        }
        for (size_t layer_idx = 0; layer_idx < dst_tensor_names.size(); layer_idx++) {
            const auto& target_tensor_name = dst_tensor_names[layer_idx];
            size_t tensor_size = m_request.get_tensor(target_tensor_name).get_size();
            size_t last_filled_element_idx = filled_blocks_per_layer[layer_idx];
            OPENVINO_ASSERT(tensor_size == last_filled_element_idx, "did not fill tensor ", target_tensor_name, " completely, tensor size in elements ", tensor_size, ", last filled idx ", last_filled_element_idx);
        }
    }

    void _set_block_indices(const std::vector<SequenceGroup::Ptr>& sequence_groups,
                            const Scheduler::Output& scheduler_output,
                            size_t total_num_blocks,
                            const std::map<size_t, std::set<size_t>>& seq_id_to_skipped_blocks_map) {
        if (cb_verbose()) {
            std::cout << "\n--- _set_block_indices called ---\n";
            std::cout << "Total num blocks requested: " << total_num_blocks << "\n";
        }
        std::vector<std::string> tensor_names = {"block_indices"};

        size_t num_layers = 1;
        if (m_is_use_per_layer_cache_control) {
            num_layers = m_num_decoder_layers;
            tensor_names.resize(m_num_decoder_layers);
            for (size_t i = 0; i < tensor_names.size(); i++) {
                tensor_names[i] = std::string("block_indices.") + std::to_string(i);
            }
        }


        std::vector<size_t> num_blocks_per_layer(num_layers);

        // Create map with correct size based on num_layers, not m_num_decoder_layers
        std::vector<std::map<size_t, std::vector<size_t>>> seq_id_to_select_logical_idx_map(num_layers);
        size_t num_sequence_groups = scheduler_output.m_scheduled_sequence_groups_ids.size();
        for (size_t layer_idx = 0; layer_idx < num_layers; layer_idx++) {
            for (size_t i = 0; i < num_sequence_groups; ++i) {
                size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
                SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id];
                std::vector<Sequence::CPtr> running_sequences = sequence_group->get_running_sequences();
                size_t num_running_sequences = running_sequences.size();
                for (size_t k = 0; k < num_running_sequences; ++k) {
                    Sequence::CPtr sequence = running_sequences[k];
                    size_t num_blocks = sequence_group->get_num_logical_blocks();
                    size_t seq_id = sequence->get_id();
                    if (cb_verbose()) {
                        std::cout << "  Processing sequence: seq_id=" << seq_id << ", num_blocks=" << num_blocks << "\n";
                        std::cout << "    layer_idx=" << layer_idx << ", num_blocks_per_layer[" << layer_idx << "] BEFORE=" << num_blocks_per_layer[layer_idx] << "\n";
                    }
                    std::vector<size_t> remaining_logical_block_ids;
                    if (seq_id_to_skipped_blocks_map.find(seq_id) != seq_id_to_skipped_blocks_map.end()) {
                        if (cb_verbose()) std::cout << "    Taking SKIPPED BLOCKS path\n";
                        const auto& skip_set = seq_id_to_skipped_blocks_map.at(seq_id);
                        OPENVINO_ASSERT(num_blocks >= skip_set.size());
                        remaining_logical_block_ids.reserve(num_blocks - skip_set.size());
                        for (size_t j = 0; j < num_blocks; j++) {
                            if (skip_set.find(j) == skip_set.end()) {
                                remaining_logical_block_ids.push_back(j);
                            }
                        }
                        seq_id_to_select_logical_idx_map[layer_idx][seq_id] = remaining_logical_block_ids;
                        num_blocks_per_layer[layer_idx] += remaining_logical_block_ids.size();
                        if (cb_verbose()) std::cout << "    Added " << remaining_logical_block_ids.size() << " blocks\n";
                    }
                    else
                    {
                        if (cb_verbose()) std::cout << "    Taking NORMAL (no skipped blocks) path\n";
                        auto& vec = seq_id_to_select_logical_idx_map[layer_idx][seq_id];
                        vec.resize(num_blocks);
                        std::iota(vec.begin(), vec.end(), 0);
                        num_blocks_per_layer[layer_idx] += num_blocks;
                        if (cb_verbose()) std::cout << "    Added " << num_blocks << " blocks\n";
                    }
                    if (cb_verbose()) std::cout << "    num_blocks_per_layer[" << layer_idx << "] AFTER=" << num_blocks_per_layer[layer_idx] << "\n";
                }

            }
        }

        if (cb_verbose()) {
            std::cout << "\n--- Setting block_indices tensor shapes ---\n";
            for (size_t i = 0; i < num_layers; i++) {
                std::cout << "  Setting " << tensor_names[i] << " shape to [" << num_blocks_per_layer[i] << "]\n";
            }
        }
        for (size_t i = 0; i < num_layers; i++) {
            m_request.get_tensor(tensor_names[i]).set_shape({num_blocks_per_layer[i]});
            if (cb_verbose()) {
                auto actual_shape = m_request.get_tensor(tensor_names[i]).get_shape();
                std::cout << "  " << tensor_names[i] << " actual shape after set_shape: [" << actual_shape[0] << "]\n";
            }
        }

        _fill_indices_from_block_tables(tensor_names, sequence_groups, scheduler_output, seq_id_to_select_logical_idx_map);
    }

    void _set_cache_rotation_coefficients(const std::vector<SequenceGroup::Ptr>& sequence_groups,
                                          const Scheduler::Output& scheduler_output) {
        std::vector<std::string> rotation_indices_tensor_names(m_num_decoder_layers);
        for (size_t i = 0; i < m_num_decoder_layers; i++) {
            auto tensor_name = std::string("rotated_block_indices.") + std::to_string(i);
            rotation_indices_tensor_names[i] = tensor_name;
            size_t num_indices = 0;
            for (const auto& entry : m_rotated_block_logical_indices_per_sequence_for_each_layer[i]) {
                num_indices += entry.second.size();
            }
            auto rotated_block_indices_tensor = m_request.get_tensor(tensor_name);
            rotated_block_indices_tensor.set_shape({num_indices});
        }

        for (size_t i = 0; i < m_num_decoder_layers; i++) {
            auto tensor_name = std::string("rotation_deltas.") + std::to_string(i);
            m_request.set_tensor(tensor_name, m_cache_rotation_deltas_for_each_layer[i]);
        }


        // NB: the order of per-sequence index filling in the function below must be the same
        // as the order of `seq_id`s in which the "rotation_coefficients.N" inputs are filled
        // (i.e. ascending by seq_id values)
        _fill_select_indices_from_block_tables(rotation_indices_tensor_names,
                                               scheduler_output,
                                               m_rotated_block_logical_indices_per_sequence_for_each_layer);
    }

    void _reset_cache_rotation_coefficients() {
        m_cache_rotation_deltas_for_each_layer.clear();
        for (size_t i = 0; i < m_num_decoder_layers; i++) {
            m_cache_rotation_deltas_for_each_layer.push_back(ov::Tensor());
        }
    }

    void _collect_attention_scores(const std::vector<SequenceGroup::Ptr> & sequence_groups, const Scheduler::Output& scheduler_output) {
        m_last_attention_scores.clear();
        size_t num_sequence_groups = scheduler_output.m_scheduled_sequence_groups_ids.size();
        using IndexSpan = std::pair<size_t, size_t>;
        std::list<std::pair<size_t, IndexSpan>> running_sequence_group_ids_and_kvcache_spans;
        size_t offset = 0;
        for (size_t i = 0; i < num_sequence_groups; ++i) { size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i]; SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id]; std::vector<Sequence::CPtr> running_sequences = sequence_group->get_running_sequences();

            for (size_t seq_idx = 0; seq_idx < running_sequences.size(); ++seq_idx) {
                Sequence::CPtr sequence = running_sequences[seq_idx];
                size_t global_sequence_id = sequence->get_id();
                size_t subsequence_length = sequence_group->get_context_len() - sequence_group->get_num_evicted_tokens();
                if (scheduler_output.m_apply_sparse_attention_mask) {
                    size_t num_past_blocks_to_discard = 0;
                    const auto& skip_map = scheduler_output.m_sparse_attention_skipped_logical_blocks;
                    auto it = skip_map.find(global_sequence_id);
                    if (it != skip_map.end()) {
                        num_past_blocks_to_discard = it->second.size();
                    }
                    subsequence_length -= num_past_blocks_to_discard * m_block_size;
                }

                IndexSpan span = {offset, offset + subsequence_length};
                offset += subsequence_length;


                bool is_prefill_finished = sequence_group->can_generate_tokens();
                bool has_snapkv_scores = (scheduler_output.m_score_aggregation_windows.find(global_sequence_id) != scheduler_output.m_score_aggregation_windows.end());
                if (is_prefill_finished || (!is_prefill_finished && has_snapkv_scores)) {
                    // During prompt phase, will only collect the scores for sequences that have been processed up to their SnapKV window size
                    // (this may happen across multiple scheduling iterations - assuming here that the code using the collected scores does simple aggregation
                    // such as addition and therefore does not need to know which part of the SnapKV window a given score vector belongs to).
                    //
                    // During generation phase, the scores may be either SnapKV-aggregated (if the phase included the very last part of the prompt) or
                    // not (regular non-aggregated single-token-position scores for the newly generated token), but this should also not matter to the simple aggregation
                    // code.
                    running_sequence_group_ids_and_kvcache_spans.emplace_back(global_sequence_id, span);
                }
            }
        }

        for (const auto& seq_id_and_score_span : running_sequence_group_ids_and_kvcache_spans) {
            auto attention_scores_across_decoder_layers_for_current_sequence = AttentionScoresForEachDecoderLayer(m_num_decoder_layers);
            size_t global_sequence_id = seq_id_and_score_span.first;
            IndexSpan span = seq_id_and_score_span.second;
            for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_decoder_layers; decoder_layer_id++) {
                auto attention_score = m_request.get_tensor(get_paged_attention_score_output_for_decoder_layer(decoder_layer_id));
                auto scores_for_cache_of_current_sequence_group = ov::Tensor(attention_score, ov::Coordinate{span.first}, ov::Coordinate{span.second});
                auto copied_tensor = ov::Tensor(scores_for_cache_of_current_sequence_group.get_element_type(), ov::Shape{span.second - span.first});
                scores_for_cache_of_current_sequence_group.copy_to(copied_tensor);
                attention_scores_across_decoder_layers_for_current_sequence[decoder_layer_id] = scores_for_cache_of_current_sequence_group;
            }
            m_last_attention_scores[global_sequence_id] = attention_scores_across_decoder_layers_for_current_sequence;
        }
    }

    void _collect_adaptive_rkv_block_diversities(const std::vector<SequenceGroup::Ptr> & sequence_groups, const Scheduler::Output& scheduler_output) {
        m_last_block_diversities.clear();
        size_t num_sequence_groups = scheduler_output.m_scheduled_sequence_groups_ids.size();
        using IndexSpan = std::pair<size_t, size_t>;
        std::list<std::pair<size_t, IndexSpan>> running_seq_ids_and_kvcache_spans;
        size_t offset = 0;
        for (size_t i = 0; i < num_sequence_groups; ++i) {
            size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
            SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id];

            if (!sequence_group->can_generate_tokens()) {
                // As we only evict during generation phase, so will the similarity calculation will only be
                // scheduled after prefill is finished
                continue;
            }
            std::vector<Sequence::CPtr> running_sequences = sequence_group->get_running_sequences();

            for (size_t seq_idx = 0; seq_idx < running_sequences.size(); ++seq_idx) {
                Sequence::CPtr sequence = running_sequences[seq_idx];
                size_t global_sequence_id = sequence->get_id();
                auto it = scheduler_output.m_adaptive_rkv_evictable_sizes.find(global_sequence_id);
                if (it == scheduler_output.m_adaptive_rkv_evictable_sizes.end()) {
                    // Adaptive R-KV diversity calculation was not scheduled for this sequence
                    continue;
                }

                // [eviction_size_in_tokens / block_size, eviction_size_in_tokens]
                size_t num_diversity_values_calculated = it->second * it->second / m_block_size;
                IndexSpan span = {offset, offset + num_diversity_values_calculated};
                offset += num_diversity_values_calculated;
                running_seq_ids_and_kvcache_spans.emplace_back(global_sequence_id, span);
            }
        }

        for (const auto& seq_id_and_score_span : running_seq_ids_and_kvcache_spans) {
            auto block_diversities_across_decoder_layers_for_current_sequence = BlockDiversityForEachDecoderLayer(m_num_decoder_layers);
            size_t global_sequence_id = seq_id_and_score_span.first;
            IndexSpan span = seq_id_and_score_span.second;
            for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_decoder_layers; decoder_layer_id++) {
                auto output_tensor_name = get_adaptive_rkv_diversity_score_output_for_decoder_layer(decoder_layer_id);
                auto diversities_in_this_layer = m_request.get_tensor(output_tensor_name);
                OPENVINO_ASSERT(diversities_in_this_layer.get_size() != 0, "Size of the output ", output_tensor_name, " may not be zero");
                OPENVINO_ASSERT(diversities_in_this_layer.get_size() >= span.second, "Size of the output ", output_tensor_name, " must be at least ", span.second);

                auto diversities_of_current_sequence_group = ov::Tensor(diversities_in_this_layer, ov::Coordinate{span.first}, ov::Coordinate{span.second});
                auto copied_tensor = ov::Tensor(diversities_of_current_sequence_group.get_element_type(), ov::Shape{span.second - span.first});
                diversities_of_current_sequence_group.copy_to(copied_tensor);
                block_diversities_across_decoder_layers_for_current_sequence[decoder_layer_id] = diversities_of_current_sequence_group;
            }
            m_last_block_diversities[global_sequence_id] = block_diversities_across_decoder_layers_for_current_sequence;
        }
    }

    void _set_xattention_tensors(const std::vector<SequenceGroup::Ptr>& sequence_groups,
                                 const Scheduler::Output& scheduler_output,
                                 size_t batch_size_in_sequences) {
        ov::Tensor xattention_block_size(ov::element::i32, {});
        ov::Tensor xattention_stride(ov::element::i32, {});
        xattention_block_size.data<int32_t>()[0] = scheduler_output.m_xattention_block_size;
        xattention_stride.data<int32_t>()[0] = scheduler_output.m_xattention_stride;
        m_request.set_tensor("xattention_block_size", xattention_block_size);
        m_request.set_tensor("xattention_stride", xattention_stride);

        ov::Tensor xattention_thresholds(ov::element::f32, {batch_size_in_sequences});
        float* xattention_threshold_data = xattention_thresholds.data<float>();
        for (size_t i = 0; i < scheduler_output.m_scheduled_sequence_groups_ids.size(); i++) {
            size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
            SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id];
            std::vector<Sequence::CPtr> running_sequences = sequence_group->get_running_sequences();
            size_t num_running_sequences = running_sequences.size();
            for (size_t k = 0; k < num_running_sequences; ++k) {
                Sequence::CPtr sequence = running_sequences[k];
                size_t seq_id = sequence->get_id();
                float threshold = 0.0;

                if (scheduler_output.m_xattention_thresholds.find(seq_id) != scheduler_output.m_xattention_thresholds.end()) {
                    threshold = scheduler_output.m_xattention_thresholds.at(seq_id);
                }
                *xattention_threshold_data = threshold;
                xattention_threshold_data += 1;
            }
        }

        for (size_t i = 0; i < m_num_decoder_layers; i++) {
            auto tensor_name = std::string("xattention_threshold.") + std::to_string(i);
            m_request.set_tensor(tensor_name, xattention_thresholds);
        }

    }

    void _set_adaptive_rkv_tensors(const std::vector<SequenceGroup::Ptr>& sequence_groups,
                                   const Scheduler::Output& scheduler_output,
                                   size_t batch_size_in_sequences) {
        ov::Tensor adaptive_rkv_start_size(ov::element::i32, {});
        adaptive_rkv_start_size.data<int32_t>()[0] = scheduler_output.m_adaptive_rkv_start_size;
        m_request.set_tensor("adaptive_rkv_start_size", adaptive_rkv_start_size);

        ov::Tensor adaptive_rkv_evictable_sizes(ov::element::i32, {batch_size_in_sequences});
        int32_t* adaptive_rkv_evictable_sizes_data = adaptive_rkv_evictable_sizes.data<int32_t>();
        std::vector<size_t> running_seq_ids;
        for (size_t i = 0; i < scheduler_output.m_scheduled_sequence_groups_ids.size(); i++) {
            size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
            SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id];
            std::vector<Sequence::CPtr> running_sequences = sequence_group->get_running_sequences();
            size_t num_running_sequences = running_sequences.size();
            for (size_t k = 0; k < num_running_sequences; ++k) {
                Sequence::CPtr sequence = running_sequences[k];
                size_t seq_id = sequence->get_id();
                running_seq_ids.push_back(seq_id);
            }
        }

        for (size_t seq_id : running_seq_ids) {
            size_t evictable_size = 0;

            if (scheduler_output.m_adaptive_rkv_evictable_sizes.find(seq_id) != scheduler_output.m_adaptive_rkv_evictable_sizes.end()) {
                evictable_size = scheduler_output.m_adaptive_rkv_evictable_sizes.at(seq_id);
            }
            *adaptive_rkv_evictable_sizes_data = evictable_size;
            adaptive_rkv_evictable_sizes_data += 1;
        }

        m_request.set_tensor("adaptive_rkv_evictable_sizes", adaptive_rkv_evictable_sizes);

        std::vector<size_t> num_diversity_set_blocks_per_layer(m_num_decoder_layers, 0);
        if (scheduler_output.m_adaptive_rkv_diversity_block_sets_for_each_layer_per_sequence.empty()) {
            // Set the auxiliary tensors to zero-shape if the scheduler did not provide information on which blocks of the
            // evictable area belong to the diversity subset
            for (size_t i = 0; i < m_num_decoder_layers; i++) {
                auto begins_name = std::string("adaptive_rkv_diversity_block_set_indices_begins.") + std::to_string(i);
                auto indices_name = std::string("adaptive_rkv_diversity_block_set_indices.") + std::to_string(i);
                ov::Tensor adaptive_rkv_diversity_block_set_begins(ov::element::i32, {0});
                ov::Tensor adaptive_rkv_diversity_block_set_indices(ov::element::i32, {0});
                m_request.set_tensor(begins_name, adaptive_rkv_diversity_block_set_begins);
                m_request.set_tensor(indices_name, adaptive_rkv_diversity_block_set_indices);
            }
        }
        else {
            // This will provide opportunity for optimization of the in-kernel diversity calculation by only computing the
            // diversity scores between the actual blocks of the "diversity" set and not among all of the evictable blocks,
            // which also include the blocks that will be necessarily kept as part of attention mass preservation.
            for (size_t i = 0; i < m_num_decoder_layers; i++) {
                ov::Tensor adaptive_rkv_diversity_block_set_begins(ov::element::i32, {batch_size_in_sequences + 1});
                OPENVINO_ASSERT(batch_size_in_sequences == running_seq_ids.size());
                auto begins_data = adaptive_rkv_diversity_block_set_begins.data<int32_t>();
                begins_data[0] = 0;
                begins_data += 1;

                auto begins_name = std::string("adaptive_rkv_diversity_block_set_indices_begins.") + std::to_string(i);
                const auto& adaptive_rkv_diversity_block_map = scheduler_output.m_adaptive_rkv_diversity_block_sets_for_each_layer_per_sequence[i];
                for (size_t seq_id : running_seq_ids) {
                    OPENVINO_ASSERT(adaptive_rkv_diversity_block_map.find(seq_id) != adaptive_rkv_diversity_block_map.end());
                    size_t num_blocks = adaptive_rkv_diversity_block_map.at(seq_id).size();
                    num_diversity_set_blocks_per_layer[i] += num_blocks;
                    *begins_data  = num_diversity_set_blocks_per_layer[i];
                    begins_data += 1;
                }
                m_request.set_tensor(begins_name, adaptive_rkv_diversity_block_set_begins);
            }

            std::vector<std::string> indices_tensor_names(m_num_decoder_layers);
            for (size_t i = 0; i < m_num_decoder_layers; i++) {
                auto indices_name = std::string("adaptive_rkv_diversity_block_set_indices.") + std::to_string(i);
                indices_tensor_names[i] = indices_name;
                auto indices_tensor = m_request.get_tensor(indices_name);
                indices_tensor.set_shape({num_diversity_set_blocks_per_layer[i]});
            }
            _fill_select_indices_from_block_tables(indices_tensor_names,
                                                   scheduler_output,
                                                   scheduler_output.m_adaptive_rkv_diversity_block_sets_for_each_layer_per_sequence);
        }
    }
};
}
