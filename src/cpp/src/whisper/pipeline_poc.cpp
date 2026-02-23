// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/whisper_pipeline.hpp"
#include "utils.hpp"
#include "whisper/config.hpp"
#include "whisper/feature_extractor.hpp"
#include "whisper/context_tokens.hpp"
#include "whisper/models/decoder.hpp"
#include "whisper/logit_processor.hpp"

// Standard library
#include <limits>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <future>
#include <atomic>

// OpenVINO pass and pattern matching headers
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/matcher.hpp"

// OpenVINO operations
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"

namespace ov {
namespace genai {

// TODO: extract from model metadata when available
static constexpr int NUM_LAYERS = 12;

// ----------- Model transformations -------------
namespace {

// Check if model already has attention_mask input
bool has_attention_mask_input(const std::shared_ptr<ov::Model>& model) {
    for (const auto& input : model->inputs()) {
        if (input.get_any_name() == "attention_mask") {
            return true;
        }
    }
    return false;
}

// Add attention_mask input to decoder model for continuous batching
// This function modifies the model graph to accept dynamic attention masks
// for handling sequences of different lengths in batch processing
void add_attention_mask_input(std::shared_ptr<ov::Model> model, bool for_cross_attention = false) {
    using namespace ov::op;

    if (has_attention_mask_input(model)) {
        std::cout << "Model already has attention_mask input, skipping transformation\n";
        return;
    }

    std::vector<std::shared_ptr<ov::Node>> self_attn_nodes;
    std::vector<std::shared_ptr<ov::Node>> cross_attn_nodes;
    const auto kAttnMaskPort = 3;

    // Find all ScaledDotProductAttention nodes and classify them
    for (const auto& node : model->get_ops()) {
        if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(node)) {
            // Use node name to distinguish: cross-attention has "encoder_attn" in name
            std::string node_name = node->get_friendly_name();
            if (node_name.find("encoder_attn") != std::string::npos || 
                node_name.find("cross_attn") != std::string::npos) {
                // Cross-attention node - don't modify
                cross_attn_nodes.push_back(node);
            } else {
                // Self-attention node - needs our mask
                self_attn_nodes.push_back(node);
            }
        }
    }

    std::cout << "Found " << self_attn_nodes.size() << " self-attention SDPA nodes\n";
    std::cout << "Found " << cross_attn_nodes.size() << " cross-attention SDPA nodes\n";

    if (self_attn_nodes.empty() && cross_attn_nodes.empty()) {
        std::cout << "No SDPA nodes found in model\n";
        return;
    }

    // Create attention_mask parameter with dynamic shape
    auto attention_mask = std::make_shared<v0::Parameter>(
        ov::element::f32, 
        ov::PartialShape{-1, -1}  // [batch_size, seq_len]
    );
    attention_mask->get_output_tensor(0).set_names({"attention_mask"});
    model->add_parameters({attention_mask});

    std::cout << "Added attention_mask parameter to model\n";

    // Create constants for mask transformation
    auto cst_minus_inf = std::make_shared<v0::Constant>(
        ov::element::f32, ov::Shape{1}, 
        std::vector<float>{-std::numeric_limits<float>::infinity()}
    );
    auto cst_zero = std::make_shared<v0::Constant>(
        ov::element::f32, ov::Shape{1}, std::vector<float>{0.0f}
    );
    auto cst_one = std::make_shared<v0::Constant>(
        ov::element::f32, ov::Shape{1}, std::vector<float>{1.0f}
    );
    auto cst_axis_1 = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{1});
    auto cst_axis_2 = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{2});

    // Convert attention mask: 1 -> 0.0 (attend), 0 -> -inf (mask)
    auto equal = std::make_shared<v1::Equal>(attention_mask->output(0), cst_one->output(0));
    auto select = std::make_shared<v1::Select>(
        equal->output(0), 
        cst_zero->output(0), 
        cst_minus_inf->output(0)
    );

    // Reshape from [batch_size, seq_len] to [batch_size, 1, 1, seq_len] for SDPA
    // SDPA expects [batch_size, num_heads, seq_len_q, seq_len_k]
    // We provide [batch_size, 1, 1, seq_len_k] which broadcasts across heads and query positions
    auto unsqueeze_1 = std::make_shared<v0::Unsqueeze>(select->output(0), cst_axis_1->output(0));
    auto unsqueeze_2 = std::make_shared<v0::Unsqueeze>(unsqueeze_1->output(0), cst_axis_2->output(0));

    // Apply transformed mask ONLY to self-attention nodes
    for (const auto& sdpa_node : self_attn_nodes) {
        if (sdpa_node->inputs().size() > kAttnMaskPort) {
            // Has existing mask - combine with our mask using Add
            // This preserves causal masking while adding padding mask
            auto existing_mask = sdpa_node->input(kAttnMaskPort).get_source_output();
            auto combined_mask = std::make_shared<v1::Add>(existing_mask, unsqueeze_2->output(0));
            sdpa_node->input(kAttnMaskPort).replace_source_output(combined_mask->output(0));
        } else {
            // No mask input - create new SDPA with mask
            auto new_sdpa = std::make_shared<v13::ScaledDotProductAttention>(
                sdpa_node->input(0).get_source_output(),
                sdpa_node->input(1).get_source_output(),
                sdpa_node->input(2).get_source_output(),
                unsqueeze_2->output(0),
                false  // causal
            );
            ov::replace_node(sdpa_node, new_sdpa);
        }
    }

    std::cout << "Connected attention_mask to " << self_attn_nodes.size() << " self-attention SDPA nodes\n";
    std::cout << "Left " << cross_attn_nodes.size() << " cross-attention SDPA nodes unchanged\n";
    
    model->validate_nodes_and_infer_types();
    std::cout << "Model transformation completed successfully\n";
}

}  // anonymous namespace
// ----------- End model transformations -------------

// ----------- Cache management -------------

class Request {
    size_t request_id = 0;  // Async request ID (for mapping back to promise)
    ov::Tensor encoder_hidden_state;
    ov::Tensor initial_prompt_ids;  // Full initial prompt for first inference
    int64_t last_token;
    size_t sequence_length;
    bool initial_step = true;  // Track if this is the first decoding step for this request
    
    // KV cache storage: per layer, storing both decoder and encoder caches
    // Shape: decoder - [1, num_heads, seq_len, head_size], encoder - [1, num_heads, encoder_seq_len, head_size]
    std::vector<ov::Tensor> decoder_key_cache;   // One per layer
    std::vector<ov::Tensor> decoder_value_cache; // One per layer
    std::vector<ov::Tensor> encoder_key_cache;   // One per layer (stays constant after first step)
    std::vector<ov::Tensor> encoder_value_cache; // One per layer (stays constant after first step)

public:
    Request(ov::Tensor encoder_hidden_state, ov::Tensor input_ids)
        : encoder_hidden_state(encoder_hidden_state) {
            sequence_length = input_ids.get_shape()[1];  // Initial prompt length
            
            // Store full initial prompt (needed for first inference)
            initial_prompt_ids = ov::Tensor(input_ids.get_element_type(), input_ids.get_shape());
            input_ids.copy_to(initial_prompt_ids);
            
            // Store the last token from input_ids
            const int64_t* input_data = input_ids.data<int64_t>();
            last_token = input_data[sequence_length - 1];
        }
    
    int64_t get_last_token() const { return last_token; }
    void set_last_token(int64_t token) { last_token = token; }
    size_t get_sequence_length() const { return sequence_length; }
    void increment_sequence_length() { sequence_length++; }
    bool is_initial_step() const { return initial_step; }
    void mark_initial_step_done() { initial_step = false; }
    
    size_t get_request_id() const { return request_id; }
    void set_request_id(size_t id) { request_id = id; }
    
    const ov::Tensor& get_encoder_hidden_state() const { return encoder_hidden_state; }
    const ov::Tensor& get_initial_prompt_ids() const { return initial_prompt_ids; }
    
    // KV cache getters/setters (non-const for modification)
    std::vector<ov::Tensor>& get_decoder_key_cache() { return decoder_key_cache; }
    std::vector<ov::Tensor>& get_decoder_value_cache() { return decoder_value_cache; }
    std::vector<ov::Tensor>& get_encoder_key_cache() { return encoder_key_cache; }
    std::vector<ov::Tensor>& get_encoder_value_cache() { return encoder_value_cache; }
    
    // KV cache getters (const for reading)
    const std::vector<ov::Tensor>& get_decoder_key_cache() const { return decoder_key_cache; }
    const std::vector<ov::Tensor>& get_decoder_value_cache() const { return decoder_value_cache; }
    const std::vector<ov::Tensor>& get_encoder_key_cache() const { return encoder_key_cache; }
    const std::vector<ov::Tensor>& get_encoder_value_cache() const { return encoder_value_cache; }
};

class BatchManager {
    size_t current_batch_size = 0;  // Current batch size being processed
    size_t max_batch_size = 4;  // Maximum batch size
    size_t num_layers = NUM_LAYERS;  // Number of transformer layers
    std::deque<size_t> free_batch_slots;  // Deque of available batch slots (freed slots go to front for reuse)
    std::unordered_map<size_t, Request> requests_map;  // Map of batch index to Request for active requests
    
    // References to the two decoder models
    ov::InferRequest& m_request_decoder;            // For first inference (no past)
    ov::InferRequest& m_request_decoder_with_past;  // For subsequent inferences (with past)
    
    // Compiled models for recreating InferRequests
    ov::CompiledModel m_compiled_decoder;
    ov::CompiledModel m_compiled_decoder_with_past;
    
    // Track previous batch composition to detect when reconstruction is needed
    std::vector<size_t> prev_phase0_batch_indices;  // Last batch run on decoder
    std::vector<size_t> prev_phase1_batch_indices;  // Last batch run on decoder_with_past
    
    // Flag to reset decoders before next step (set when batch becomes empty)
    bool need_decoder_reset = false;
    
    // Thread safety for batch processing (mutable so they can be used in const methods)
    mutable std::mutex batch_mutex;  // Protects requests_map and batch state
    mutable std::condition_variable batch_cv;
    
    // Pending requests queue (for thread-safe async submission)
    struct PendingRequest {
        size_t request_id;
        ov::Tensor encoder_hidden_state;
        ov::Tensor input_ids;
    };
    std::queue<PendingRequest> awaiting_requests;
    mutable std::mutex awaiting_mutex;  // Protects awaiting_requests queue ONLY
    mutable std::condition_variable awaiting_cv;
    
    // Request state tracking for async operation
    struct RequestState {
        bool completed = false;
        std::vector<int64_t> generated_tokens;
        std::shared_ptr<std::promise<std::vector<int64_t>>> result_promise;
    };
    std::unordered_map<size_t, RequestState> request_states;
    mutable std::mutex request_states_mutex;  // Protects request_states map
    
    // Control
    std::atomic<bool> should_stop{false};
    std::atomic<size_t> next_request_id{0};  // Atomic for thread-safe ID generation
    int64_t eos_token_id = 50257;  // Whisper EOS token
    size_t max_decode_steps = 50;  // Maximum generation steps

public:
    BatchManager(size_t max_batch_size, 
                 ov::InferRequest& request_decoder,
                 ov::InferRequest& request_decoder_with_past,
                 ov::CompiledModel compiled_decoder,
                 ov::CompiledModel compiled_decoder_with_past) 
        : max_batch_size(max_batch_size),
          m_request_decoder(request_decoder),
          m_request_decoder_with_past(request_decoder_with_past),
          m_compiled_decoder(compiled_decoder),
          m_compiled_decoder_with_past(compiled_decoder_with_past) {
        for (size_t i = 0; i < max_batch_size; ++i) {
            free_batch_slots.push_back(i);
        }
    }
    
    size_t reserve_slot_for_request(ov::Tensor& encoder_hidden_state, ov::Tensor& input_ids) {
        current_batch_size++;
        if (current_batch_size > max_batch_size) {
            throw std::runtime_error("Exceeded maximum batch size");
        }
        size_t batch_index = free_batch_slots.front();
        free_batch_slots.pop_front();
        requests_map.emplace(batch_index, Request{encoder_hidden_state, input_ids});
        return batch_index;
    }

    void release_slot(size_t batch_index) {
        if (requests_map.count(batch_index)) {
            requests_map.erase(batch_index);
        }
        free_batch_slots.push_front(batch_index);  // Add to front for stack-like behavior
        current_batch_size--;
        
        // If batch is now empty, flag that decoders need reset before next use
        if (requests_map.empty()) {
            need_decoder_reset = true;
        }
    }

    // Run inference for all active requests, handling mixed phases (first vs subsequent)
    // Returns map of batch_idx -> new_token
    std::map<size_t, int64_t> run_inference() {
        size_t active_batch_size = requests_map.size();
        //std::cout << "BatchManager::run_inference - active batch size: " << active_batch_size << "\n";
        if (active_batch_size == 0) {
            return {};
        }
        
        // Fast path: if all requests in phase 1 and batch stable, reuse prev tracking
        // After step 0, prev_phase0_batch_indices becomes empty, so this triggers
        if (prev_phase0_batch_indices.empty() && !prev_phase1_batch_indices.empty() && 
            active_batch_size == prev_phase1_batch_indices.size()) {
            // Verify all requests from previous batch still exist
            bool all_exist = true;
            for (size_t batch_idx : prev_phase1_batch_indices) {
                if (requests_map.find(batch_idx) == requests_map.end()) {
                    all_exist = false;
                    //std::cout << "  Fast path DISABLED: batch_idx " << batch_idx << " no longer exists\n";
                    break;
                }
            }
            
            if (all_exist) {
                // All requests in phase 1, batch size unchanged - use fast path
                //std::cout << "  Using fast path (all phase 1, stable batch)\n";
                auto new_tokens = run_inference_batch(prev_phase1_batch_indices, false);
                
                // Always extract to keep stored cache fresh
                extract_kv_cache_to_requests(m_request_decoder_with_past, prev_phase1_batch_indices, true, true);
                
                return new_tokens;
            }
        }
        
        //std::cout << "  Using normal path (separating by phase)\n";
        
        // Normal path: separate requests by phase and sort
        std::vector<size_t> initial_step_requests;  // Phase 0: first inference
        std::vector<size_t> subsequent_step_requests;  // Phase 1+: with KV cache
        
        for (const auto& [batch_idx, request] : requests_map) {
            if (request.is_initial_step()) {
                initial_step_requests.push_back(batch_idx);
                //std::cout << "  Request " << request.get_request_id() << " (batch_idx=" << batch_idx << ") -> Phase 0 (initial)\n";
            } else {
                subsequent_step_requests.push_back(batch_idx);
                //std::cout << "  Request " << request.get_request_id() << " (batch_idx=" << batch_idx << ") -> Phase 1 (subsequent)\n";
            }
        }
        
        // Sort to ensure deterministic order (critical for zero-copy optimization)
        std::sort(initial_step_requests.begin(), initial_step_requests.end());
        std::sort(subsequent_step_requests.begin(), subsequent_step_requests.end());
        
        std::map<size_t, int64_t> all_new_tokens;
        
        // Phase 0: Run initial requests on decoder model (no past KV cache)
        if (!initial_step_requests.empty()) {
            //std::cout << "  Running Phase 0 for " << initial_step_requests.size() << " requests\n";
            
            // CRITICAL FIX: Recreate decoder InferRequest to ensure clean state
            // OpenVINO InferRequests can retain internal state between infer() calls
            // This prevents contamination from previous phase 0 requests
            //std::cout << "    Recreating m_request_decoder for clean phase 0 state\n";
            m_request_decoder = m_compiled_decoder.create_infer_request();
            
            // LAZY EXTRACTION: Before running phase 0, extract KV from previous phase 1 batch
            // (if exists) since phase 0 will overwrite m_request_decoder tensors
            if (!prev_phase1_batch_indices.empty()) {
                // Filter out completed requests that no longer exist
                std::vector<size_t> still_active;
                for (size_t batch_idx : prev_phase1_batch_indices) {
                    if (requests_map.find(batch_idx) != requests_map.end()) {
                        still_active.push_back(batch_idx);
                    }
                }
                
                if (!still_active.empty()) {
                    //std::cout << "    Extracting KV from previous phase 1 batch (size=" << still_active.size() << ")\n";
                    extract_kv_cache_to_requests(m_request_decoder_with_past, still_active, true, true);
                }
            }
            
            auto new_tokens = run_inference_batch(initial_step_requests, true);
            all_new_tokens.insert(new_tokens.begin(), new_tokens.end());
            
            //std::cout << "    Extracting KV from phase 0 requests\n";
            // CRITICAL: Extract KV cache from phase 0 requests immediately
            // They will be batched with phase 1 requests in the next step
            // and need their stored cache for reconstruction
            // Phase 0 outputs both decoder and encoder KV in present.* (NOT in inputs)
            extract_kv_cache_to_requests(m_request_decoder, initial_step_requests, true, false);
        }
        
        // Phase 1+: Run subsequent requests on decoder_with_past model
        if (!subsequent_step_requests.empty()) {
            //std::cout << "  Running Phase 1 for " << subsequent_step_requests.size() << " requests\n";
            auto new_tokens = run_inference_batch(subsequent_step_requests, false);
            all_new_tokens.insert(new_tokens.begin(), new_tokens.end());
            
            // Always extract to keep stored cache fresh
            extract_kv_cache_to_requests(m_request_decoder_with_past, subsequent_step_requests, true, true);
        }
        
        // Update tracking for next iteration (must be done AFTER both phases complete)
        prev_phase0_batch_indices = initial_step_requests;
        prev_phase1_batch_indices = subsequent_step_requests;
        
        return all_new_tokens;
    }

private:
    // Extract KV cache from batched tensor at specific batch index, optionally skipping padding
    // With right-padding: padding zeros at START, real data at END
    ov::Tensor extract_kv_cache_slice(const ov::Tensor& batched_tensor, size_t batch_idx, 
                                       size_t padding_to_skip = 0) {
        auto shape = batched_tensor.get_shape();
        // KV cache shape: [batch_size, num_heads, seq_len, head_size]
        size_t num_heads = shape[1];
        size_t seq_len = shape[2];
        size_t head_size = shape[3];
        
        // Real data length after skipping padding
        size_t real_seq_len = seq_len - padding_to_skip;
        
        ov::Tensor slice{batched_tensor.get_element_type(), {1, num_heads, real_seq_len, head_size}};
        const float* src_batch = batched_tensor.data<float>() + batch_idx * num_heads * seq_len * head_size;
        float* dst = slice.data<float>();
        
        if (padding_to_skip == 0) {
            // Fast path: no padding, simple memcpy
            size_t total_size = num_heads * seq_len * head_size;
            std::copy_n(src_batch, total_size, dst);
        } else {
            // With padding: copy per-head, skipping padding at start
            for (size_t h = 0; h < num_heads; ++h) {
                const float* src_head = src_batch + h * seq_len * head_size + padding_to_skip * head_size;
                float* dst_head = dst + h * real_seq_len * head_size;
                std::copy_n(src_head, real_seq_len * head_size, dst_head);
            }
        }
        
        return slice;
    }
    
    // Store KV cache from InferRequest present_* outputs into Request objects
    // For phase 1+ (decoder_with_past), encoder KV is in inputs (past_key_values.*), not outputs
    // With right-padding, need to calculate padding offset for each request to skip padding in extraction
    void extract_kv_cache_to_requests(ov::InferRequest& infer_request, 
                                      const std::vector<size_t>& batch_indices,
                                      bool extract_encoder = true,
                                      bool encoder_from_inputs = false) {
        // Get actual output tensor shape to determine the max length in present.* tensors
        // present.* shape is [batch, heads, new_seq_len, head_size] where new_seq_len = past_len + 1
        auto present_shape = infer_request.get_tensor("present.0.decoder.key").get_shape();
        size_t output_seq_len = present_shape[2];  // This is the max length after inference (max_past + 1)
        
        //std::cout << "        [EXTRACT_DEBUG] Output present.* has seq_len=" << output_seq_len << "\n";
        //for (size_t i = 0; i < batch_indices.size(); ++i) {
        //    const auto& req = requests_map.at(batch_indices[i]);
        //    std::cout << "          Batch " << i << ": sequence_length=" << req.get_sequence_length() 
        //              << ", will extract with padding=" << (output_seq_len - req.get_sequence_length()) << "\n";
        //}
        
        for (size_t i = 0; i < batch_indices.size(); ++i) {
            size_t batch_idx = batch_indices[i];
            auto& request = requests_map.at(batch_idx);
            
            // Calculate padding offset: output has length output_seq_len, this request's real data length is sequence_length
            // With right-padding: padding zeros at start, real data at end
            size_t this_request_len = request.get_sequence_length();
            size_t padding_offset = output_seq_len - this_request_len;
            
            // Extract decoder and encoder KV cache for all layers
            request.get_decoder_key_cache().resize(num_layers);
            request.get_decoder_value_cache().resize(num_layers);
            if (extract_encoder) {
                request.get_encoder_key_cache().resize(num_layers);
                request.get_encoder_value_cache().resize(num_layers);
            }
            
            for (size_t layer = 0; layer < num_layers; ++layer) {
                std::string decoder_key_present = "present." + std::to_string(layer) + ".decoder.key";
                std::string decoder_value_present = "present." + std::to_string(layer) + ".decoder.value";
                
                // Extract decoder KV, skipping right-padding at the start
                request.get_decoder_key_cache()[layer] = extract_kv_cache_slice(
                    infer_request.get_tensor(decoder_key_present), i, padding_offset);
                request.get_decoder_value_cache()[layer] = extract_kv_cache_slice(
                    infer_request.get_tensor(decoder_value_present), i, padding_offset);
                
                if (extract_encoder) {
                    if (encoder_from_inputs) {
                        // For decoder_with_past (phase 1+), encoder KV is in inputs not outputs
                        std::string encoder_key_past = "past_key_values." + std::to_string(layer) + ".encoder.key";
                        std::string encoder_value_past = "past_key_values." + std::to_string(layer) + ".encoder.value";
                        
                        request.get_encoder_key_cache()[layer] = extract_kv_cache_slice(
                            infer_request.get_tensor(encoder_key_past), i);
                        request.get_encoder_value_cache()[layer] = extract_kv_cache_slice(
                            infer_request.get_tensor(encoder_value_past), i);
                    } else {
                        // For decoder (phase 0), encoder KV is in outputs
                        std::string encoder_key_present = "present." + std::to_string(layer) + ".encoder.key";
                        std::string encoder_value_present = "present." + std::to_string(layer) + ".encoder.value";
                        
                        request.get_encoder_key_cache()[layer] = extract_kv_cache_slice(
                            infer_request.get_tensor(encoder_key_present), i);
                        request.get_encoder_value_cache()[layer] = extract_kv_cache_slice(
                            infer_request.get_tensor(encoder_value_present), i);
                    }
                }
            }
        }
    }
    
    // Reconstruct batched KV cache tensors from individual Request storage
    void reconstruct_kv_cache_from_requests(ov::InferRequest& infer_request,
                                            const std::vector<size_t>& batch_indices) {
        //std::cout << "        Reconstructing KV cache for " << batch_indices.size() << " requests: [";
        //for (size_t idx : batch_indices) {
        //    std::cout << requests_map.at(idx).get_request_id() << " ";
        //}
        //std::cout << "]\n";
        
        if (batch_indices.empty()) return;
        
        // Get shape from first request's stored cache
        const auto& first_req = requests_map.at(batch_indices[0]);
        if (first_req.get_decoder_key_cache().empty()) {
            //std::cout << "        ERROR: First request (id=" << first_req.get_request_id() 
            //          << ") has EMPTY decoder cache!\n";
            return;  // No stored cache to reconstruct from
        }
        if (first_req.get_encoder_key_cache().empty()) {
            //std::cout << "        ERROR: First request (id=" << first_req.get_request_id() 
            //          << ") has EMPTY encoder cache!\n";
        }
        //std::cout << "        First request has decoder cache with " << first_req.get_decoder_key_cache().size() << " layers\n";
        
        size_t batch_size = batch_indices.size();
        
        // CRITICAL: Decoder seq_len can differ between requests (especially after phase 0→1 transition)
        // KV cache stored length is sequence_length - 1 (the position we're generating is not in cache yet)
        size_t num_heads = first_req.get_decoder_key_cache()[0].get_shape()[1];
        size_t head_size = first_req.get_decoder_key_cache()[0].get_shape()[3];
        
        size_t max_dec_seq_len = 0;
        //std::cout << "        [DEBUG] Reconstructing KV cache for batch:\n";
        for (size_t batch_idx : batch_indices) {
            const auto& req = requests_map.at(batch_idx);
            size_t past_len = req.get_sequence_length() - 1;  // Past length is always sequence_length - 1
            //std::cout << "          Request " << req.get_request_id() 
            //          << ": past_length=" << past_len
            //          << ", sequence_length=" << req.get_sequence_length() << "\n";
            max_dec_seq_len = std::max(max_dec_seq_len, past_len);
        }
        //std::cout << "        Reconstructing KV cache with max past length: " << max_dec_seq_len << "\n";
        
        // Encoder seq_len is constant for all requests
        auto enc_cache_shape = first_req.get_encoder_key_cache()[0].get_shape();
        size_t enc_seq_len = enc_cache_shape[2];  // Typically 1500 for Whisper encoder
        size_t enc_slice_size = num_heads * enc_seq_len * head_size;
        
        for (size_t layer = 0; layer < num_layers; ++layer) {
            // Create batched tensors with max decoder seq_len
            ov::Tensor batched_dec_key{ov::element::f32, {batch_size, num_heads, max_dec_seq_len, head_size}};
            ov::Tensor batched_dec_value{ov::element::f32, {batch_size, num_heads, max_dec_seq_len, head_size}};
            ov::Tensor batched_enc_key{ov::element::f32, {batch_size, num_heads, enc_seq_len, head_size}};
            ov::Tensor batched_enc_value{ov::element::f32, {batch_size, num_heads, enc_seq_len, head_size}};
            
            // Zero-initialize decoder tensors (for padding)
            std::fill_n(batched_dec_key.data<float>(), batched_dec_key.get_size(), 0.0f);
            std::fill_n(batched_dec_value.data<float>(), batched_dec_value.get_size(), 0.0f);
            
            // Copy from each request's storage (handling variable seq_len)
            for (size_t i = 0; i < batch_size; ++i) {
                const auto& req = requests_map.at(batch_indices[i]);
                size_t req_dec_seq_len = req.get_sequence_length() - 1;  // Past length = sequence_length - 1
                
                // RIGHT-PAD: Place KV at the END of max_dec_seq_len buffer
                // This ensures all requests' new K,V will be placed at position max_dec_seq_len
                size_t padding_positions = max_dec_seq_len - req_dec_seq_len;
                
                //if (layer == 0 && i < 2) {
                //    std::cout << "        [KV_PADDING] Batch " << i << ": RIGHT-padding " 
                //              << req_dec_seq_len << " positions with " << padding_positions 
                //              << " zeros at start, data at end [" << padding_positions << "-" 
                //              << (max_dec_seq_len-1) << "]\n";
                //}
                
                // Copy per-head to achieve right-padding in seq_len dimension
                const float* src_key = req.get_decoder_key_cache()[layer].data<float>();
                const float* src_value = req.get_decoder_value_cache()[layer].data<float>();
                float* dst_key = batched_dec_key.data<float>() + i * num_heads * max_dec_seq_len * head_size;
                float* dst_value = batched_dec_value.data<float>() + i * num_heads * max_dec_seq_len * head_size;
                
                for (size_t h = 0; h < num_heads; ++h) {
                    // Source: [req_dec_seq_len, head_size] for this head
                    const float* src_head_key = src_key + h * req_dec_seq_len * head_size;
                    const float* src_head_value = src_value + h * req_dec_seq_len * head_size;
                    
                    // Destination: offset by padding_positions to place at end
                    float* dst_head_key = dst_key + h * max_dec_seq_len * head_size + padding_positions * head_size;
                    float* dst_head_value = dst_value + h * max_dec_seq_len * head_size + padding_positions * head_size;
                    
                    // Copy this head's KV data
                    std::copy_n(src_head_key, req_dec_seq_len * head_size, dst_head_key);
                    std::copy_n(src_head_value, req_dec_seq_len * head_size, dst_head_value);
                }
                
                // Copy encoder KV (constant size for all requests, no padding needed)
                std::copy_n(req.get_encoder_key_cache()[layer].data<float>(), enc_slice_size,
                           batched_enc_key.data<float>() + i * enc_slice_size);
                std::copy_n(req.get_encoder_value_cache()[layer].data<float>(), enc_slice_size,
                           batched_enc_value.data<float>() + i * enc_slice_size);
            }
            
            // Set reconstructed tensors as past_key_values
            std::string decoder_key_past = "past_key_values." + std::to_string(layer) + ".decoder.key";
            std::string decoder_value_past = "past_key_values." + std::to_string(layer) + ".decoder.value";
            std::string encoder_key_past = "past_key_values." + std::to_string(layer) + ".encoder.key";
            std::string encoder_value_past = "past_key_values." + std::to_string(layer) + ".encoder.value";
            
            infer_request.set_tensor(decoder_key_past, batched_dec_key);
            infer_request.set_tensor(decoder_value_past, batched_dec_value);
            infer_request.set_tensor(encoder_key_past, batched_enc_key);
            infer_request.set_tensor(encoder_value_past, batched_enc_value);
        }
    }
    
    // Run inference for a subset of requests (all in same phase)
    // is_initial_step: true = use decoder model, false = use decoder_with_past model
    std::map<size_t, int64_t> run_inference_batch(const std::vector<size_t>& batch_indices, bool is_initial_step) {
        auto setup_start = std::chrono::high_resolution_clock::now();
        
        size_t batch_size = batch_indices.size();
        ov::InferRequest& infer_request = is_initial_step ? m_request_decoder : m_request_decoder_with_past;
        
        // Cache pointers to avoid repeated hash map lookups
        std::vector<Request*> batch_requests;
        batch_requests.reserve(batch_size);
        for (size_t batch_idx : batch_indices) {
            batch_requests.push_back(&requests_map.at(batch_idx));
        }
        
        // Determine sequence length for input_ids (first call has full prompt, subsequent have single token)
        size_t input_seq_len = is_initial_step ? batch_requests[0]->get_sequence_length() : 1;
        
        auto input_ids_start = std::chrono::high_resolution_clock::now();
        
        // 1. Build batched input_ids
        ov::Tensor input_ids{ov::element::i64, {batch_size, input_seq_len}};
        int64_t* input_ids_data = input_ids.data<int64_t>();
        
        // 2. Find max sequence length for attention_mask
        // CRITICAL: For decoder_with_past, attention_mask must cover BOTH:
        //   - The cached past positions (from past_key_values)
        //   - The current new token being generated (from input_ids)
        // Use sequence_length which is kept up-to-date (not KV cache shape which can be stale/padded)
        size_t max_seq_len = 0;
        for (size_t i = 0; i < batch_size; ++i) {
            // sequence_length tracks current position (past + current token)
            max_seq_len = std::max(max_seq_len, batch_requests[i]->get_sequence_length());
        }
        
        //std::cout << "      Max seq_len: " << max_seq_len << " (batch_size=" << batch_size << ")\n";
        
        auto attention_mask_start = std::chrono::high_resolution_clock::now();
        
        ov::Tensor attention_mask{ov::element::f32, {batch_size, max_seq_len}};
        float* attention_mask_data = attention_mask.data<float>();
        
        // Check if all sequences have the same length
        bool all_same_length = true;
        size_t first_seq_len;
        if (is_initial_step) {
            first_seq_len = batch_requests[0]->get_sequence_length();
        } else {
            // Phase 1+: Use sequence_length - 1 (past positions), not KV cache shape
            // KV cache shape can be padded and doesn't reflect individual request positions
            first_seq_len = batch_requests[0]->get_sequence_length();  // Current position (past + 1)
        }
        
        for (size_t i = 1; i < batch_size; ++i) {
            size_t seq_len;
            if (is_initial_step) {
                seq_len = batch_requests[i]->get_sequence_length();
            } else {
                seq_len = batch_requests[i]->get_sequence_length();  // Current position (past + 1)
            }
            
            if (seq_len != first_seq_len) {
                all_same_length = false;
                break;
            }
        }
        
        if (all_same_length) {
            // All same: optimized fill
            std::fill_n(attention_mask_data, batch_size * max_seq_len, 1.0f);
            //std::cout << "      [DEBUG] Attention mask: all same length (" 
            //          << batch_size << " x " << max_seq_len << ")"
            //          << (is_initial_step ? " [Phase 0]" : " [Phase 1+]") << "\n";
        } else {
            // Different lengths: handle padding (can happen after mixed-phase batching)
            //std::cout << "      [DEBUG] Attention mask: variable lengths detected (using RIGHT-PADDING)\n";
            for (size_t i = 0; i < batch_size; ++i) {
                size_t seq_len;
                if (is_initial_step) {
                    seq_len = batch_requests[i]->get_sequence_length();
                } else {
                    // Phase 1+: sequence_length is current position (includes all past + current)
                    seq_len = batch_requests[i]->get_sequence_length();
                }
                
                // RIGHT-PAD: Place attention at the END to match right-padded KV cache
                size_t padding = max_seq_len - seq_len;
                float* row = attention_mask_data + i * max_seq_len;
                std::fill_n(row, padding, 0.0f);  // Padding zeros at start
                std::fill_n(row + padding, seq_len, 1.0f);  // Real positions at end
                
                //if (i < 2) {
                //    std::cout << "        Batch pos " << i << ": padding=" << padding 
                //              << ", real_positions=" << seq_len << " at end\n";
                //}
            }
        }
        
        // Print full attention mask for debugging
        //std::cout << "      [ATTENTION_MASK] Shape: [" << batch_size << ", " << max_seq_len << "]\n";
        //for (size_t i = 0; i < batch_size; ++i) {
        //    std::cout << "        Row " << i << ": [";
        //    for (size_t j = 0; j < max_seq_len; ++j) {
        //        std::cout << (attention_mask_data[i * max_seq_len + j] > 0.5f ? "1" : "0");
        //        if (j < max_seq_len - 1) std::cout << " ";
        //    }
        //    std::cout << "]\n";
        //}
        
        auto encoder_start = std::chrono::high_resolution_clock::now();
        
        // 3. Build batched encoder_hidden_states (only for initial step or batch change)
        // For phase 1+ with stable batch, encoder_hidden_states stays in InferRequest
        // Check if this is a clean phase 0→1 transition where we can link from decoder
        bool is_clean_phase_transition = !is_initial_step && 
                                         !prev_phase0_batch_indices.empty() &&
                                         batch_size == prev_phase0_batch_indices.size() &&
                                         std::equal(batch_indices.begin(), batch_indices.end(), 
                                                   prev_phase0_batch_indices.begin());
        
        if (is_clean_phase_transition) {
            // Zero-copy: link encoder_hidden_states from decoder to decoder_with_past
            infer_request.set_tensor("encoder_hidden_states", 
                                     m_request_decoder.get_tensor("encoder_hidden_states"));
        } else {
            // Check if rebuild needed
            bool needs_encoder_rebuild = is_initial_step || 
                                        prev_phase1_batch_indices.empty() || 
                                        batch_size != prev_phase1_batch_indices.size();
            
            if (needs_encoder_rebuild) {
                auto enc_shape = batch_requests[0]->get_encoder_hidden_state().get_shape();
                ov::Tensor encoder_hidden_states{ov::element::f32, {batch_size, enc_shape[1], enc_shape[2]}};
                float* enc_data = encoder_hidden_states.data<float>();
                size_t enc_size = enc_shape[1] * enc_shape[2];
                
                // Fill encoder_hidden_states for each request
                for (size_t i = 0; i < batch_size; ++i) {
                    const float* src_enc = batch_requests[i]->get_encoder_hidden_state().data<float>();
                    std::copy(src_enc, src_enc + enc_size, enc_data + i * enc_size);
                }
                
                infer_request.set_tensor("encoder_hidden_states", encoder_hidden_states);
            }
        }
        
        auto fill_tensors_start = std::chrono::high_resolution_clock::now();
        
        // Fill tensors for each request in this batch
        for (size_t i = 0; i < batch_size; ++i) {
            // Set input_ids
            if (is_initial_step) {
                // Copy full initial prompt for first inference
                const int64_t* src_ids = batch_requests[i]->get_initial_prompt_ids().data<int64_t>();
                std::copy(src_ids, src_ids + input_seq_len, input_ids_data + i * input_seq_len);
                
                // Debug: Print input_ids for first batch position
                if (i == 0 || i == 1) {
                    //std::cout << "      [DEBUG] Batch position " << i << " input_ids: [";
                    //for (size_t j = 0; j < std::min(input_seq_len, size_t(10)); ++j) {
                    //    std::cout << input_ids_data[i * input_seq_len + j];
                    //    if (j < std::min(input_seq_len, size_t(10)) - 1) std::cout << ", ";
                    //}
                    //std::cout << "]\n";
                }
            } else {
                input_ids_data[i] = batch_requests[i]->get_last_token();
            }
        }
        
        // Set inputs
        infer_request.set_tensor("input_ids", input_ids);
        infer_request.set_tensor("attention_mask", attention_mask);
        
        auto kv_cache_start = std::chrono::high_resolution_clock::now();
        
        // For decoder_with_past, manage KV cache linkage/reconstruction
        if (!is_initial_step) {
            // For stable batch in phase 1 (most common), batch_indices == prev_phase1_batch_indices
            // This pointer comparison is faster than vector comparison
            bool is_exact_same_batch = (&batch_indices == &prev_phase1_batch_indices) || 
                                       (batch_indices.size() == prev_phase1_batch_indices.size() && 
                                        batch_indices.data() == prev_phase1_batch_indices.data());
            
            if (is_exact_same_batch || (!prev_phase0_batch_indices.empty() && 
                                        batch_indices.size() == prev_phase0_batch_indices.size() && 
                                        std::equal(batch_indices.begin(), batch_indices.end(), 
                                                  prev_phase0_batch_indices.begin()))) {
                // Case 1: Exact same batch (fast path for phase 1 stable), OR
                // Case 2: Clean phase 0→1 transition
                bool is_phase_transition = !prev_phase0_batch_indices.empty();
                //std::cout << "      [DEBUG] KV cache: " << (is_phase_transition ? "Zero-copy linking (phase 0→1 transition)" : "Zero-copy linking (stable batch)") << "\n";
                
                ov::InferRequest& source_request = is_phase_transition ? m_request_decoder : infer_request;
                
                // Debug: Print shapes on first phase transition
                if (is_phase_transition && batch_size > 0) {
                    auto dec_key_shape = source_request.get_tensor("present.0.decoder.key").get_shape();
                    //std::cout << "        Source decoder KV shape: [";
                    //for (size_t i = 0; i < dec_key_shape.size(); ++i) {
                    //    std::cout << dec_key_shape[i];
                    //    if (i < dec_key_shape.size() - 1) std::cout << ", ";
                    //}
                    //std::cout << "]\n";
                    //std::cout << "        Target attention_mask shape: [" << batch_size << ", " << max_seq_len << "]\n";
                    //std::cout << "        Target input_ids shape: [" << batch_size << ", " << input_seq_len << "]\n";
                }
                
                for (size_t layer = 0; layer < num_layers; ++layer) {
                    std::string decoder_key_past = "past_key_values." + std::to_string(layer) + ".decoder.key";
                    std::string decoder_value_past = "past_key_values." + std::to_string(layer) + ".decoder.value";
                    std::string decoder_key_present = "present." + std::to_string(layer) + ".decoder.key";
                    std::string decoder_value_present = "present." + std::to_string(layer) + ".decoder.value";
                    
                    // Direct linking for decoder KV cache - no copy!
                    infer_request.set_tensor(decoder_key_past, source_request.get_tensor(decoder_key_present));
                    infer_request.set_tensor(decoder_value_past, source_request.get_tensor(decoder_value_present));
                    
                    if (is_phase_transition) {
                        // Also link encoder KV from phase 0 outputs
                        std::string encoder_key_past = "past_key_values." + std::to_string(layer) + ".encoder.key";
                        std::string encoder_value_past = "past_key_values." + std::to_string(layer) + ".encoder.value";
                        std::string encoder_key_present = "present." + std::to_string(layer) + ".encoder.key";
                        std::string encoder_value_present = "present." + std::to_string(layer) + ".encoder.value";
                        
                        infer_request.set_tensor(encoder_key_past, source_request.get_tensor(encoder_key_present));
                        infer_request.set_tensor(encoder_value_past, source_request.get_tensor(encoder_value_present));
                    }
                }
            } else {
                // Batch composition changed: need to reconstruct from stored caches
                //std::cout << "      KV cache: Reconstructing from stored caches (batch composition changed)\n";
                reconstruct_kv_cache_from_requests(infer_request, batch_indices);
            }
        }
        
        auto kv_cache_end = std::chrono::high_resolution_clock::now();
        auto inference_start = std::chrono::high_resolution_clock::now();
        
        // Run inference
        infer_request.infer();
        
        auto inference_end = std::chrono::high_resolution_clock::now();
        auto token_extraction_start = std::chrono::high_resolution_clock::now();
        
        // Extract results: get new token for each request
        ov::Tensor logits = infer_request.get_tensor("logits");
        auto logits_shape = logits.get_shape();
        size_t seq_len = logits_shape[1];
        size_t vocab_size = logits_shape[2];
        float* logits_data = logits.data<float>();
        
        //std::cout << "      [DEBUG] Logits shape: [" << logits_shape[0] << ", " 
        //          << logits_shape[1] << ", " << logits_shape[2] << "]\n";
        
        std::map<size_t, int64_t> new_tokens;
        for (size_t i = 0; i < batch_size; ++i) {
            size_t batch_idx = batch_indices[i];
            
            // Get last token's logits
            size_t last_token_offset = i * seq_len * vocab_size + (seq_len - 1) * vocab_size;
            float* last_token_logits = logits_data + last_token_offset;
            
            // Greedy sampling
            int64_t best_token = 0;
            float best_score = last_token_logits[0];
            for (size_t token = 1; token < vocab_size; ++token) {
                if (last_token_logits[token] > best_score) {
                    best_score = last_token_logits[token];
                    best_token = static_cast<int64_t>(token);
                }
            }
            
            if (i < 2) {
                //std::cout << "      [DEBUG] Batch position " << i << " generated token: " 
                //          << best_token << " (score: " << best_score << ")\n";
            }
            
            new_tokens[batch_idx] = best_token;
        }
        
        auto extraction_check_start = std::chrono::high_resolution_clock::now();
        
        // Update tracking: For phase 1, extract KV cache only once (first phase 1 call)
        // After that, caches are extracted and we use zero-copy
        // LAZY EXTRACTION: Only extract if batch composition might change
        if (!is_initial_step) {
            // Check if this is clean phase transition or stable batch
            bool is_clean_phase_transition = !prev_phase0_batch_indices.empty() &&
                                           batch_indices.size() == prev_phase0_batch_indices.size() &&
                                           std::equal(batch_indices.begin(), batch_indices.end(), 
                                                     prev_phase0_batch_indices.begin());
            
            bool is_stable_phase1 = (&batch_indices == &prev_phase1_batch_indices) || 
                                   (batch_indices.size() == prev_phase1_batch_indices.size() && 
                                    batch_indices.data() == prev_phase1_batch_indices.data());
            
            // Only extract if batch is neither clean transition nor stable phase 1
            // This means batch composition changed and we'll need stored KV for reconstruction
            if (!is_clean_phase_transition && !is_stable_phase1) {
                // Batch composition changed - need to extract for reconstruction
                bool might_need_extraction = batch_requests[0]->get_decoder_key_cache().empty() || 
                                            batch_requests[0]->get_encoder_key_cache().empty();
                
                if (might_need_extraction) {
                    extract_kv_cache_to_requests(infer_request, batch_indices, true, true);
                }
            }
            // Otherwise: let KV caches stay in InferRequest tensors (zero-copy)
        }
        
        auto extraction_check_end = std::chrono::high_resolution_clock::now();
        
        // Print detailed timing breakdown
        auto setup_time = std::chrono::duration_cast<std::chrono::microseconds>(input_ids_start - setup_start);
        auto input_ids_time = std::chrono::duration_cast<std::chrono::microseconds>(attention_mask_start - input_ids_start);
        auto attention_mask_time = std::chrono::duration_cast<std::chrono::microseconds>(encoder_start - attention_mask_start);
        auto encoder_time = std::chrono::duration_cast<std::chrono::microseconds>(fill_tensors_start - encoder_start);
        auto fill_time = std::chrono::duration_cast<std::chrono::microseconds>(kv_cache_start - fill_tensors_start);
        auto kv_cache_time = std::chrono::duration_cast<std::chrono::microseconds>(kv_cache_end - kv_cache_start);
        auto inference_time = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start);
        auto token_time = std::chrono::duration_cast<std::chrono::microseconds>(extraction_check_start - token_extraction_start);
        auto extraction_time = std::chrono::duration_cast<std::chrono::microseconds>(extraction_check_end - extraction_check_start);
        
        // Print timing statistics only if OPENVINO_LOG_LEVEL > 1
        const char* log_level_str = std::getenv("OPENVINO_LOG_LEVEL");
        int log_level = 0;
        if (log_level_str) {
            try {
            log_level = std::stoi(log_level_str);
            } catch (...) {
            log_level = 0;
            }
        }
        
        if (log_level > 1) {
            std::cout << "  [Timing] Setup: " << setup_time.count() 
                  << " μs, InputIDs: " << input_ids_time.count()
                  << " μs, AttentionMask: " << attention_mask_time.count()
                  << " μs, Encoder: " << encoder_time.count()
                  << " μs, FillTensors: " << fill_time.count()
                  << " μs, KVCache: " << kv_cache_time.count()
                  << " μs, Inference: " << inference_time.count()
                  << " μs, TokenExtract: " << token_time.count()
                  << " μs, ExtractionCheck: " << extraction_time.count() << " μs\n";
        }
        
        return new_tokens;
    }

public:
    
    void update_requests_after_inference(const std::map<size_t, int64_t>& new_tokens) {
        // After inference, update request state with new tokens
        // KV cache remains in the InferRequest tensors for zero-copy optimization
        // Only extract and store if batch composition will change
        
        for (const auto& [batch_idx, new_token] : new_tokens) {
            if (requests_map.count(batch_idx)) {
                auto& request = requests_map.at(batch_idx);
                request.set_last_token(new_token);
                request.increment_sequence_length();
                request.mark_initial_step_done();
                
                // Note: KV cache is NOT extracted here for efficiency
                // It stays in m_request_decoder/m_request_decoder_with_past tensors
                // and is reused via direct tensor linking in next inference
                // Only extract when a request is removed or batch rebalancing is needed
            }
        }
    }
    
    // === ASYNC API FOR THREAD-SAFE OPERATION ===
    
    // Get number of active requests (thread-safe)
    size_t active_request_count() const {
        std::lock_guard<std::mutex> lock(batch_mutex);
        return requests_map.size();
    }
    
    // Add request from any thread (returns request ID immediately)
    // This is fast - just queues the request without processing
    size_t add_request(ov::Tensor encoder_hidden_state, ov::Tensor input_ids) {
        size_t request_id = next_request_id++;
        
        std::cout << "BatchManager::add_request - request " << request_id 
                  << ", encoder_hidden_state shape: [";
        for (size_t i = 0; i < encoder_hidden_state.get_shape().size(); ++i) {
            std::cout << encoder_hidden_state.get_shape()[i];
            if (i < encoder_hidden_state.get_shape().size() - 1) std::cout << ", ";
        }
        std::cout << "], input_ids shape: [";
        for (size_t i = 0; i < input_ids.get_shape().size(); ++i) {
            std::cout << input_ids.get_shape()[i];
            if (i < input_ids.get_shape().size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        
        // Add to awaiting queue (will be processed in next step())
        PendingRequest pending;
        pending.request_id = request_id;
        pending.encoder_hidden_state = encoder_hidden_state;
        pending.input_ids = input_ids;
        
        {
            std::lock_guard<std::mutex> lock(awaiting_mutex);
            awaiting_requests.push(std::move(pending));
        }
        awaiting_cv.notify_one();  // Wake up generation thread
        
        // Initialize request state for async tracking
        RequestState state;
        state.result_promise = std::make_shared<std::promise<std::vector<int64_t>>>();
        {
            std::lock_guard<std::mutex> lock(request_states_mutex);
            request_states[request_id] = std::move(state);
        }
        
        return request_id;
    }
    
    // Get future for async result retrieval
    std::future<std::vector<int64_t>> get_result_future(size_t request_id) {
        std::lock_guard<std::mutex> lock(request_states_mutex);
        return request_states.at(request_id).result_promise->get_future();
    }
    
    // Run one generation step (called by generation thread)
    // Returns true if step was executed, false if no active requests
    bool step() {
        // Step 1: Check and pull pending requests from awaiting queue
        std::vector<PendingRequest> pending_to_activate;
        {
            std::lock_guard<std::mutex> lock(awaiting_mutex);
            while (!awaiting_requests.empty()) {
                pending_to_activate.push_back(std::move(awaiting_requests.front()));
                awaiting_requests.pop();
            }
        }
        
        // Step 2: Activate pulled requests in batch (with batch_mutex)
        std::unique_lock<std::mutex> lock(batch_mutex);
        
        // Check stop flag
        if (should_stop.load()) {
            return false;
        }
        
        // Activate all pending requests
        for (auto& pending : pending_to_activate) {
            //std::cout << "BatchManager::step - activating request " << pending.request_id << "\n";
            
            // Add to active requests map
            size_t batch_idx = reserve_slot_for_request(pending.encoder_hidden_state, pending.input_ids);
            
            // Store request_id for later mapping back to promise
            requests_map.at(batch_idx).set_request_id(pending.request_id);
        }
        
        // CRITICAL FIX: If we added new phase-0 requests while phase-1 requests exist,
        // force KV cache reconstruction to prevent contamination
        if (!pending_to_activate.empty() && !prev_phase1_batch_indices.empty()) {
            //std::cout << "BatchManager: New requests added while phase-1 active - forcing KV cache reconstruction\n";
            // Clear tracking to force reconstruction from stored caches
            prev_phase0_batch_indices.clear();
            prev_phase1_batch_indices.clear();
        }
        
        // If no active requests, wait for new arrivals
        if (requests_map.empty()) {
            lock.unlock();
            
            // Wait on awaiting_cv (new requests arriving)
            std::unique_lock<std::mutex> await_lock(awaiting_mutex);
            awaiting_cv.wait_for(await_lock, std::chrono::milliseconds(10), [this] {
                return !awaiting_requests.empty() || should_stop.load();
            });
            
            // Check stop flag after waking up
            if (should_stop.load()) {
                return false;
            }
            
            // Pull newly arrived requests
            std::vector<PendingRequest> new_pending;
            while (!awaiting_requests.empty()) {
                new_pending.push_back(std::move(awaiting_requests.front()));
                awaiting_requests.pop();
            }
            await_lock.unlock();
            
            if (new_pending.empty()) {
                return false;  // No requests to process
            }
            
            // Activate them
            lock.lock();
            for (auto& pending : new_pending) {
                //std::cout << "BatchManager::step - activating request " << pending.request_id << "\n";
                size_t batch_idx = reserve_slot_for_request(pending.encoder_hidden_state, pending.input_ids);
                requests_map.at(batch_idx).set_request_id(pending.request_id);
            }
        }
        
        // Reset decoders if needed (when starting fresh after batch was empty)
        if (need_decoder_reset) {
            //std::cout << "BatchManager: Resetting decoder InferRequests to clean state...\n";
            m_request_decoder = m_compiled_decoder.create_infer_request();
            m_request_decoder_with_past = m_compiled_decoder_with_past.create_infer_request();
            
            // Clear batch composition tracking so KV cache links are rebuilt
            prev_phase0_batch_indices.clear();
            prev_phase1_batch_indices.clear();
            
            need_decoder_reset = false;
            //std::cout << "BatchManager: Decoder reset complete\n";
        }
        
        // Snapshot current batch for inference BEFORE unlocking
        size_t current_batch_count = requests_map.size();
        std::cout << "BatchManager::step - running inference for " << current_batch_count << " requests\n";
        
        // Release lock during inference to allow new requests to be added in parallel
        // This is safe because:
        // - requests_map insertions don't invalidate existing entries
        // - run_inference() only reads from existing requests (doesn't modify map structure)
        // - New requests added during inference will be processed in next step()
        lock.unlock();
        
        // Run inference for current batch (without holding batch_mutex)
        auto new_tokens = run_inference();
        
        // CRITICAL: Check for new requests that arrived during inference BEFORE relocking
        // This minimizes time holding batch_mutex
        std::vector<PendingRequest> late_arrivals;
        {
            std::lock_guard<std::mutex> await_lock(awaiting_mutex);
            while (!awaiting_requests.empty()) {
                late_arrivals.push_back(std::move(awaiting_requests.front()));
                awaiting_requests.pop();
            }
        }
        
        // Reacquire lock to update state
        lock.lock();
        
        // Activate late arrival requests
        for (auto& pending : late_arrivals) {
            //std::cout << "BatchManager::step - activating late-arrival request " << pending.request_id << "\n";
            size_t batch_idx = reserve_slot_for_request(pending.encoder_hidden_state, pending.input_ids);
            requests_map.at(batch_idx).set_request_id(pending.request_id);
        }
        
        if (!late_arrivals.empty()) {
            //std::cout << "BatchManager::step - batch size increased from " << (requests_map.size() - late_arrivals.size())
            //          << " to " << requests_map.size() << " requests\n";
            
            // Force KV cache reconstruction when late arrivals join
            //std::cout << "BatchManager: Late-arrival requests detected (" << late_arrivals.size() 
            //          << ") - forcing KV cache reconstruction\n";
            prev_phase0_batch_indices.clear();
            prev_phase1_batch_indices.clear();
        }
        
        update_requests_after_inference(new_tokens);
        
        // Check for completed requests
        std::vector<size_t> completed_batch_indices;
        {
            std::lock_guard<std::mutex> states_lock(request_states_mutex);
            for (const auto& [batch_idx, token] : new_tokens) {
                // Get request_id from the Request object
                size_t request_id = requests_map.at(batch_idx).get_request_id();
                auto& state = request_states[request_id];
                state.generated_tokens.push_back(token);
                
                // Check EOS or max length
                if (token == eos_token_id || state.generated_tokens.size() >= max_decode_steps) {
                    state.completed = true;
                    state.result_promise->set_value(state.generated_tokens);
                    completed_batch_indices.push_back(batch_idx);
                }
            }
        }
        
        // Clean up completed requests
        for (size_t batch_idx : completed_batch_indices) {
            size_t request_id = requests_map.at(batch_idx).get_request_id();
            release_slot(batch_idx);
            {
                std::lock_guard<std::mutex> states_lock(request_states_mutex);
                request_states.erase(request_id);
            }
        }
        
        // Release batch_mutex - allows next step() to begin immediately
        lock.unlock();
        
        return true;  // Step executed
    }
    
    // Check if any requests are active
    bool has_active_requests() {
        std::lock_guard<std::mutex> lock(batch_mutex);
        return !requests_map.empty();
    }
    
    // Stop signal
    void stop() {
        should_stop = true;
        awaiting_cv.notify_all();
        batch_cv.notify_all();
    }
    
    bool should_continue() {
        return !should_stop.load();
    }
    
    void set_eos_token_id(int64_t token_id) {
        eos_token_id = token_id;
    }
    
    void set_max_decode_steps(size_t steps) {
        max_decode_steps = steps;
    }
};


class BatchManagerNew {
    /*


Assumptions:

Different approach. Preallocated KV cache buffers for max batch size, and track which slots are occupied.
Incremental updates to KV cache tensors instead of full reconstruction when batch changes.

We have two models: 
1. decoder without past used for phase 0 with newly received requests - inputs: input_ids, encoder_hidden_state, attention mask - outputs: logits, present KV
2. decoder with past for all next steps until completion - inputs: input_ids, encoder_hidden_state, attention mask, past KV - outputs: logits, present KV

For 2. physically static buffer, logically dynamic. We track which slots in the batch are occupied by which requests, and manage KV cache updates accordingly.
For every request we track: batch slot index, sequence length, last saved KV

For 1. every request is processed just once, so we need to copy anyway - can be dynamic.

We have three separate containers: awaiting requests, init_phase_requests (for phase0), requests (all requests in progress, after phase0)

Flow (per step):

1. Check awaiting requests (lock the container) - are there any? Do we have capacity to process them now?
2. Pull awaiting requests (according to possible capacity), prepare inputs for phase0 inference
3. Check if phase1+ got any new or completed requests (batch changes)
	3.1 If no: directly link output KV from previous step with input in current step (zero-copy), keep track of new KV that is not stored
	3.2 If yes: 
		- check if we had some runs with zero-copy, in such case we need to update stored KV cache with KV for X recent tokens that were generated during optimized runs
		- create inputs based on stored KV, active requests and counters (full copy here - unless we want to trade compute power and run inference on free batch slots too?)
4. Run inferences (phase0 and phase1+
5. Extract tokens, determine which requests finished, for requests in phase1+ free batch slot
6. For ongoing phase1+ requests extract KV for new token and write it to stored KV at last position, selectively switch attention mask position 0->1 (can we do that in right-padding?) (selective writes, no full KV copy), increase counters
7. For requests from phase0 - transition to phase1+. Assign batch slot, and store KV and encoder_hidden_state



Questions:
	1. X zero-copy runs, new request arrives - do we need to copy whole state to reconstruct? is there a smarter way to reuse stored data chunks?
	2. X zero-copy runs, request completes - do we need full reconstruction? if there is a new request in the queue it could take completed request place in the batch, if not maybe it makes more sense to leave it (pay compute cost instead of copy?)

    */
};


// ----------- End cache management -------------

// ---------- Pipeline implementation -------------
class WhisperSpeechEncoder {
    // TODO: use pool of infer requests for better performance in multithreaded scenarios
    ov::InferRequest m_encoder;
    ov::CompiledModel m_compiled_encoder;  // Store for creating additional InferRequests
    std::shared_ptr<WhisperDecoder> m_decoder;
    WhisperFeatureExtractor m_feature_extractor;
    WhisperConfig m_model_config;

    std::vector<int64_t> prepare_init_tokens(ov::Tensor& encoder_hidden_state,
                                         std::shared_ptr<ov::genai::WhisperDecoder> decoder,
                                         const ov::genai::WhisperGenerationConfig& config,
                                         const bool return_timestamps,
                                         ov::genai::RawPerfMetrics& raw_metrics) {
        if (!config.is_multilingual) {
            if (return_timestamps) {
                return std::vector<int64_t>{config.decoder_start_token_id};
            } else {
                return std::vector<int64_t>{config.decoder_start_token_id, config.no_timestamps_token_id};
            }
        }

        int64_t language_token_id = 0;
        if (config.language.has_value()) {
            std::string language = *config.language;
            if (config.lang_to_id.count(language)) {
                language_token_id = config.lang_to_id.at(language);
            }
        } else {
            auto [language_token, infer_ms] = decoder->detect_language(encoder_hidden_state, config.decoder_start_token_id);
            language_token_id = language_token;
            raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
        }

        int64_t task_token_id = config.transcribe_token_id;
        if (config.task.has_value() && *config.task == "translate") {
            task_token_id = config.translate_token_id;
        }

        if (return_timestamps) {
            return std::vector<int64_t>{config.decoder_start_token_id, language_token_id, task_token_id};
        }

        return std::vector<int64_t>{config.decoder_start_token_id,
                                    language_token_id,
                                    task_token_id,
                                    config.no_timestamps_token_id};
    }

public:
    WhisperSpeechEncoder(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties)
    : m_feature_extractor(models_path / "preprocessor_config.json"),
      m_model_config(models_path / "config.json") {
        std::cout << "Speech encoder constructor called" << std::endl;
        ov::Core core;
        auto model = core.read_model(models_path / "openvino_encoder_model.xml");
        m_compiled_encoder = core.compile_model(model, device, properties);
        m_encoder = m_compiled_encoder.create_infer_request();
        // For now stateful decoder to create init input_ids 
        m_decoder = WhisperDecoder::from_path(models_path, device, properties, m_encoder.get_compiled_model().output("last_hidden_state").get_partial_shape(), true);
    }
    
    // Create a fresh InferRequest for thread-safe encoding
    ov::InferRequest create_encoder_infer_request() {
        return m_compiled_encoder.create_infer_request();
    }

    // Encode raw speech into decoder inputs (input_ids and encoder_hidden_state) for each chunk
    std::vector<std::pair<ov::Tensor, ov::Tensor>> encode(const RawSpeechInput& raw_speech, const WhisperContextTokens& context_tokens, const ov::genai::WhisperGenerationConfig& config) {
        return encode_with_infer_request(m_encoder, raw_speech, context_tokens, config);
    }
    
    // Encode with a specific InferRequest (for thread-safe multi-request scenarios)
    std::vector<std::pair<ov::Tensor, ov::Tensor>> encode_with_infer_request(
        ov::InferRequest& encoder,
        const RawSpeechInput& raw_speech,
        const WhisperContextTokens& context_tokens,
        const ov::genai::WhisperGenerationConfig& config
    ) {
        //std::cout << "Starting SpeechEncoder::encode_with_infer_request\n";
        auto input_features = m_feature_extractor.extract(raw_speech);
        //std::cout << "Extracted features: " << input_features.n_frames << " frames\n";
        size_t segment_offset = 0;

        OPENVINO_ASSERT(m_feature_extractor.sampling_rate != 0, "Sampling Rate for Feature Extractor is 0");
        const float frame_length_in_seconds =
            static_cast<float>(m_feature_extractor.hop_length) / m_feature_extractor.sampling_rate;

        std::vector<std::pair<ov::Tensor, ov::Tensor>> decoder_inputs;
        for (size_t chunk_offset = 0; chunk_offset < input_features.n_frames; chunk_offset += segment_offset) {
            auto input_features_chunk = input_features.get_data_with_offset(chunk_offset, m_feature_extractor.nb_max_frames);
            OPENVINO_ASSERT(input_features_chunk.size() == m_feature_extractor.feature_size * m_feature_extractor.nb_max_frames,
                            "Mel spectrogram required size mismatch");
            
            ov::Tensor input_tensor(ov::element::f32, {1, m_feature_extractor.feature_size, m_feature_extractor.nb_max_frames}, input_features_chunk.data());
            encoder.set_tensor("input_features", input_tensor);
            //std::cout << "Running inference for chunk at offset " << chunk_offset << "\n";
            encoder.infer();
            //std::cout << "Encoder inference completed\n";

            ov::Tensor encoder_output = encoder.get_tensor("last_hidden_state");
            ov::Tensor encoder_hidden_state(encoder_output.get_element_type(), encoder_output.get_shape());
            encoder_output.copy_to(encoder_hidden_state);

            ov::genai::RawPerfMetrics raw_metrics;
            auto init_tokens = prepare_init_tokens(encoder_hidden_state, m_decoder, config, false, raw_metrics);
            std::vector<int64_t> chunk_init_tokens = ov::genai::get_prompt_tokens(context_tokens, config, chunk_offset);
            chunk_init_tokens.insert(chunk_init_tokens.end(), init_tokens.begin(), init_tokens.end());
            
            //std::cout << "chunk_init_tokens: ";
            //for (const auto& token : chunk_init_tokens) {
            //    std::cout << token << " ";
            //}
            //std::cout << std::endl;

            ov::Tensor input_ids{ov::element::i64, {1, chunk_init_tokens.size()}};
            std::copy(chunk_init_tokens.begin(), chunk_init_tokens.end(), input_ids.data<int64_t>());

            decoder_inputs.emplace_back(std::make_pair(input_ids, encoder_hidden_state));
            segment_offset = std::min(input_features.n_frames, m_feature_extractor.nb_max_frames);
        }
        //std::cout << "encode_with_infer_request completed, returning " << decoder_inputs.size() << " chunks\n";
        return decoder_inputs;
    }

    // Merge multiple decoder inputs into single batch tensors
    std::pair<ov::Tensor, ov::Tensor> merge_decoder_inputs(const std::vector<std::pair<ov::Tensor, ov::Tensor>>& decoder_inputs) {
        //std::cout << "Starting SpeechEncoder::merge_decoder_inputs\n";
        size_t batch_size = decoder_inputs.size();
        
        // At the beginning all input ids and encoder hidden states have the same shape 
        // encoder_hidden_state shape: [1 1500 1280]
        // input_ids shape: [1 4]
        // No padding needed here

        // Merge those tensors from decoder_inputs into two ov::Tensors
        ov::Tensor merged_input_ids = ov::Tensor(ov::element::i64, {batch_size, decoder_inputs[0].first.get_shape()[1]});
        ov::Tensor merged_encoder_hidden_state = ov::Tensor(ov::element::f32, {batch_size, decoder_inputs[0].second.get_shape()[1], decoder_inputs[0].second.get_shape()[2]});
        for (size_t i = 0; i < batch_size; ++i) {
            // Copy input_ids
            int64_t* dest_input_ids = merged_input_ids.data<int64_t>() + i * decoder_inputs[0].first.get_shape()[1];
            const int64_t* src_input_ids = decoder_inputs[i].first.data<int64_t>();
            std::copy(src_input_ids, src_input_ids + decoder_inputs[i].first.get_shape()[1], dest_input_ids);

            // Copy encoder_hidden_state
            float* dest_encoder_hidden_state = merged_encoder_hidden_state.data<float>() + i * decoder_inputs[0].second.get_shape()[1] * decoder_inputs[0].second.get_shape()[2];
            const float* src_encoder_hidden_state = decoder_inputs[i].second.data<float>();
            std::copy(src_encoder_hidden_state, src_encoder_hidden_state + decoder_inputs[i].second.get_shape()[1] * decoder_inputs[i].second.get_shape()[2], dest_encoder_hidden_state);
        }
        return {merged_input_ids, merged_encoder_hidden_state};
    }
};


class WhisperPipelinePocImpl {
public:
    WhisperGenerationConfig m_generation_config;
    Tokenizer m_tokenizer;
    WhisperSpeechEncoder m_speech_encoder;
    WhisperConfig m_model_config;

    // For first run
    ov::InferRequest m_request_decoder;
    // For consecutive runs when we have kv cache
    ov::InferRequest m_request_decoder_with_past;

    bool initial_step = true;
    size_t m_sequence_position = 0;  // Track total tokens processed (for attention_mask)

    float m_load_time_ms = 0;
    
    // Continuous batching async support
    std::unique_ptr<BatchManager> m_batch_manager;
    std::thread m_generation_thread;
    std::atomic<bool> m_generation_running{false};
    std::mutex m_encode_mutex;  // Protect encoder from concurrent access

    WhisperPipelinePocImpl(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties)
        : m_generation_config(utils::from_config_json_if_exists<WhisperGenerationConfig>(models_path)),
          m_tokenizer{models_path},
          m_speech_encoder{models_path, device, properties},
          m_model_config{models_path / "config.json"} {
            ov::Core core = utils::singleton_core();
            auto decoder_model = core.read_model(models_path / "openvino_decoder_model.xml");
            add_attention_mask_input(decoder_model);  // Add attention mask support
            auto compiled_decoder_model = core.compile_model(decoder_model, device, properties);

            auto decoder_with_past_model = core.read_model(models_path / "openvino_decoder_with_past_model.xml");
            add_attention_mask_input(decoder_with_past_model);  // Add for with_past too
            auto compiled_decoder_with_past_model = core.compile_model(decoder_with_past_model, device, properties);
            m_request_decoder = compiled_decoder_model.create_infer_request();
            m_request_decoder_with_past = compiled_decoder_with_past_model.create_infer_request();

            // Print decoder inputs
            std::cout << "=== Decoder Model Inputs ===" << std::endl;
            for (const auto& input : compiled_decoder_model.inputs()) {
                std::cout << "Name: " << input.get_any_name() << ", Shape: " << input.get_partial_shape() << ", Type: " << input.get_element_type() << std::endl;
            }

            // Print decoder outputs
            std::cout << "=== Decoder Model Outputs ===" << std::endl;
            for (const auto& output : compiled_decoder_model.outputs()) {
                std::cout << "Name: " << output.get_any_name() << ", Shape: " << output.get_partial_shape() << ", Type: " << output.get_element_type() << std::endl;
            }

            // Print decoder with past inputs
            std::cout << "=== Decoder With Past Model Inputs ===" << std::endl;
            for (const auto& input : compiled_decoder_with_past_model.inputs()) {
                std::cout << "Name: " << input.get_any_name() << ", Shape: " << input.get_partial_shape() << ", Type: " << input.get_element_type() << std::endl;
            }

            // Print decoder with past outputs
            std::cout << "=== Decoder With Past Model Outputs ===" << std::endl;
            for (const auto& output : compiled_decoder_with_past_model.outputs()) {
                std::cout << "Name: " << output.get_any_name() << ", Shape: " << output.get_partial_shape() << ", Type: " << output.get_element_type() << std::endl;
            }
          }


    //virtual WhisperDecodedResults generate(const RawSpeechInput& raw_speech_input,
    //                                       OptionalWhisperGenerationConfig generation_config,
    //                                       const std::shared_ptr<StreamerBase> streamer) = 0;

    std::vector<std::pair<ov::Tensor, ov::Tensor>> encode(const RawSpeechInput& raw_speech, const WhisperContextTokens& context_tokens, const ov::genai::WhisperGenerationConfig& config) {
        return m_speech_encoder.encode(raw_speech, context_tokens, config);
    }

    void process_whisper_logits(ov::Tensor logits,
                            const ov::genai::WhisperGenerationConfig& config,
                            const bool return_timestamps,
                            const std::map<size_t, std::vector<int64_t>>& batch_to_generated_ids) {
        const bool initial_step = batch_to_generated_ids.empty();
        const size_t batch_size = logits.get_shape().at(0);

        for (size_t batch = 0; batch < batch_size; batch++) {
            if (initial_step) {
                ov::genai::do_suppress_tokens(logits, batch, config.begin_suppress_tokens);
            }

            ov::genai::do_suppress_tokens(logits, batch, config.suppress_tokens);

            if (return_timestamps) {
                const auto& generated_ids = initial_step ? std::vector<int64_t>{} : batch_to_generated_ids.at(batch);
                ov::genai::process_whisper_timestamp_logits(logits, batch, config, generated_ids, initial_step);
            }
        }
    }

    int64_t decode_first(ov::Tensor& encoder_hidden_state, ov::Tensor& input_ids) {
        //std::cout << "Decoder first inference call\n";
        // Print input tensor information before setting
        //std::cout << "Setting encoder_hidden_states - Shape: " << encoder_hidden_state.get_shape() 
        //      << ", Type: " << encoder_hidden_state.get_element_type() << std::endl;
        //std::cout << "Setting input_ids - Shape: " << input_ids.get_shape() 
        //      << ", Type: " << input_ids.get_element_type() << std::endl;
        
        // Set inputs
        // Create copies to ensure data persistence
        ov::Tensor encoder_hidden_state_copy(encoder_hidden_state.get_element_type(), encoder_hidden_state.get_shape());
        encoder_hidden_state.copy_to(encoder_hidden_state_copy);
        ov::Tensor input_ids_copy(input_ids.get_element_type(), input_ids.get_shape());
        input_ids.copy_to(input_ids_copy);

        // Set the copied tensors
        m_request_decoder.set_tensor("encoder_hidden_states", encoder_hidden_state_copy);
        m_request_decoder.set_tensor("input_ids", input_ids_copy);

        // Set attention_mask (all ones - no padding)
        std::cout << "Setting attention_mask tensor - Shape: " << input_ids_copy.get_shape() 
              << ", Type: " << ov::element::f32 << std::endl;
        auto input_shape = input_ids_copy.get_shape();
        ov::Tensor attention_mask(ov::element::f32, input_shape);
        std::fill_n(attention_mask.data<float>(), attention_mask.get_size(), 1.0f);
        m_request_decoder.set_tensor("attention_mask", attention_mask);

        // Initialize sequence position
        m_sequence_position = input_shape[1];

        // Run inference
        m_request_decoder.infer();

        // Get outputs and print their shapes and types
        for (const auto& output : m_request_decoder.get_compiled_model().outputs()) {
            ov::Tensor output_tensor = m_request_decoder.get_tensor(output);
        //    std::cout << "Output '" << output.get_any_name() 
        //          << "' - Shape: " << output_tensor.get_shape() 
        //          << ", Type: " << output_tensor.get_element_type() << std::endl;
        }
        ov::Tensor logits = m_request_decoder.get_tensor("logits");
        bool return_timestamps = false;
        process_whisper_logits(logits, m_generation_config, return_timestamps, {});

        // Get the logits tensor and extract the last token's logits
        auto logits_shape = logits.get_shape();
        size_t batch_size = logits_shape[0];
        size_t seq_len = logits_shape[1];
        size_t vocab_size = logits_shape[2];
        
        // Extract logits for the last token in the sequence
        float* logits_data = logits.data<float>();
        std::vector<int64_t> next_tokens;
        
        for (size_t batch = 0; batch < batch_size; ++batch) {
            // Get pointer to the last token's logits for this batch
            size_t last_token_offset = batch * seq_len * vocab_size + (seq_len - 1) * vocab_size;
            float* last_token_logits = logits_data + last_token_offset;
            
            // Find the token with the highest probability
            int64_t best_token = 0;
            float best_score = last_token_logits[0];
            
            for (size_t token = 1; token < vocab_size; ++token) {
                if (last_token_logits[token] > best_score) {
                    best_score = last_token_logits[token];
                    best_token = static_cast<int64_t>(token);
                }
            }
            
            next_tokens.push_back(best_token);
            std::cout << "Batch " << batch << ": Selected token " << best_token 
                  << " with score " << best_score << "; Decoded to: " << m_tokenizer.decode(next_tokens, {ov::genai::skip_special_tokens(false)}) << std::endl;
            initial_step = true;
            return best_token; // early return for single batch now - to be changed
        }
    }

    // For now as we process sequentially, encoder hidden state is already set in infer request so we don't have to pass it
    int64_t decode_next(int64_t last_token_id) {
        ov::InferRequest& source_request = initial_step ? m_request_decoder : m_request_decoder_with_past;
        
        // If initial step, copy encoder_hidden_states from source request
        if (initial_step) {
            if (false) {  // Always false - use direct tensors instead of copies
                ov::Tensor encoder_hidden_states_copy(source_request.get_tensor("encoder_hidden_states").get_element_type(), 
                                                       source_request.get_tensor("encoder_hidden_states").get_shape());
                source_request.get_tensor("encoder_hidden_states").copy_to(encoder_hidden_states_copy);
                m_request_decoder_with_past.set_tensor("encoder_hidden_states", encoder_hidden_states_copy);
            } else {
                // Use direct tensors without copying
                m_request_decoder_with_past.set_tensor("encoder_hidden_states", source_request.get_tensor("encoder_hidden_states"));
            }
        }
        //std::cout << "Decoder next inference call with last_token_id: " << last_token_id << "\n";
        ov::Tensor input_ids{ov::element::i64, {1, 1}}; // single token input
        //std::cout << "Input_ids tensor shape set to {1, 1}\n";
        // Update input_ids tensor with last_token_id
        int64_t* input_ids_data = input_ids.data<int64_t>();
        input_ids_data[0] = last_token_id;
        m_request_decoder_with_past.set_tensor("input_ids", input_ids);
        //std::cout << "Set input_ids tensor - Shape: " << input_ids.get_shape() 
        //          << ", Type: " << input_ids.get_element_type() << std::endl;

        // Increment sequence position for the new token
        m_sequence_position++;

        // Set attention_mask (all ones - no padding) with shape matching KV cache length
        ov::Tensor attention_mask(ov::element::f32, {1, m_sequence_position});
        std::fill_n(attention_mask.data<float>(), attention_mask.get_size(), 1.0f);
        m_request_decoder_with_past.set_tensor("attention_mask", attention_mask);

        // Copy present outputs from previous step to past_key_values inputs for next step
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            // Copy decoder key/value pairs
            std::string decoder_key_present = "present." + std::to_string(layer) + ".decoder.key";
            std::string decoder_value_present = "present." + std::to_string(layer) + ".decoder.value";
            std::string decoder_key_past = "past_key_values." + std::to_string(layer) + ".decoder.key";
            std::string decoder_value_past = "past_key_values." + std::to_string(layer) + ".decoder.value";
            
            if (false) {  // Always false - use direct tensors instead of copies
                ov::Tensor decoder_key_copy(source_request.get_tensor(decoder_key_present).get_element_type(), 
                                            source_request.get_tensor(decoder_key_present).get_shape());
                source_request.get_tensor(decoder_key_present).copy_to(decoder_key_copy);
                m_request_decoder_with_past.set_tensor(decoder_key_past, decoder_key_copy);
                
                ov::Tensor decoder_value_copy(source_request.get_tensor(decoder_value_present).get_element_type(), 
                                              source_request.get_tensor(decoder_value_present).get_shape());
                source_request.get_tensor(decoder_value_present).copy_to(decoder_value_copy);
                m_request_decoder_with_past.set_tensor(decoder_value_past, decoder_value_copy);
            } else {
                // Use direct tensors without copying
                m_request_decoder_with_past.set_tensor(decoder_key_past, source_request.get_tensor(decoder_key_present));
                m_request_decoder_with_past.set_tensor(decoder_value_past, source_request.get_tensor(decoder_value_present));
            }
            
            if (initial_step) {
                // Copy encoder key/value pairs
                std::string encoder_key_present = "present." + std::to_string(layer) + ".encoder.key";
                std::string encoder_value_present = "present." + std::to_string(layer) + ".encoder.value";
                std::string encoder_key_past = "past_key_values." + std::to_string(layer) + ".encoder.key";
                std::string encoder_value_past = "past_key_values." + std::to_string(layer) + ".encoder.value";
                
                if (false) {  // Always false - use direct tensors instead of copies
                    ov::Tensor encoder_key_copy(source_request.get_tensor(encoder_key_present).get_element_type(), 
                                                source_request.get_tensor(encoder_key_present).get_shape());
                    source_request.get_tensor(encoder_key_present).copy_to(encoder_key_copy);
                    m_request_decoder_with_past.set_tensor(encoder_key_past, encoder_key_copy);
                    
                    ov::Tensor encoder_value_copy(source_request.get_tensor(encoder_value_present).get_element_type(), 
                                                  source_request.get_tensor(encoder_value_present).get_shape());
                    source_request.get_tensor(encoder_value_present).copy_to(encoder_value_copy);
                    m_request_decoder_with_past.set_tensor(encoder_value_past, encoder_value_copy);
                } else {
                    // Use direct tensors without copying
                    m_request_decoder_with_past.set_tensor(encoder_key_past, source_request.get_tensor(encoder_key_present));
                    m_request_decoder_with_past.set_tensor(encoder_value_past, source_request.get_tensor(encoder_value_present));
                }
            }
        }

        // Run inference
        //std::cout << "Running inference for decoder with past\n";
        auto infer_start_time = std::chrono::high_resolution_clock::now();
        m_request_decoder_with_past.infer();
        auto infer_end_time = std::chrono::high_resolution_clock::now();
        auto infer_execution_time = std::chrono::duration_cast<std::chrono::microseconds>(infer_end_time - infer_start_time);
        std::cout << "Decoder with past inference time for non-batched mode: " << infer_execution_time.count() << " microseconds" << std::endl;
        initial_step = false;

        /*
        // Print inputs and outputs for decoder_with_past after inference
        std::cout << "=== Decoder With Past After Inference ===" << std::endl;
        std::cout << "Inputs:" << std::endl;
        for (const auto& input : m_request_decoder_with_past.get_compiled_model().inputs()) {
            ov::Tensor input_tensor = m_request_decoder_with_past.get_tensor(input);
            std::cout << "Name: " << input.get_any_name() 
                      << ", Shape: " << input_tensor.get_shape() 
                      << ", Type: " << input_tensor.get_element_type() << std::endl;
        }

        std::cout << "Outputs:" << std::endl;
        for (const auto& output : m_request_decoder_with_past.get_compiled_model().outputs()) {
            ov::Tensor output_tensor = m_request_decoder_with_past.get_tensor(output);
            std::cout << "Name: " << output.get_any_name() 
                      << ", Shape: " << output_tensor.get_shape() 
                      << ", Type: " << output_tensor.get_element_type() << std::endl;
        }
        */
        ov::Tensor logits = m_request_decoder_with_past.get_tensor("logits");
        bool return_timestamps = false;
        // last argument is dummy, only to get into correct conditional branch (no timestampts support yet so should be fine)
        process_whisper_logits(logits, m_generation_config, return_timestamps, {{0,{42}}});

        // Get the logits tensor and extract the last token's logits
        auto logits_shape = logits.get_shape();
        size_t batch_size = logits_shape[0];
        size_t seq_len = logits_shape[1];
        size_t vocab_size = logits_shape[2];
        
        // Extract logits for the last token in the sequence
        float* logits_data = logits.data<float>();
        std::vector<int64_t> next_tokens;
        
        for (size_t batch = 0; batch < batch_size; ++batch) {
            // Get pointer to the last token's logits for this batch
            size_t last_token_offset = batch * seq_len * vocab_size + (seq_len - 1) * vocab_size;
            float* last_token_logits = logits_data + last_token_offset;
            
            // Find the token with the highest probability
            int64_t best_token = 0;
            float best_score = last_token_logits[0];
            
            for (size_t token = 1; token < vocab_size; ++token) {
                if (last_token_logits[token] > best_score) {
                    best_score = last_token_logits[token];
                    best_token = static_cast<int64_t>(token);
                }
            }
            
            next_tokens.push_back(best_token);
            std::cout << "Batch " << batch << ": Selected token " << best_token 
                  << " with score " << best_score << "; Decoded to: " << m_tokenizer.decode(next_tokens, {ov::genai::skip_special_tokens(false)}) << std::endl;
            return best_token; // early return for single batch now - to be changed
        }
    }

    void reset_decoder() {
        initial_step = true;
        m_sequence_position = 0;
        //m_request_decoder = m_request_decoder.get_compiled_model().create_infer_request();
        //m_request_decoder_with_past = m_request_decoder_with_past.get_compiled_model().create_infer_request();
    }

    std::map<size_t, int64_t> batch_decode_first(ov::Tensor& encoder_hidden_state, ov::Tensor& input_ids) {
        std::cout << "Batch Decoder first inference call\n";
        // Set inputs
        m_request_decoder.set_tensor("encoder_hidden_states", encoder_hidden_state);
        m_request_decoder.set_tensor("input_ids", input_ids);

        // Set attention_mask (all ones - no padding)
        auto input_shape = input_ids.get_shape();
        ov::Tensor attention_mask(ov::element::f32, input_shape);
        std::fill_n(attention_mask.data<float>(), attention_mask.get_size(), 1.0f);
        m_request_decoder.set_tensor("attention_mask", attention_mask);

        // Initialize sequence position for batched mode
        m_sequence_position = input_shape[1];

        // Run inference
        m_request_decoder.infer();

        ov::Tensor logits = m_request_decoder.get_tensor("logits");
        bool return_timestamps = false;
        process_whisper_logits(logits, m_generation_config, return_timestamps, {});

        // Get the logits tensor and extract the last token's logits
        auto logits_shape = logits.get_shape();
        size_t batch_size = logits_shape[0];
        size_t seq_len = logits_shape[1];
        size_t vocab_size = logits_shape[2];
        
        // Extract logits for the last token in the sequence
        float* logits_data = logits.data<float>();
        std::map<size_t, int64_t> next_token_map;
        
        for (size_t batch = 0; batch < batch_size; ++batch) {
            // Get pointer to the last token's logits for this batch
            size_t last_token_offset = batch * seq_len * vocab_size + (seq_len - 1) * vocab_size;
            float* last_token_logits = logits_data + last_token_offset;
            
            // Find the token with the highest probability
            int64_t best_token = 0;
            float best_score = last_token_logits[0];
            
            for (size_t token = 1; token < vocab_size; ++token) {
                if (last_token_logits[token] > best_score) {
                    best_score = last_token_logits[token];
                    best_token = static_cast<int64_t>(token);
                }
            }
            
            next_token_map[batch] = best_token;
            std::vector<int64_t> single_token_vec = {best_token};
            //std::cout << "Batch " << batch << ": Selected token " << best_token 
            //      << " with score " << best_score << "; Decoded to: " << m_tokenizer.decode(single_token_vec, {ov::genai::skip_special_tokens(false)}) << std::endl;
        }
        initial_step = true;
        return next_token_map;
    }

        // For now as we process sequentially, encoder hidden state is already set in infer request so we don't have to pass it
    std::map<size_t, int64_t> batch_decode_next(std::map<size_t, int64_t>& last_token_map) {
        std::cout << "Batch Decoder next inference call\n";
        ov::InferRequest& source_request = initial_step ? m_request_decoder : m_request_decoder_with_past;
        
        // If initial step, copy encoder_hidden_states from source request
        if (initial_step) {
            if (false) {  // Always false - use direct tensors instead of copies
                ov::Tensor encoder_hidden_states_copy(source_request.get_tensor("encoder_hidden_states").get_element_type(), 
                                                       source_request.get_tensor("encoder_hidden_states").get_shape());
                source_request.get_tensor("encoder_hidden_states").copy_to(encoder_hidden_states_copy);
                m_request_decoder_with_past.set_tensor("encoder_hidden_states", encoder_hidden_states_copy);
            } else {
                // Use direct tensors without copying
                m_request_decoder_with_past.set_tensor("encoder_hidden_states", source_request.get_tensor("encoder_hidden_states"));
            }
        }
        size_t batch_size = last_token_map.size();
        ov::Tensor input_ids{ov::element::i64, {batch_size, 1}}; // single token input per batch
        int64_t* input_ids_data = input_ids.data<int64_t>();
        for (size_t batch = 0; batch < batch_size; ++batch) {
            int64_t last_token_id = last_token_map.at(batch);
            input_ids_data[batch] = last_token_id;
        }

        m_request_decoder_with_past.set_tensor("input_ids", input_ids);
        //std::cout << "Set input_ids tensor - Shape: " << input_ids.get_shape() 
        //          << ", Type: " << input_ids.get_element_type() << std::endl;

        // Increment sequence position for the new token
        m_sequence_position++;

        // Set attention_mask (all ones - no padding) with shape matching KV cache length
        ov::Tensor attention_mask(ov::element::f32, {batch_size, m_sequence_position});
        std::fill_n(attention_mask.data<float>(), attention_mask.get_size(), 1.0f);
        m_request_decoder_with_past.set_tensor("attention_mask", attention_mask);

        // Copy present outputs from previous step to past_key_values inputs for next step
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            // Copy decoder key/value pairs
            std::string decoder_key_present = "present." + std::to_string(layer) + ".decoder.key";
            std::string decoder_value_present = "present." + std::to_string(layer) + ".decoder.value";
            std::string decoder_key_past = "past_key_values." + std::to_string(layer) + ".decoder.key";
            std::string decoder_value_past = "past_key_values." + std::to_string(layer) + ".decoder.value";
            
            if (false) {  // Always false - use direct tensors instead of copies
                ov::Tensor decoder_key_copy(source_request.get_tensor(decoder_key_present).get_element_type(), 
                                            source_request.get_tensor(decoder_key_present).get_shape());
                source_request.get_tensor(decoder_key_present).copy_to(decoder_key_copy);
                m_request_decoder_with_past.set_tensor(decoder_key_past, decoder_key_copy);
                
                ov::Tensor decoder_value_copy(source_request.get_tensor(decoder_value_present).get_element_type(), 
                                              source_request.get_tensor(decoder_value_present).get_shape());
                source_request.get_tensor(decoder_value_present).copy_to(decoder_value_copy);
                m_request_decoder_with_past.set_tensor(decoder_value_past, decoder_value_copy);
            } else {
                // Use direct tensors without copying
                m_request_decoder_with_past.set_tensor(decoder_key_past, source_request.get_tensor(decoder_key_present));
                m_request_decoder_with_past.set_tensor(decoder_value_past, source_request.get_tensor(decoder_value_present));
            }
            
            if (initial_step) {
                // Copy encoder key/value pairs
                std::string encoder_key_present = "present." + std::to_string(layer) + ".encoder.key";
                std::string encoder_value_present = "present." + std::to_string(layer) + ".encoder.value";
                std::string encoder_key_past = "past_key_values." + std::to_string(layer) + ".encoder.key";
                std::string encoder_value_past = "past_key_values." + std::to_string(layer) + ".encoder.value";
                
                if (false) {  // Always false - use direct tensors instead of copies
                    ov::Tensor encoder_key_copy(source_request.get_tensor(encoder_key_present).get_element_type(), 
                                                source_request.get_tensor(encoder_key_present).get_shape());
                    source_request.get_tensor(encoder_key_present).copy_to(encoder_key_copy);
                    m_request_decoder_with_past.set_tensor(encoder_key_past, encoder_key_copy);
                    
                    ov::Tensor encoder_value_copy(source_request.get_tensor(encoder_value_present).get_element_type(), 
                                                  source_request.get_tensor(encoder_value_present).get_shape());
                    source_request.get_tensor(encoder_value_present).copy_to(encoder_value_copy);
                    m_request_decoder_with_past.set_tensor(encoder_value_past, encoder_value_copy);
                } else {
                    // Use direct tensors without copying
                    m_request_decoder_with_past.set_tensor(encoder_key_past, source_request.get_tensor(encoder_key_present));
                    m_request_decoder_with_past.set_tensor(encoder_value_past, source_request.get_tensor(encoder_value_present));
                }
            }
        }

        // Run inference
        //std::cout << "Running inference for decoder with past\n";
        auto infer_start_time = std::chrono::high_resolution_clock::now();
        m_request_decoder_with_past.infer();
        auto infer_end_time = std::chrono::high_resolution_clock::now();
        auto infer_execution_time = std::chrono::duration_cast<std::chrono::microseconds>(infer_end_time - infer_start_time);
        std::cout << "Decoder with past inference time for batched mode: " << infer_execution_time.count() << " microseconds" << std::endl;
        initial_step = false;

        /*
        // Print inputs and outputs for decoder_with_past after inference
        std::cout << "=== Decoder With Past After Inference ===" << std::endl;
        std::cout << "Inputs:" << std::endl;
        for (const auto& input : m_request_decoder_with_past.get_compiled_model().inputs()) {
            ov::Tensor input_tensor = m_request_decoder_with_past.get_tensor(input);
            std::cout << "Name: " << input.get_any_name() 
                      << ", Shape: " << input_tensor.get_shape() 
                      << ", Type: " << input_tensor.get_element_type() << std::endl;
        }

        std::cout << "Outputs:" << std::endl;
        for (const auto& output : m_request_decoder_with_past.get_compiled_model().outputs()) {
            ov::Tensor output_tensor = m_request_decoder_with_past.get_tensor(output);
            std::cout << "Name: " << output.get_any_name() 
                      << ", Shape: " << output_tensor.get_shape() 
                      << ", Type: " << output_tensor.get_element_type() << std::endl;
        }
        */

        ov::Tensor logits = m_request_decoder_with_past.get_tensor("logits");
        bool return_timestamps = false;
        // last argument is dummy, only to get into correct conditional branch (no timestampts support yet so should be fine)
        process_whisper_logits(logits, m_generation_config, return_timestamps, {{0,{42}}});

        // Get the logits tensor and extract the last token's logits
        auto logits_shape = logits.get_shape();
        batch_size = logits_shape[0];
        size_t seq_len = logits_shape[1];
        size_t vocab_size = logits_shape[2];
        
        // Extract logits for the last token in the sequence
        float* logits_data = logits.data<float>();
        std::map<size_t, int64_t> next_token_map;
        
        for (size_t batch = 0; batch < batch_size; ++batch) {
            // Get pointer to the last token's logits for this batch
            size_t last_token_offset = batch * seq_len * vocab_size + (seq_len - 1) * vocab_size;
            float* last_token_logits = logits_data + last_token_offset;
            
            // Find the token with the highest probability
            int64_t best_token = 0;
            float best_score = last_token_logits[0];
            
            for (size_t token = 1; token < vocab_size; ++token) {
                if (last_token_logits[token] > best_score) {
                    best_score = last_token_logits[token];
                    best_token = static_cast<int64_t>(token);
                }
            }
            
            next_token_map[batch] = best_token;
            std::vector<int64_t> single_token_vec = {best_token};
            //std::cout << "Batch " << batch << ": Selected token " << best_token 
            //      << " with score " << best_score << "; Decoded to: " << m_tokenizer.decode(single_token_vec, {ov::genai::skip_special_tokens(false)}) << std::endl;
        }
        return next_token_map;
    }

    // ========== EXPERIMENTAL: Continuous Batching with BatchManager ==========
    
    void experimental_generate_with_batch_manager(std::vector<std::pair<ov::Tensor, ov::Tensor>>& decoder_inputs,
                                                   size_t max_decode_steps = 10) {
        std::cout << "\n=== EXPERIMENTAL: Using BatchManager for Continuous Batching ===\n";
        
        // Get compiled models
        auto compiled_decoder = m_request_decoder.get_compiled_model();
        auto compiled_decoder_with_past = m_request_decoder_with_past.get_compiled_model();
        
        // Create BatchManager with max batch size
        size_t max_batch_size = decoder_inputs.size();
        BatchManager batch_manager(max_batch_size, 
                                   m_request_decoder, 
                                   m_request_decoder_with_past,
                                   compiled_decoder,
                                   compiled_decoder_with_past);
        
        // Add all requests to BatchManager
        std::vector<size_t> request_ids;
        for (auto& [input_ids, encoder_hidden_state] : decoder_inputs) {
            size_t request_id = batch_manager.reserve_slot_for_request(encoder_hidden_state, input_ids);
            request_ids.push_back(request_id);
            std::cout << "Added request " << request_id << " to batch with following input_ids: ";
            for (size_t i = 0; i < input_ids.get_shape()[1]; ++i) {
                std::cout << input_ids.data<int64_t>()[i] << " ";
            }
            std::cout << "\n";
        }
        
        // Track which requests are still active
        std::map<size_t, std::vector<int64_t>> generated_tokens;
        std::set<size_t> active_requests(request_ids.begin(), request_ids.end());
        
        // Generation loop
        for (size_t step = 0; step < max_decode_steps && !active_requests.empty(); ++step) {
            std::cout << "\nStep " << step << ": Active requests = " << active_requests.size() << "\n";
            
            // Run inference for all active requests
            auto infer_start = std::chrono::high_resolution_clock::now();
            auto new_tokens = batch_manager.run_inference();
            auto infer_end = std::chrono::high_resolution_clock::now();
            auto infer_time = std::chrono::duration_cast<std::chrono::microseconds>(infer_end - infer_start);
            
            auto update_start = std::chrono::high_resolution_clock::now();
            batch_manager.update_requests_after_inference(new_tokens);
            auto update_end = std::chrono::high_resolution_clock::now();
            auto update_time = std::chrono::duration_cast<std::chrono::microseconds>(update_end - update_start);
            
            std::cout << "Inference time: " << infer_time.count() << " μs, Update time: " << update_time.count() << " μs\n";
            
            // Store generated tokens and check for completion
            std::vector<size_t> completed_requests;
            for (const auto& [request_id, token] : new_tokens) {
                generated_tokens[request_id].push_back(token);
                
                // Check for EOS token (50257 for Whisper)
                if (token == m_generation_config.eos_token_id) {
                    std::cout << "Request " << request_id << " completed (EOS token)\n";
                    completed_requests.push_back(request_id);
                }
            }
            
            // Remove completed requests from batch
            for (size_t request_id : completed_requests) {
                batch_manager.release_slot(request_id);
                active_requests.erase(request_id);
            }
            
            // Demonstrate dynamic request addition (optional)
            // Uncomment to test adding new requests mid-generation:
            // if (step == 10 && decoder_inputs.size() > 3) {
            //     auto& [input_ids, encoder_hidden_state] = decoder_inputs[3];
            //     size_t new_request_id = batch_manager.reserve_slot_for_request(encoder_hidden_state, input_ids);
            //     active_requests.insert(new_request_id);
            //     std::cout << "Dynamically added request " << new_request_id << " at step " << step << "\n";
            // }
        }
        
        // Print results
        std::cout << "\n=== Generation Results ===\n";
        for (const auto& [request_id, tokens] : generated_tokens) {
            std::cout << "Request " << request_id << " (" << tokens.size() << " tokens): "
                      << m_tokenizer.decode(tokens, {ov::genai::skip_special_tokens(false)}) << "\n";
        }
        std::cout << "\n=== END EXPERIMENTAL ===\n";
    }
    
    // === ASYNC CONTINUOUS BATCHING METHODS ===
    
    ~WhisperPipelinePocImpl() {
        stop_continuous_batching();
    }
    
    // Generation loop (runs in dedicated thread)
    void generation_loop() {
        std::cout << "Generation thread started\n";
        
        while (m_batch_manager && m_batch_manager->should_continue()) {
            bool stepped = m_batch_manager->step();
            
            if (!stepped) {
                // No active requests, brief sleep
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
        
        std::cout << "Generation thread stopped\n";
    }
    
    // Start continuous batching mode
    void start_continuous_batching(size_t max_batch_size) {
        if (m_generation_running) {
            std::cout << "Continuous batching already running\n";
            return;
        }
        
        std::cout << "Starting continuous batching with max batch size: " << max_batch_size << "\n";
        
        // IMPORTANT: Reset decoder InferRequests to clean state
        // Warmup may have left them with incompatible tensor shapes
        //std::cout << "Resetting decoder InferRequests to clean state...\n";
        
        // Get a clean decoder InferRequest
        auto compiled_decoder = m_request_decoder.get_compiled_model();
        m_request_decoder = compiled_decoder.create_infer_request();
        
        // Get a clean decoder_with_past InferRequest  
        auto compiled_decoder_with_past = m_request_decoder_with_past.get_compiled_model();
        m_request_decoder_with_past = compiled_decoder_with_past.create_infer_request();
        
        //std::cout << "Decoder InferRequests reset complete\n";
        
        m_batch_manager = std::make_unique<BatchManager>(
            max_batch_size, 
            m_request_decoder, 
            m_request_decoder_with_past,
            compiled_decoder,
            compiled_decoder_with_past
        );
        
        // Configure batch manager
        m_batch_manager->set_eos_token_id(m_generation_config.eos_token_id);
        m_batch_manager->set_max_decode_steps(50);  // Can be made configurable
        
        m_generation_running = true;
        m_generation_thread = std::thread(&WhisperPipelinePocImpl::generation_loop, this);
        
        std::cout << "Continuous batching started\n";
    }
    
    // Stop continuous batching
    void stop_continuous_batching() {
        if (!m_generation_running) {
            return;
        }
        
        std::cout << "Stopping continuous batching...\n";
        
        if (m_batch_manager) {
            m_batch_manager->stop();
        }
        
        if (m_generation_thread.joinable()) {
            m_generation_thread.join();
        }
        
        m_generation_running = false;
        m_batch_manager.reset();
        
        std::cout << "Continuous batching stopped\n";
    }
    
    // Add request asynchronously
    std::future<std::vector<int64_t>> add_request_async(
        const RawSpeechInput& audio,
        const WhisperGenerationConfig& config
    ) {
        if (!m_generation_running) {
            throw std::runtime_error("Continuous batching not started. Call start_continuous_batching() first.");
        }
        
        // Encode audio - MUST be protected since encoder is not thread-safe
        ov::Tensor input_ids_copy;
        ov::Tensor encoder_hidden_state_copy;
        
        {
            std::lock_guard<std::mutex> lock(m_encode_mutex);
            
            // Create fresh encoder InferRequest for clean state
            auto fresh_encoder = m_speech_encoder.create_encoder_infer_request();
            
            auto [context_tokens, _] = prepare_context_tokens(config, m_tokenizer);
            
            // Use the encoder's method with fresh InferRequest
            auto encode_start = std::chrono::high_resolution_clock::now();
            auto decoder_inputs = m_speech_encoder.encode_with_infer_request(fresh_encoder, audio, context_tokens, config);
            auto encode_end = std::chrono::high_resolution_clock::now();
            auto encode_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(encode_end - encode_start).count();
            std::cout << "[add_request_async] Encoding took " << encode_duration_ms << " ms\n";
            
            if (decoder_inputs.empty()) {
                throw std::runtime_error("Failed to encode audio");
            }
            
            // Add first chunk (for multi-chunk audio, would need different handling)
            // Make explicit copies to ensure proper ownership
            auto& [input_ids_ref, encoder_hidden_state_ref] = decoder_inputs[0];
            
            input_ids_copy = ov::Tensor(input_ids_ref.get_element_type(), input_ids_ref.get_shape());
            input_ids_ref.copy_to(input_ids_copy);
            
            encoder_hidden_state_copy = ov::Tensor(encoder_hidden_state_ref.get_element_type(), 
                                                   encoder_hidden_state_ref.get_shape());
            encoder_hidden_state_ref.copy_to(encoder_hidden_state_copy);
        }
        // Lock released here - encoding complete
        
        // Add to batch manager (no encoder lock held)
        size_t request_id = m_batch_manager->add_request(encoder_hidden_state_copy, input_ids_copy);
        
        return m_batch_manager->get_result_future(request_id);
    }
    
    // Check if generation thread is running
    bool is_continuous_batching_running() const {
        return m_generation_running;
    }
};

WhisperPipelinePoc::WhisperPipelinePoc(const std::filesystem::path& models_path,
                     const std::string& device,
                     const ov::AnyMap& properties)
    : m_impl(std::make_unique<WhisperPipelinePocImpl>(models_path, device, properties)) {}


void WhisperPipelinePoc::generate(const RawSpeechInput& raw_speech_input,
                  bool batched_mode,
                  OptionalWhisperGenerationConfig generation_config,
                  const std::shared_ptr<StreamerBase> streamer) {
    auto start_time = std::chrono::high_resolution_clock::now();

    WhisperGenerationConfig config = generation_config.has_value() ? *generation_config : m_impl->m_generation_config;
    // Encode raw speech into hidden states for each data frame in order
    auto [context_tokens, tokenization_duration_microseconds] = prepare_context_tokens(config, m_impl->m_tokenizer);
    // Returning vector of (input_ids, encoder_hidden_state) pairs for each chunk
    auto decoder_inputs = m_impl->encode(raw_speech_input, context_tokens, config);
    int broadcast_factor = 1;
    if (const char* env_broadcast = std::getenv("WHISPER_BROADCAST_FACTOR")) {
        try {
            broadcast_factor = std::stoi(env_broadcast);
            if (broadcast_factor < 1) {
                std::cout << "Warning: WHISPER_BROADCAST_FACTOR must be >= 1, using default value 1" << std::endl;
                broadcast_factor = 1;
            }
        } catch (const std::exception& e) {
            std::cout << "Warning: Invalid WHISPER_BROADCAST_FACTOR value, using default value 1" << std::endl;
            broadcast_factor = 1;
        }
    }
    // Expand decoder_inputs by broadcast_factor
    if (broadcast_factor > 1) {
        std::vector<std::pair<ov::Tensor, ov::Tensor>> expanded_decoder_inputs;
        for (const auto& input_pair : decoder_inputs) {
            for (int i = 0; i < broadcast_factor; ++i) {
                // Create copies of the tensors
                ov::Tensor input_ids_copy(input_pair.first.get_element_type(), input_pair.first.get_shape());
                input_pair.first.copy_to(input_ids_copy);
                
                ov::Tensor encoder_hidden_state_copy(input_pair.second.get_element_type(), input_pair.second.get_shape());
                input_pair.second.copy_to(encoder_hidden_state_copy);
                
                expanded_decoder_inputs.emplace_back(std::make_pair(input_ids_copy, encoder_hidden_state_copy));
            }
        }
        decoder_inputs = std::move(expanded_decoder_inputs);
    }
    auto [batched_input_ids, batched_encoder_hidden_state] = m_impl->m_speech_encoder.merge_decoder_inputs(decoder_inputs);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Execution time for speech encoding and batching (common path for now): " << execution_time.count() << " ms" << std::endl;
    // Further processing to be implemented
    if (!batched_mode) {
        auto non_batched_start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<int64_t> generated_ids;
        std::cout << "Running in non-batched mode\n";
        for (auto& [input_ids, encoder_hidden_state] : decoder_inputs) {
            // m_impl->reset_decoder();
            // First decode call on stateful decoder
            auto next_token = m_impl->decode_first(encoder_hidden_state, input_ids);
            generated_ids.push_back(next_token);
            // Subsequent decode calls
            size_t max_decode_steps = 50; // to be replaced with actual stopping criteria
            size_t step = 0;
            while (step < max_decode_steps) {
            if (next_token == /*<|endoftext|>*/ 50257) {
                std::cout << "End of text token generated, stopping decoding.\n";
                break;
            }
            auto decode_next_start_time = std::chrono::high_resolution_clock::now();
            next_token = m_impl->decode_next(next_token);
            generated_ids.push_back(next_token);
            auto decode_next_end_time = std::chrono::high_resolution_clock::now();
            auto decode_next_execution_time = std::chrono::duration_cast<std::chrono::microseconds>(decode_next_end_time - decode_next_start_time);
            std::cout << "Decode next time: " << decode_next_execution_time.count() << " microseconds" << std::endl;
            step++;
            }
            m_impl->reset_decoder();
        }
        std::cout << "\n\n\n Final Decoded Output: " << m_impl->m_tokenizer.decode(generated_ids, {ov::genai::skip_special_tokens(false)}) << std::endl;
        
        auto non_batched_end_time = std::chrono::high_resolution_clock::now();
        auto non_batched_execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(non_batched_end_time - non_batched_start_time);
        std::cout << "Execution time for decoding loop in non-batched mode: " << non_batched_execution_time.count() << " ms" << std::endl;
    } else {
        auto batched_start_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "Running in batched mode\n";
        std::map<size_t, std::vector<int64_t>> generated_ids;
        // Merge decoder inputs into batch tensors
        // auto [batched_input_ids, batched_encoder_hidden_state] = m_impl->m_speech_encoder.merge_decoder_inputs(decoder_inputs);
        std::map<size_t, bool> active_batches;
        size_t active_sequences = decoder_inputs.size();
        for (size_t i = 0; i < decoder_inputs.size(); ++i) {
            active_batches[i] = true;
        }
        // First decode call on stateful decoder - return map <batch_idx, next_token>
        auto next_token_map = m_impl->batch_decode_first(batched_encoder_hidden_state, batched_input_ids);
        // Subsequent decode calls
        size_t max_decode_steps = 50; // to be replaced with actual stopping criteria
        size_t step = 0;
        while (step < max_decode_steps && active_sequences > 0) {
            for (const auto& [batch_idx, next_token] : next_token_map) {
                // Collect generated ids per batch
                if (active_batches[batch_idx]) {
                    generated_ids[batch_idx].push_back(next_token);
                    if (next_token == /*<|endoftext|>*/ 50257) {
                        std::cout << "Batch " << batch_idx << ": End of text token generated, stopping decoding for this batch.\n";
                        active_batches[batch_idx] = false;
                        active_sequences--;
                    }
                }
            }
            if (active_sequences == 0) {
                break;
            }
            auto batch_decode_next_start_time = std::chrono::high_resolution_clock::now();
            next_token_map = m_impl->batch_decode_next(next_token_map);
            auto batch_decode_next_end_time = std::chrono::high_resolution_clock::now();
            auto batch_decode_next_execution_time = std::chrono::duration_cast<std::chrono::microseconds>(batch_decode_next_end_time - batch_decode_next_start_time);
            std::cout << "Batch decode next time: " << batch_decode_next_execution_time.count() << " microseconds" << std::endl;
            step++;
        }

        // Merge all generated ids into single vector for decoding
        std::vector<int64_t> all_generated_ids;
        for (const auto& [batch_idx, ids] : generated_ids) {
            all_generated_ids.insert(all_generated_ids.end(), ids.begin(), ids.end());
        }

        // Print decoded batches
        for (const auto& [batch_idx, ids] : generated_ids) {
            std::cout << "Batch " << batch_idx << ": " << m_impl->m_tokenizer.decode(ids, {ov::genai::skip_special_tokens(false)}) << std::endl;
        }

        //std::cout << "\n\n\n Final Decoded Output: " << m_impl->m_tokenizer.decode(all_generated_ids, {ov::genai::skip_special_tokens(false)}) << std::endl;
        
        auto batched_end_time = std::chrono::high_resolution_clock::now();
        auto batched_execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(batched_end_time - batched_start_time);
        std::cout << "Execution time for decoding loop in batched mode: " << batched_execution_time.count() << " ms" << std::endl;
    }
}

// ========== EXPERIMENTAL: Continuous Batching API ==========

void WhisperPipelinePoc::experimental_generate_with_continuous_batching(
    const RawSpeechInput& raw_speech_input,
    OptionalWhisperGenerationConfig generation_config,
    const std::shared_ptr<StreamerBase> streamer) {
    
    std::cout << "\n========================================\n";
    std::cout << "EXPERIMENTAL: Continuous Batching Mode\n";
    std::cout << "========================================\n\n";
    
    WhisperGenerationConfig config = generation_config.has_value() 
        ? *generation_config 
        : m_impl->m_generation_config;
    
    // Encode speech into decoder inputs
    auto [context_tokens, tokenization_duration] = prepare_context_tokens(
        config, m_impl->m_tokenizer);
    auto decoder_inputs = m_impl->encode(raw_speech_input, context_tokens, config);
    
    std::cout << "Encoded " << decoder_inputs.size() << " audio chunks\n";
    
    // Optional: Broadcast factor for testing
    int broadcast_factor = 1;
    if (const char* env_broadcast = std::getenv("WHISPER_BROADCAST_FACTOR")) {
        try {
            broadcast_factor = std::stoi(env_broadcast);
            if (broadcast_factor < 1) broadcast_factor = 1;
        } catch (...) {
            broadcast_factor = 1;
        }
    }
    
    if (broadcast_factor > 1) {
        std::cout << "Broadcasting inputs by factor " << broadcast_factor << "\n";
        std::vector<std::pair<ov::Tensor, ov::Tensor>> expanded_inputs;
        for (const auto& input_pair : decoder_inputs) {
            for (int i = 0; i < broadcast_factor; ++i) {
                ov::Tensor input_ids_copy(input_pair.first.get_element_type(), 
                                         input_pair.first.get_shape());
                input_pair.first.copy_to(input_ids_copy);
                
                ov::Tensor encoder_hidden_state_copy(input_pair.second.get_element_type(), 
                                                     input_pair.second.get_shape());
                input_pair.second.copy_to(encoder_hidden_state_copy);
                
                expanded_inputs.emplace_back(input_ids_copy, encoder_hidden_state_copy);
            }
        }
        decoder_inputs = std::move(expanded_inputs);
    }
    
    // Use BatchManager for continuous batching
    auto start_time = std::chrono::high_resolution_clock::now();
    m_impl->experimental_generate_with_batch_manager(decoder_inputs, 50);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\nTotal continuous batching time: " << total_time.count() << " ms\n";
}

// === ASYNC CONTINUOUS BATCHING API ===

void WhisperPipelinePoc::start_continuous_batching(size_t max_batch_size) {
    m_impl->start_continuous_batching(max_batch_size);
}

void WhisperPipelinePoc::stop_continuous_batching() {
    m_impl->stop_continuous_batching();
}

std::future<std::vector<int64_t>> WhisperPipelinePoc::generate_async(
    const RawSpeechInput& audio,
    OptionalWhisperGenerationConfig config
) {
    WhisperGenerationConfig gen_config = config.has_value() 
        ? *config 
        : m_impl->m_generation_config;
    
    return m_impl->add_request_async(audio, gen_config);
}

bool WhisperPipelinePoc::is_continuous_batching_running() const {
    return m_impl->is_continuous_batching_running();
}

WhisperPipelinePoc::~WhisperPipelinePoc() = default;

}  // namespace genai
}  // namespace ov
