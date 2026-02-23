// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "statefull_decoder.hpp"

#include "openvino/op/softmax.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "utils.hpp"
#include "whisper/alignment_heads.hpp"
#include "whisper/word_level_timestamps.hpp"
#include <iostream>

namespace {
void reshape_hidden_states_to_static(std::shared_ptr<ov::Model> model, const ov::PartialShape& lhstates_shape) {
    ov::PartialShape new_shape = model->input("encoder_hidden_states").get_partial_shape();
    OPENVINO_ASSERT(new_shape.size() > 1 && lhstates_shape.size() > 1);
    new_shape[1] = lhstates_shape[1];
    std::map<std::string, ov::PartialShape> name_to_shape{{"encoder_hidden_states", new_shape}};
    model->reshape(name_to_shape);
}

// Replace SDPA nodes with PagedAttention for self-attention layers only
// Cross-attention layers (encoder_attn) are kept as SDPA
void replace_self_attention_sdpa_with_pa(std::shared_ptr<ov::Model> model) {
    using namespace ov::op;

    std::vector<std::shared_ptr<ov::Node>> self_attn_nodes;
    std::vector<std::shared_ptr<ov::Node>> cross_attn_nodes;

    // Find all ScaledDotProductAttention nodes and classify them
    std::cout << "Analyzing SDPA nodes in model...\n";
    for (const auto& node : model->get_ops()) {
        if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(node)) {
            // Use node name to distinguish: cross-attention has "encoder_attn" in name
            std::string node_name = node->get_friendly_name();
            if (node_name.find("encoder_attn") != std::string::npos || 
                node_name.find("cross_attn") != std::string::npos) {
                // Cross-attention node - keep as SDPA
                cross_attn_nodes.push_back(node);
                std::cout << "  Found cross-attention SDPA node: " << node_name << "\n";
            } else {
                // Self-attention node - will be replaced with PA
                self_attn_nodes.push_back(node);
                std::cout << "  Found self-attention SDPA node: " << node_name << "\n";
            }
        }
    }

    std::cout << "Found " << self_attn_nodes.size() << " self-attention SDPA nodes\n";
    std::cout << "Found " << cross_attn_nodes.size() << " cross-attention SDPA nodes\n";

    if (self_attn_nodes.empty()) {
        std::cout << "No self-attention SDPA nodes found to replace\n";
        return;
    }

    // Store cross-attention connections before transformation
    struct CrossAttnInfo {
        std::shared_ptr<ov::Node> node;
        std::vector<ov::Output<ov::Node>> inputs;
        std::vector<std::shared_ptr<ov::Node>> output_targets;
        std::vector<size_t> output_ports;
    };
    std::vector<CrossAttnInfo> cross_attn_info;

    for (const auto& cross_attn : cross_attn_nodes) {
        CrossAttnInfo info;
        info.node = cross_attn;
        
        // Store input connections
        for (size_t i = 0; i < cross_attn->get_input_size(); ++i) {
            info.inputs.push_back(cross_attn->input(i).get_source_output());
        }
        
        // Store output connections
        for (const auto& output : cross_attn->output(0).get_target_inputs()) {
            info.output_targets.push_back(output.get_node()->shared_from_this());
            info.output_ports.push_back(output.get_index());
        }
        
        cross_attn_info.push_back(info);
        std::cout << "  Stored connections for: " << cross_attn->get_friendly_name() << "\n";
    }

    // Apply SDPAToPagedAttention pass to the entire model
    // This will replace ALL SDPA nodes (both self and cross attention)
    std::cout << "Applying SDPAToPagedAttention transformation...\n";
    bool use_per_layer_block_indices = false;
    bool use_score_outputs = false;
    bool allow_score_aggregation = false;
    bool allow_cache_rotation = false;
    bool allow_xattention = false;
    ov::pass::SDPAToPagedAttention(use_per_layer_block_indices, 
                                    use_score_outputs, 
                                    allow_score_aggregation, 
                                    allow_cache_rotation, 
                                    allow_xattention).run_on_model(model);
    std::cout << "SDPAToPagedAttention transformation completed\n";

    // Now restore cross-attention nodes back to SDPA
    std::cout << "Restoring cross-attention nodes back to SDPA...\n";
    for (const auto& info : cross_attn_info) {
        // Find the PagedAttention node that replaced this SDPA node
        // We need to locate it by comparing input connections
        std::shared_ptr<ov::Node> pa_node = nullptr;
        
        for (const auto& node : model->get_ops()) {
            // PagedAttention nodes have specific type name
            if (node->get_type_info().name == std::string("PagedAttentionExtension")) {
                // Check if this PA node has the same Q input as our stored cross-attn SDPA
                if (node->input(0).get_source_output() == info.inputs[0]) {
                    pa_node = node;
                    break;
                }
            }
        }

        if (pa_node) {
            std::cout << "  Found PagedAttention node to revert: " << pa_node->get_friendly_name() << "\n";
            
            // Create new SDPA node with original connections
            // SDPA inputs: Q, K, V, attention_mask (optional), scale (optional)
            std::shared_ptr<ov::Node> new_sdpa;
            if (info.inputs.size() >= 4) {
                new_sdpa = std::make_shared<v13::ScaledDotProductAttention>(
                    info.inputs[0],  // Q
                    info.inputs[1],  // K  
                    info.inputs[2],  // V
                    info.inputs[3],  // attention_mask
                    false  // causal
                );
            } else {
                new_sdpa = std::make_shared<v13::ScaledDotProductAttention>(
                    info.inputs[0],  // Q
                    info.inputs[1],  // K
                    info.inputs[2],  // V
                    false  // causal
                );
            }
            new_sdpa->set_friendly_name(info.node->get_friendly_name());
            
            // Replace the PA node with the new SDPA node
            ov::replace_node(pa_node, new_sdpa);
            std::cout << "  Restored SDPA node: " << new_sdpa->get_friendly_name() << "\n";
        } else {
            std::cout << "  Warning: Could not find PagedAttention node for cross-attention: " 
                      << info.node->get_friendly_name() << "\n";
        }
    }

    model->validate_nodes_and_infer_types();
    std::cout << "Model transformation completed: self-attention uses PA, cross-attention uses SDPA\n";
}
}  // namespace

namespace ov::genai {
WhisperStatefullDecoder::WhisperStatefullDecoder(const std::filesystem::path& models_path,
                                                 const std::string& device,
                                                 const ov::AnyMap& properties,
                                                 const ov::PartialShape& lhs_shape,
                                                 const bool decompose_cross_attention_spda)
    : m_decompose_cross_attention_spda_ops(decompose_cross_attention_spda) {
    ov::Core core = utils::singleton_core();

    auto model = core.read_model(models_path / "openvino_decoder_model.xml", {}, properties);

    if (m_decompose_cross_attention_spda_ops) {
        ov::genai::decompose_scaled_dot_product_attention_for_whisper(model);
        ov::genai::add_cross_attention_qk_scaled_scores_outputs_for_whisper(model);
    }

    // Create model2: Read the model again and replace SDPA with PA for self-attention only
    std::cout << "\n=== Creating model2 with PagedAttention for self-attention layers ===\n";
    m_model_with_pa = core.read_model(models_path / "openvino_decoder_model.xml", {}, properties);
    
    // Apply the transformation to replace self-attention SDPA with PA
    replace_self_attention_sdpa_with_pa(m_model_with_pa);
    
    // Note: We cannot save models with PagedAttentionExtension operations as they cannot be 
    // easily loaded back (requires custom extension registration at runtime).
    // The transformed model is available via get_model_with_paged_attention() for runtime use.
    std::cout << "=== model2 creation completed and stored in m_model_with_pa ===\n";
    std::cout << "Note: Model with PA operations is available at runtime but not saved to disk.\n\n";

    m_has_cache_position = utils::has_input(model, "cache_position");

    ov::CompiledModel compiled_model;
    if (device == "NPU") {
        auto kv_pos = ov::genai::utils::get_kv_axes_pos(model);

        reshape_hidden_states_to_static(model, lhs_shape);

        utils::KVDesc kv_desc;
        std::tie(compiled_model, kv_desc) = utils::compile_decoder_for_npu(model, properties, kv_pos, true);
    } else {
        utils::apply_slice_before_matmul_transformation(model);

        compiled_model = core.compile_model(model, device, properties);
    }

    utils::print_compiled_model_properties(compiled_model, "whisper decoder model");
    m_request = compiled_model.create_infer_request();
}

void WhisperStatefullDecoder::start_async(const Tensor& encoder_hidden_state,
                                          const Tensor& input_ids,
                                          const Tensor& beam_idx) {
    const size_t batch_size = input_ids.get_shape().at(0);
    const size_t seq_len = input_ids.get_shape().at(1);

    _set_encoder_hidden_states_tensor(encoder_hidden_state, batch_size, m_request);

    if (m_has_cache_position) {
        _set_cache_position_tensor(seq_len);
    }
    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("beam_idx", beam_idx);

    m_request.start_async();
};

void WhisperStatefullDecoder::_set_cache_position_tensor(const size_t seq_len) {
    ov::Tensor cache_position_tensor = m_request.get_tensor("cache_position");

    int64_t start_cache_position = 0;

    if (cache_position_tensor.get_size() != 0) {
        start_cache_position = cache_position_tensor.data<int64_t>()[cache_position_tensor.get_size() - 1] + 1;
    }

    cache_position_tensor.set_shape({seq_len});

    auto cache_data = cache_position_tensor.data<int64_t>();
    std::iota(cache_data, cache_data + seq_len, start_cache_position);
};

Tensor WhisperStatefullDecoder::wait() {
    m_request.wait();
    return m_request.get_tensor("logits");
}

void WhisperStatefullDecoder::reset_state() {
    m_request.reset_state();
    if (m_has_cache_position) {
        m_request.set_tensor("cache_position", create_host_tensor(ov::element::i64, {0}));
    }

    Shape encoder_hidden_states_shape{m_request.get_tensor("encoder_hidden_states").get_shape()};
    encoder_hidden_states_shape[0] = 0;
    m_request.set_tensor("encoder_hidden_states", create_host_tensor(ov::element::f32, encoder_hidden_states_shape));
};

ov::Tensor WhisperStatefullDecoder::create_host_tensor(const element::Type element_type, const Shape& shape) {
    try {
        return m_request.get_compiled_model().get_context().create_host_tensor(element_type, shape);
    } catch (std::exception& ex) {
        return ov::Tensor(element_type, shape);
    }
}

std::vector<Tensor> WhisperStatefullDecoder::get_alignments_heads_qks(
    const std::vector<std::pair<size_t, size_t>>& alignment_heads) {
    OPENVINO_ASSERT(m_decompose_cross_attention_spda_ops,
                    "Encoder attention heads are not decomposed. Cannot get encoder attention QKs.");

    return ov::genai::get_whisper_alignments_heads_qks(m_request, alignment_heads);
}

}  // namespace ov::genai
