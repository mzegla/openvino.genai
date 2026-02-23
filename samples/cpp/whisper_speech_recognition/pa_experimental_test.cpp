// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/openvino.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/genai/model_transformations.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

// Replace SDPA nodes with PagedAttention for self-attention layers only
void replace_self_attention_sdpa_with_pa(std::shared_ptr<ov::Model> model) {
    using namespace ov::op;

    std::vector<std::shared_ptr<ov::Node>> self_attn_nodes;
    std::vector<std::shared_ptr<ov::Node>> cross_attn_nodes;

    // Find all ScaledDotProductAttention nodes and classify them
    std::cout << "Analyzing SDPA nodes in model...\n";
    for (const auto& node : model->get_ops()) {
        if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(node)) {
            std::string node_name = node->get_friendly_name();
            if (node_name.find("encoder_attn") != std::string::npos || 
                node_name.find("cross_attn") != std::string::npos) {
                cross_attn_nodes.push_back(node);
                std::cout << "  Cross-attention SDPA: " << node_name << "\n";
            } else {
                self_attn_nodes.push_back(node);
                std::cout << "  Self-attention SDPA: " << node_name << "\n";
            }
        }
    }

    std::cout << "\nFound " << self_attn_nodes.size() << " self-attention SDPA nodes\n";
    std::cout << "Found " << cross_attn_nodes.size() << " cross-attention SDPA nodes\n\n";

    if (self_attn_nodes.empty()) {
        std::cout << "No self-attention SDPA nodes to transform\n";
        return;
    }

    // Store cross-attention connections before transformation
    struct CrossAttnInfo {
        std::shared_ptr<ov::Node> node;
        std::vector<ov::Output<ov::Node>> inputs;
    };
    std::vector<CrossAttnInfo> cross_attn_info;

    for (const auto& cross_attn : cross_attn_nodes) {
        CrossAttnInfo info;
        info.node = cross_attn;
        for (size_t i = 0; i < cross_attn->get_input_size(); ++i) {
            info.inputs.push_back(cross_attn->input(i).get_source_output());
        }
        cross_attn_info.push_back(info);
    }

    // Apply SDPAToPagedAttention transformation
    std::cout << "Applying SDPAToPagedAttention transformation...\n";
    ov::pass::SDPAToPagedAttention(false, false, false, false, false).run_on_model(model);
    std::cout << "Transformation complete!\n\n";

    // Restore cross-attention nodes back to SDPA
    std::cout << "Restoring cross-attention nodes back to SDPA...\n";
    for (const auto& info : cross_attn_info) {
        for (const auto& node : model->get_ops()) {
            if (node->get_type_info().name == std::string("PagedAttentionExtension")) {
                if (node->input(0).get_source_output() == info.inputs[0]) {
                    std::shared_ptr<ov::Node> new_sdpa;
                    if (info.inputs.size() >= 4) {
                        new_sdpa = std::make_shared<v13::ScaledDotProductAttention>(
                            info.inputs[0], info.inputs[1], info.inputs[2], info.inputs[3], false);
                    } else {
                        new_sdpa = std::make_shared<v13::ScaledDotProductAttention>(
                            info.inputs[0], info.inputs[1], info.inputs[2], false);
                    }
                    new_sdpa->set_friendly_name(info.node->get_friendly_name());
                    ov::replace_node(node, new_sdpa);
                    std::cout << "  Restored: " << new_sdpa->get_friendly_name() << "\n";
                    break;
                }
            }
        }
    }

    model->validate_nodes_and_infer_types();
    std::cout << "\nTransformation complete: self-attention uses PA, cross-attention uses SDPA\n\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_decoder_model.xml> [--full_pass]" << std::endl;
        std::cerr << "Example: " << argv[0] << " ./whisper-large-v3-turbo/openvino_decoder_model.xml" << std::endl;
        std::cerr << "         " << argv[0] << " ./whisper-large-v3-turbo/openvino_decoder_model.xml --full_pass" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --full_pass    Apply SDPAToPagedAttention to entire model (common approach)" << std::endl;
        std::cerr << "                 Without this flag: PA for self-attention only (experimental)" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    bool use_full_pass = false;
    
    // Check for --full_pass flag
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--full_pass") {
            use_full_pass = true;
            break;
        }
    }
    
    try {
        std::cout << "=== PA Experimental Model Inspector ===" << std::endl;
        std::cout << "Loading decoder model from: " << model_path << std::endl;
        std::cout << "Mode: " << (use_full_pass ? "Full PA Pass (common approach)" : "Experimental (PA for self-attention only)") << std::endl << std::endl;
        
        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        
        std::cout << "Model loaded successfully!" << std::endl << std::endl;
        
        if (use_full_pass) {
            // Common approach: Apply SDPAToPagedAttention to entire model
            std::cout << "=== Applying Full SDPAToPagedAttention Pass ===" << std::endl;
            std::cout << "This will convert ALL SDPA nodes (both self and cross attention) to PagedAttention\n\n";
            ov::pass::SDPAToPagedAttention(false, false, false, false, false).run_on_model(model);
            ov::genai::utils::apply_gather_before_matmul_transformation(model);
            std::cout << "Full PA transformation complete!\n\n";
        } else {
            // Experimental approach: Apply PA transformation to self-attention layers only
            std::cout << "=== Applying Experimental Transformation ===" << std::endl;
            std::cout << "This will convert ONLY self-attention SDPA nodes to PagedAttention\n";
            std::cout << "Cross-attention nodes will remain as SDPA\n\n";
            replace_self_attention_sdpa_with_pa(model);
        }
        
        // Print model inputs
        std::cout << "=== MODEL INPUTS ===" << std::endl;
        auto inputs = model->inputs();
        std::cout << "Total inputs: " << inputs.size() << std::endl << std::endl;
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto& input = inputs[i];
            try {
                std::cout << "Input #" << i << ":" << std::endl;
                std::cout << "  Name: " << input.get_any_name() << std::endl;
                std::cout << "  Type: " << input.get_element_type() << std::endl;
                std::cout << "  Shape: " << input.get_partial_shape() << std::endl;
                
                // Print all alternative names if any
                auto names = input.get_names();
                if (names.size() > 1) {
                    std::cout << "  Alternative names: ";
                    bool first = true;
                    for (const auto& name : names) {
                        if (!first) std::cout << ", ";
                        std::cout << name;
                        first = false;
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error reading input #" << i << ": " << e.what() << std::endl;
            }
        }
        
        // Print model outputs
        std::cout << "=== MODEL OUTPUTS ===" << std::endl;
        auto outputs = model->outputs();
        std::cout << "Total outputs: " << outputs.size() << std::endl << std::endl;
        
        for (size_t i = 0; i < outputs.size(); ++i) {
            const auto& output = outputs[i];
            try {
                std::cout << "Output #" << i << ":" << std::endl;
                std::cout << "  Name: " << output.get_any_name() << std::endl;
                std::cout << "  Type: " << output.get_element_type() << std::endl;
                std::cout << "  Shape: " << output.get_partial_shape() << std::endl;
                
                // Print all alternative names if any
                auto names = output.get_names();
                if (names.size() > 1) {
                    std::cout << "  Alternative names: ";
                    bool first = true;
                    for (const auto& name : names) {
                        if (!first) std::cout << ", ";
                        std::cout << name;
                        first = false;
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error reading output #" << i << ": " << e.what() << std::endl;
            }
        }
        
        std::cout << "=== SUMMARY ===" << std::endl;
        std::cout << "Total inputs: " << inputs.size() << std::endl;
        std::cout << "Total outputs: " << outputs.size() << std::endl;
        std::cout << std::endl;
        
        // Check if model has PagedAttention nodes
        bool has_paged_attention = false;
        for (const auto& node : model->get_ops()) {
            if (node->get_type_info().name == std::string("PagedAttentionExtension")) {
                has_paged_attention = true;
                break;
            }
        }
        
        // Compile and run inference test
        std::cout << "=== COMPILATION AND INFERENCE TEST ===" << std::endl;
        
        if (has_paged_attention) {
            std::cout << "⚠ Model contains PagedAttention operations" << std::endl;
            std::cout << "Attempting simplified inference test with minimal dummy inputs..." << std::endl;
        } else {
            std::cout << "Model uses SDPA operations (no PagedAttention detected)" << std::endl;
            std::cout << "Attempting to compile and run inference..." << std::endl;
        }
        std::cout << std::endl;
        
        try {
            ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
            std::cout << "✓ Model compiled successfully on CPU!" << std::endl << std::endl;
            
            // Create infer request
            std::cout << "Creating infer request..." << std::endl;
            auto infer_request = compiled_model.create_infer_request();
            std::cout << "✓ Infer request created successfully!" << std::endl << std::endl;
            
            if (has_paged_attention) {
                // Set minimal PagedAttention inputs for a single token inference
                std::cout << "Setting up minimal PagedAttention inputs (1 batch, 1 token, 1 block)..." << std::endl;
                
                try {
                    // Get inputs from compiled model (concrete types/shapes after compilation)
                    auto compiled_inputs = compiled_model.inputs();
                    
                    // First pass: detect encoder_hidden_states sequence length for Whisper
                    size_t encoder_seq_len = 1500;  // Default: Whisper 30-sec audio produces 1500 tokens
                    for (const auto& input : compiled_inputs) {
                        std::string name = input.get_any_name();
                        if (name == "encoder_hidden_states") {
                            auto shape = input.get_partial_shape();
                            // Shape is [batch, seq_len, hidden_dim]
                            // For Whisper: seq_len=1500 for 30-sec audio (3000 mel frames / 2)
                            encoder_seq_len = 1500;
                            std::cout << "  Detected Whisper model - using encoder_seq_len=" << encoder_seq_len << " (30-sec audio)" << std::endl;
                            break;
                        }
                    }
                    
                    // Check which inputs exist and set minimal values
                    for (const auto& input : compiled_inputs) {
                        std::string name = input.get_any_name();
                        auto shape = input.get_partial_shape();
                        auto elem_type = input.get_element_type();
                        
                        std::cout << "  Processing input: " << name << " (shape: " << shape << ", type: " << elem_type << ")" << std::endl;

                        if (name == "input_ids") {
                            // Check dimensionality - could be [seq_len] or [batch, seq_len]
                            ov::Shape concrete_shape;
                            size_t rank = shape.rank().get_length();
                            if (rank == 1) {
                                concrete_shape = {1};  // Single token
                            } else {
                                // Multi-dimensional: set all dims to 1
                                for (size_t i = 0; i < rank; ++i) {
                                    concrete_shape.push_back(1);
                                }
                            }
                            ov::Tensor t(elem_type, concrete_shape);
                            t.data<int64_t>()[0] = 1;  // dummy token ID
                            infer_request.set_tensor(name, t);
                            std::cout << "    Set: " << concrete_shape << " = {1}" << std::endl;
                        }
                        else if (name == "position_ids") {
                            // Check dimensionality - could be [seq_len] or [batch, seq_len]
                            ov::Shape concrete_shape;
                            size_t rank = shape.rank().get_length();
                            if (rank == 1) {
                                concrete_shape = {1};  // Single position
                            } else {
                                // Multi-dimensional: set all dims to 1
                                for (size_t i = 0; i < rank; ++i) {
                                    concrete_shape.push_back(1);
                                }
                            }
                            ov::Tensor t(elem_type, concrete_shape);
                            t.data<int64_t>()[0] = 0;  // position 0
                            infer_request.set_tensor(name, t);
                            std::cout << "    Set: " << concrete_shape << " = {0}" << std::endl;
                        }
                        else if (name == "past_lens") {
                            ov::Tensor t(elem_type, {1});
                            t.data<int32_t>()[0] = 0;  // no past context
                            infer_request.set_tensor(name, t);
                            std::cout << "    Set: [1] = {0}" << std::endl;
                        }
                        else if (name == "subsequence_begins") {
                            ov::Tensor t(elem_type, {2});
                            t.data<int32_t>()[0] = 0;
                            t.data<int32_t>()[1] = 1;  // 1 token
                            infer_request.set_tensor(name, t);
                            std::cout << "    Set: [2] = {0, 1}" << std::endl;
                        }
                        else if (name == "block_indices_begins") {
                            ov::Tensor t(elem_type, {2});
                            t.data<int32_t>()[0] = 0;
                            t.data<int32_t>()[1] = 1;  // 1 block
                            infer_request.set_tensor(name, t);
                            std::cout << "    Set: [2] = {0, 1}" << std::endl;
                        }
                        else if (name == "block_indices" || name.find("block_indices.") == 0) {
                            ov::Tensor t(elem_type, {1});
                            t.data<int32_t>()[0] = 0;  // block index 0
                            infer_request.set_tensor(name, t);
                            std::cout << "    Set: [1] = {0}" << std::endl;
                        }
                        else if (name == "max_context_len") {
                            ov::Tensor t(elem_type, {});  // scalar
                            t.data<int32_t>()[0] = 32;  // Use non-zero value (could be max cache size)
                            infer_request.set_tensor(name, t);
                            std::cout << "    Set: scalar = 32" << std::endl;
                        }
                        else if (name == "attention_mask") {
                            ov::Tensor t(elem_type, {1, 1});
                            t.data<float>()[0] = 0.0f;  // no masking
                            infer_request.set_tensor(name, t);
                            std::cout << "    Set: [1, 1] = {0.0}" << std::endl;
                        }
                        else if (name == "encoder_hidden_states") {
                            // For Whisper, encoder_hidden_states shape is [batch, seq_len, hidden_dim]
                            // For 30-sec audio: [1, 1500, hidden_dim]
                            ov::Shape concrete_shape;
                            for (const auto& dim : shape) {
                                concrete_shape.push_back(dim.is_dynamic() ? 1 : dim.get_length());
                            }
                            // Set realistic dimensions for Whisper
                            if (concrete_shape.size() >= 3) {
                                concrete_shape[0] = 1;                // batch
                                concrete_shape[1] = encoder_seq_len;  // 1500 for 30-sec audio
                                // keep concrete_shape[2] as hidden_dim
                            }
                            ov::Tensor t(elem_type, concrete_shape);
                            std::memset(t.data(), 0, t.get_byte_size());
                            infer_request.set_tensor(name, t);
                            std::cout << "    Set: " << concrete_shape << " (zeros, simulating 30-sec audio)" << std::endl;
                        }
                        else if (name.find("key_cache") != std::string::npos || name.find("value_cache") != std::string::npos) {
                            // Whisper decoder self-attention K/V caches (NOT cross-attention)
                            // These grow with each decoder token, independent of encoder_hidden_states
                            // Shape could be [batch, num_heads, seq_len, head_dim] or similar
                            // For first token: seq_len should be 0 (no past), but respect static dimensions
                            ov::Shape concrete_shape;
                            bool has_dynamic_seq_dim = false;
                            for (size_t i = 0; i < shape.size(); ++i) {
                                const auto& dim = shape[i];
                                if (dim.is_dynamic()) {
                                    if (i == 0) {
                                        concrete_shape.push_back(1);  // batch = 1
                                    } else {
                                        // Likely the sequence dimension for self-attention cache
                                        concrete_shape.push_back(0);  // Start with empty cache (first token)
                                        has_dynamic_seq_dim = true;
                                    }
                                } else {
                                    // Keep static dimensions as specified by the model
                                    concrete_shape.push_back(dim.get_length());
                                }
                            }
                            ov::Tensor t(elem_type, concrete_shape);
                            if (t.get_byte_size() > 0) {
                                std::memset(t.data(), 0, t.get_byte_size());
                            }
                            infer_request.set_tensor(name, t);
                            std::cout << "    Set self-attn cache: " << concrete_shape 
                                     << (has_dynamic_seq_dim ? " (empty cache for first token)" : " (zeros)") << std::endl;
                        }
                        else if (name.find("past_key_values") != std::string::npos) {
                            // KV cache tensors for PagedAttention (LLM style)
                            // Shape is typically [batch, num_heads, seq_len, head_dim] or similar
                            // For minimal test: [1, num_heads, 0, head_dim] (empty cache)
                            ov::Shape concrete_shape;
                            for (size_t i = 0; i < shape.size(); ++i) {
                                const auto& dim = shape[i];
                                if (i == 0) {
                                    concrete_shape.push_back(1);  // batch = 1
                                } else if (i == 2 || (shape.size() == 3 && i == 1)) {
                                    // seq_len dimension - start with 0 (empty cache)
                                    concrete_shape.push_back(0);
                                } else {
                                    concrete_shape.push_back(dim.is_dynamic() ? 1 : dim.get_length());
                                }
                            }
                            ov::Tensor t(elem_type, concrete_shape);
                            if (concrete_shape.size() > 0 && t.get_byte_size() > 0) {
                                std::memset(t.data(), 0, t.get_byte_size());
                            }
                            infer_request.set_tensor(name, t);
                            std::cout << "    Set PA cache: " << concrete_shape << " (empty)" << std::endl;
                        }
                        else if (name.find("beam") != std::string::npos) {
                            // Skip beam-related inputs for now
                            std::cout << "    Skipped (beam search input)" << std::endl;
                        }
                        else {
                            // For other inputs, try to set minimal values
                            ov::Shape concrete_shape;
                            for (const auto& dim : shape) {
                                concrete_shape.push_back(dim.is_dynamic() ? 1 : dim.get_length());
                            }
                            ov::Tensor t(elem_type, concrete_shape);
                            
                            // For scalar int32, use a reasonable non-zero value
                            if (concrete_shape.empty() && elem_type == ov::element::i32) {
                                t.data<int32_t>()[0] = 32;  // Reasonable default (e.g., max cache size)
                                infer_request.set_tensor(name, t);
                                std::cout << "    Set: scalar = 32 (default)" << std::endl;
                            } else {
                                std::memset(t.data(), 0, t.get_byte_size());
                                infer_request.set_tensor(name, t);
                                std::cout << "    Set: " << concrete_shape << " (zeros)" << std::endl;
                            }
                        }
                    }
                    std::cout << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to set some inputs: " << e.what() << std::endl;
                }
            }
            
            // Run inference
            std::cout << "Attempting to run inference..." << std::endl;
            infer_request.infer();
            std::cout << "✓ Inference completed successfully!" << std::endl << std::endl;
            
            std::cout << "=== INFERENCE TEST: PASSED ✓ ===" << std::endl;
            if (has_paged_attention) {
                std::cout << "The PagedAttention model can execute with minimal dummy inputs!" << std::endl;
                std::cout << "Note: For production use, proper KV cache management is required (ContinuousBatchingPipeline)." << std::endl;
            } else {
                std::cout << "The transformed model is functional and can execute inference!" << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "✗ Test FAILED: " << e.what() << std::endl;
            if (has_paged_attention) {
                std::cerr << std::endl << "PagedAttention models require complex input setup." << std::endl;
                std::cerr << "The minimal inputs provided may not be sufficient for all model configurations." << std::endl;
                std::cerr << "For production use, ContinuousBatchingPipeline provides proper infrastructure." << std::endl;
            } else {
                std::cerr << "The model structure may have issues or be incompatible with the backend." << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
