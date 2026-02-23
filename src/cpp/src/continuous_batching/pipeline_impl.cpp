// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <thread>
#include <optional>
#include <cstdlib>
#include <set>
#include "openvino/genai/cache_eviction.hpp"

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#elif !defined(_WIN32)
#include <sys/sysinfo.h>
#endif

#include "openvino/genai/text_streamer.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/runtime/properties.hpp"
#include "continuous_batching/pipeline_impl.hpp"
#include "utils.hpp"
#include "continuous_batching/paged_attention_transformations.hpp"
#include "lora/helper.hpp"
#include "continuous_batching/cache_state_dumper.hpp"
#include "whisper/logit_processor.hpp"

namespace {

// Returns available RAM memory on system if possible, otherwise returns std::numeric_limits<std::streamsize>::max()
size_t get_available_cpu_memory() {
#ifdef __APPLE__ 
    int64_t memsize;
    size_t len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &len, NULL, 0) == 0) {
        return memsize;
    }
#endif 

#if !defined(_WIN32)
    std::string token;
    std::ifstream file("/proc/meminfo");
    if(file.is_open()) {
        while(file >> token) {
            if(token == "MemTotal:") {
                size_t mem;
                if(file >> mem) {
                    constexpr auto max_bytes = std::numeric_limits<size_t>::max() / 1024;
                    if (mem > max_bytes) {
                        return std::numeric_limits<size_t>::max();
                    }
                    return mem * 1024;
                } else {
                    return std::numeric_limits<size_t>::max();
                }
            }
            file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    }
#endif
    return std::numeric_limits<size_t>::max();
}

// Helper function to apply selective PA transformation (self-attention only)
void apply_selective_pa_transformation(std::shared_ptr<ov::Model> model) {
    using namespace ov::op;

    std::vector<std::shared_ptr<ov::Node>> self_attn_nodes;
    std::vector<std::shared_ptr<ov::Node>> cross_attn_nodes;

    // Find all ScaledDotProductAttention nodes and classify them
    std::cout << "Analyzing SDPA nodes for selective PA transformation...\n";
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

    std::cout << "Found " << self_attn_nodes.size() << " self-attention SDPA nodes\n";
    std::cout << "Found " << cross_attn_nodes.size() << " cross-attention SDPA nodes\n\n";

    if (self_attn_nodes.empty()) {
        std::cout << "No self-attention SDPA nodes to transform\n";
        return;
    }

    // Store cross-attention connections before transformation
    struct CrossAttnInfo {
        std::shared_ptr<ov::Node> node;
        std::vector<ov::Output<ov::Node>> inputs;
        std::string friendly_name;
    };
    std::vector<CrossAttnInfo> cross_attn_info;

    for (const auto& cross_attn : cross_attn_nodes) {
        CrossAttnInfo info;
        info.node = cross_attn;
        info.friendly_name = cross_attn->get_friendly_name();
        for (size_t i = 0; i < cross_attn->get_input_size(); ++i) {
            info.inputs.push_back(cross_attn->input(i).get_source_output());
        }
        cross_attn_info.push_back(info);
    }

    // Apply SDPAToPagedAttention transformation to entire model
    std::cout << "Applying SDPAToPagedAttention transformation...\n";
    ov::pass::SDPAToPagedAttention(false, false, false, false, false, false).run_on_model(model);
    std::cout << "Transformation complete!\n\n";

    // Debug: print model parameters AFTER transformation to see what StatefulToStateless created
    std::cout << "=== MODEL PARAMETERS AFTER SDPAToPagedAttention ===\n";
    for (const auto& param : model->get_parameters()) {
        std::cout << "  PARAM: " << param->get_friendly_name() 
                  << " shape=" << param->get_output_partial_shape(0) << "\n";
    }
    // Debug: print all PA nodes and their first 3 input source names
    std::cout << "=== PA NODES AFTER TRANSFORMATION ===\n";
    for (const auto& op : model->get_ops()) {
        if (op->get_type_info().name == std::string("PagedAttentionExtension")) {
            std::cout << "  PA: " << op->get_friendly_name() << "\n";
            for (size_t i = 0; i < std::min(op->get_input_size(), size_t(3)); ++i) {
                auto src = op->input(i).get_source_output();
                std::cout << "    input[" << i << "]: " << src.get_node()->get_friendly_name()
                          << " type=" << src.get_node()->get_type_info().name << "\n";
            }
        }
    }
    std::cout << "=================================================\n\n";

    // Debug: inspect cross-attention info after transformation
    std::cout << "=== CROSS-ATTN INFO AFTER TRANSFORMATION ===\n";
    for (const auto& info : cross_attn_info) {
        std::cout << "  CrossAttn: " << info.friendly_name << "\n";
        // Is the original SDPA node still in the model?
        bool node_in_model = false;
        for (const auto& op : model->get_ops()) {
            if (op.get() == info.node.get()) { node_in_model = true; break; }
        }
        std::cout << "    original SDPA node still in model: " << (node_in_model ? "YES" : "NO") << "\n";
        for (size_t i = 0; i < info.inputs.size(); ++i) {
            auto* inp_node = info.inputs[i].get_node();
            bool in_model = false;
            for (const auto& op : model->get_ops()) {
                if (op.get() == inp_node) { in_model = true; break; }
            }
            std::cout << "    input[" << i << "]: " << inp_node->get_friendly_name()
                      << " type=" << inp_node->get_type_info().name
                      << " in_model=" << (in_model ? "YES" : "NO") << "\n";
        }
    }
    // Debug: print remaining SDPA nodes (cross-attn should still be here)
    std::cout << "=== SDPA NODES AFTER TRANSFORMATION ===\n";
    for (const auto& op : model->get_ops()) {
        if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(op)) {
            std::cout << "  SDPA: " << op->get_friendly_name() << "\n";
            for (size_t i = 0; i < op->get_input_size(); ++i) {
                auto src = op->input(i).get_source_output();
                std::cout << "    input[" << i << "]: " << src.get_node()->get_friendly_name()
                          << " type=" << src.get_node()->get_type_info().name << "\n";
            }
        }
    }
    std::cout << "============================================\n\n";

    // ---- Bypass cross-attention ReadValue/Assign state nodes ----
    // After SDPAToPagedAttention (including StatefulToStateless), the cross-attention K/V
    // nodes are now plain v6::ReadValue nodes with an init subgraph (K/V projection from
    // encoder_hidden_states). Cross-attention K/V never change during autoregression, but
    // the stateful ReadValue would read zero on step 0 (state uninitialized) and only
    // produce correct output from step 1 onwards (after the first Assign has run).
    // Fix: replace each ReadValue directly with its init subgraph output so that
    // cross-attention always computes from the live encoder_hidden_states input.
    {
        std::cout << "Bypassing cross-attention ReadValue/Assign state nodes...\n";
        std::set<std::string> cross_attn_var_ids;

        for (const auto& op : model->get_ops()) {
            if (!ov::is_type<ov::op::v13::ScaledDotProductAttention>(op)) continue;
            const std::string& name = op->get_friendly_name();
            if (name.find("encoder_attn") == std::string::npos &&
                name.find("cross_attn")   == std::string::npos) continue;

            // K = input[1], V = input[2]
            for (size_t i = 1; i <= 2 && i < op->get_input_size(); ++i) {
                auto src_node = op->input(i).get_source_output().get_node_shared_ptr();
                if (src_node->get_type_info().name != std::string("ReadValue")) continue;
                if (src_node->get_input_size() == 0) continue; // no init subgraph

                std::string var_id;
                if (auto rv6 = ov::as_type_ptr<ov::op::v6::ReadValue>(src_node))
                    var_id = rv6->get_variable_id();
                else if (auto rv3 = ov::as_type_ptr<ov::op::v3::ReadValue>(src_node))
                    var_id = rv3->get_variable_id();
                else
                    continue;

                cross_attn_var_ids.insert(var_id);

                // Wire consumers of ReadValue directly to the K/V projection subgraph
                auto init_value = src_node->input(0).get_source_output();
                for (auto& target : src_node->output(0).get_target_inputs())
                    target.replace_source_output(init_value);

                std::cout << "  ReadValue '" << src_node->get_friendly_name()
                          << "' (var=" << var_id << ") bypassed\n";
            }
        }

        // Remove Assign sinks for the bypassed state variables
        std::vector<std::shared_ptr<ov::op::Sink>> sinks_to_remove;
        for (const auto& op : model->get_ops()) {
            if (op->get_type_info().name != std::string("Assign")) continue;
            std::string var_id;
            if (auto a6 = ov::as_type_ptr<ov::op::v6::Assign>(op))
                var_id = a6->get_variable_id();
            else if (auto a3 = ov::as_type_ptr<ov::op::v3::Assign>(op))
                var_id = a3->get_variable_id();
            else
                continue;
            if (!cross_attn_var_ids.count(var_id)) continue;
            if (auto sink = std::dynamic_pointer_cast<ov::op::Sink>(op))
                sinks_to_remove.push_back(sink);
        }
        for (const auto& sink : sinks_to_remove) {
            model->remove_sink(sink);
            std::cout << "  Assign '" << sink->get_friendly_name() << "' removed\n";
        }
        std::cout << "  Cross-attention bypass complete ("
                  << cross_attn_var_ids.size() << " state variable(s) removed)\n\n";

        // Fix lm_head MatMul all-zero output caused by CPU plugin fusing LayerNorm+MatMul
        // into a single BF16/AMX FullyConnected kernel (which maps wrongly in our modified graph).
        //
        // Strategy: insert an explicit Convert(f32) node between the LayerNorm output and
        // MatMul input[0], and between the weight chain and MatMul input[1].  This breaks
        // the LN+FC fusion pattern so the CPU plugin executes the MatMul as a standalone op.
        // Also adds a debug Result on the lm_head activation input (plain LayerNorm output,
        // safe to tap) named "DEBUG_lm_input" to confirm whether zeros originate before
        // or at the MatMul during prefill.
        std::vector<std::shared_ptr<ov::op::v0::Result>> deferred_debug_results;
        for (const auto& result : model->get_results()) {
            const auto& names = result->get_output_tensor(0).get_names();
            bool is_logits = names.count("logits") > 0 ||
                             result->get_friendly_name().find("logits") != std::string::npos;
            if (!is_logits) {
                for (const auto& n : result->input(0).get_tensor().get_names())
                    if (n.find("logits") != std::string::npos) { is_logits = true; break; }
            }
            if (!is_logits) continue;

            auto lm_node = result->input(0).get_source_output().get_node_shared_ptr();
            if (!ov::as_type_ptr<ov::op::v0::MatMul>(lm_node)) {
                std::cout << "  lm_head fix: node before logits result is NOT a MatMul ("
                          << lm_node->get_type_name() << "), skipping\n";
                break;
            }
            std::cout << "  lm_head fix: found MatMul '"
                      << lm_node->get_friendly_name() << "'\n";

            // ROOT CAUSE (confirmed): CPU plugin's AMX BF16 tiled GEMM kernel for M>1
            // (the FullyConnected / prefill path) produces zeros in this modified graph.
            // M=1 (decode) uses the GEMV path which works correctly.
            //
            // FIX: Flatten [bs, T, H] → [bs*T, H] before the MatMul so the CPU plugin
            // always sees a 2-D input (M=1 per row), uses the GEMV path, then unflatten
            // [bs*T, V] → [bs, T, V] with a dynamic Reshape after.
            // The weight is also cast to FP32 (removing any FP16→Convert chain).

            // 1. Capture activation A and add debug tap
            auto A = lm_node->input(0).get_source_output();
            std::cout << "  lm_head fix: A input from '"
                      << A.get_node()->get_friendly_name()
                      << "'  type=" << A.get_element_type().get_type_name() << "\n";
            {
                auto dbg_res = std::make_shared<ov::op::v0::Result>(A);
                dbg_res->set_friendly_name("DEBUG_lm_input");
                dbg_res->get_output_tensor(0).set_names({"DEBUG_lm_input"});
                deferred_debug_results.push_back(dbg_res);
            }

            // 2. Walk input[1] to find the weight Constant and cast to FP32
            bool new_matmul_done = false;
            if (lm_node->get_input_size() >= 2) {
                ov::Output<ov::Node> w = lm_node->input(1).get_source_output();
                for (int d = 0; d < 4; ++d) {
                    auto wn = w.get_node_shared_ptr();
                    if (auto c = ov::as_type_ptr<ov::op::v0::Constant>(wn)) {
                        const auto& orig_shape = c->get_shape();
                        if (orig_shape.size() != 2) {
                            std::cout << "  lm_head fix: WARNING – weight rank != 2, skipping\n";
                            break;
                        }
                        // Weight shape: [V, H] with transpose_b=true  →  keep [V, H], cast to f32
                        size_t V = orig_shape[0], H = orig_shape[1];
                        auto fp32_data = c->cast_vector<float>();
                        auto fp32_w = std::make_shared<ov::op::v0::Constant>(
                            ov::element::f32, orig_shape, fp32_data);
                        fp32_w->set_friendly_name(c->get_friendly_name() + "_fp32");
                        std::cout << "  lm_head fix: weight [" << V << "," << H << "] cast to f32\n";

                        // 3. Flatten activation: [bs, T, H] → [bs*T, H]
                        auto flatten_shape_const = std::make_shared<ov::op::v0::Constant>(
                            ov::element::i64, ov::Shape{2},
                            std::vector<int64_t>{-1, static_cast<int64_t>(H)});
                        auto A_flat = std::make_shared<ov::op::v1::Reshape>(
                            A, flatten_shape_const->output(0), /*special_zero=*/false);
                        A_flat->set_friendly_name("lm_head_A_flat");
                        std::cout << "  lm_head fix: flatten reshape [bs,T," << H << "] → [bs*T," << H << "]\n";

                        // 4. MatMul [bs*T, H] x [V, H]^T → [bs*T, V]  (original transpose_b=true)
                        auto new_mm = std::make_shared<ov::op::v0::MatMul>(
                            A_flat, fp32_w->output(0),
                            /*transpose_a=*/false, /*transpose_b=*/true);
                        new_mm->set_friendly_name(lm_node->get_friendly_name() + "_flat");
                        std::cout << "  lm_head fix: new MatMul [bs*T," << H << "] x [" << V << "," << H << "]^T\n";

                        // 5. Unflatten: [bs*T, V] → [bs, T, V]
                        // Build output shape dynamically from A's shape: ShapeOf(A)[0:2] ++ [V]
                        auto A_shape_node = std::make_shared<ov::op::v3::ShapeOf>(A, ov::element::i64);
                        auto gather_axis  = std::make_shared<ov::op::v0::Constant>(
                            ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
                        auto gather_idx   = std::make_shared<ov::op::v0::Constant>(
                            ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, 1}); // [bs, T]
                        auto bs_T_dims = std::make_shared<ov::op::v8::Gather>(
                            A_shape_node, gather_idx, gather_axis);
                        auto V_dim = std::make_shared<ov::op::v0::Constant>(
                            ov::element::i64, ov::Shape{1}, std::vector<int64_t>{static_cast<int64_t>(V)});
                        auto out_shape = std::make_shared<ov::op::v0::Concat>(
                            ov::OutputVector{bs_T_dims->output(0), V_dim->output(0)}, 0);
                        auto mm_unflat = std::make_shared<ov::op::v1::Reshape>(
                            new_mm->output(0), out_shape->output(0), /*special_zero=*/false);
                        mm_unflat->set_friendly_name(lm_node->get_friendly_name() + "_unflat");
                        std::cout << "  lm_head fix: unflatten reshape [bs*T," << V << "] → [bs,T," << V << "]\n";

                        // 6. Replace old lm_node with mm_unflat for all consumers
                        auto old_names = lm_node->output(0).get_tensor().get_names();
                        mm_unflat->output(0).get_tensor().set_names(old_names);
                        lm_node->output(0).get_tensor().set_names({});
                        ov::replace_node(lm_node, mm_unflat);
                        std::cout << "  lm_head fix: ov::replace_node → flatten/unflatten done\n";

                        new_matmul_done = true;
                        break;
                    }
                    if (wn->get_input_size() == 0) break;
                    w = wn->input(0).get_source_output();
                }
                if (!new_matmul_done)
                    std::cout << "  lm_head fix: WARNING – could not locate weight Constant\n";
            }
            break;
        }
        // Add deferred debug results outside the get_results() iterator
        for (auto& dr : deferred_debug_results)
            model->add_results({dr});
        // Force OV to re-validate topology and infer shapes after all graph edits
        model->validate_nodes_and_infer_types();
        std::cout << "  lm_head fix: model validated and shape inference refreshed\n";

    }  // end apply_selective_pa_transformation

    // Restore cross-attention nodes back to SDPA
    for (const auto& info : cross_attn_info) {
        for (const auto& node : model->get_ops()) {
            if (node->get_type_info().name == std::string("PagedAttentionExtension")) {
                // Match by comparing first input (Q tensor)
                // Note: after SDPAToPagedAttention, Q input may have changed shape,
                // so match by checking if the Q source node (ignoring possible reshapes) matches.
                if (node->get_input_size() > 0 && info.inputs.size() > 0) {
                    // Try direct match first
                    auto pa_q = node->input(0).get_source_output();
                    bool q_matches = (pa_q == info.inputs[0]);
                    // Also try matching through a Reshape added by the transformation
                    if (!q_matches) {
                        auto q_node = pa_q.get_node_shared_ptr();
                        if (ov::is_type<ov::op::v1::Reshape>(q_node) &&
                            q_node->input(0).get_source_output() == info.inputs[0]) {
                            q_matches = true;
                        }
                    }
                    if (!q_matches) continue;

                    // CRITICAL: Use K/V from the CURRENT PA node inputs (which are wired to the
                    // StatefulToStateless-created Parameters or the preserved init subgraph),
                    // NOT from info.inputs[1/2] (those point to orphaned ReadValueWithSubgraph
                    // nodes that were removed by StatefulToStateless).
                    // For PA, inputs are: [Q, K_init_or_state, V_init_or_state, ...]
                    // However, PA uses block-cache inputs, not raw K/V.
                    // We need the K/V that fed the original SDPA before transformation.
                    // Walk backwards from the PA-internal K/V path to find the encoder projection output.
                    
                    // The PA node's input 1 is K - follow back through the graph to find the 
                    // last non-PA, non-Reshape, non-Assign node (the actual K projection output).
                    auto find_encoder_proj = [](ov::Output<ov::Node> out) -> ov::Output<ov::Node> {
                        // Walk back through Reshapes/Transposes/Assigns to find real K/V source
                        for (int depth = 0; depth < 8; ++depth) {
                            auto n = out.get_node_shared_ptr();
                            const auto& type = n->get_type_info().name;
                            if (type == std::string("Reshape") || 
                                type == std::string("Transpose") ||
                                type == std::string("Convert") ||
                                type == std::string("Assign")) {
                                out = n->input(0).get_source_output();
                            } else {
                                break;
                            }
                        }
                        return out;
                    };

                    // For the restored SDPA, use:
                    // - Q: from info.inputs[0] (unchanged by transformation - no state involved)
                    // - K: from info.inputs[1] IF it's still connected to the live graph, 
                    //      otherwise from PA node's K-related input
                    // - V: same logic for info.inputs[2]
                    // Check if info.inputs[1] node is still alive in the model graph
                    auto is_in_model_graph = [&model](const ov::Output<ov::Node>& out) -> bool {
                        for (const auto& op : model->get_ops()) {
                            if (op.get() == out.get_node()) return true;
                        }
                        return false;
                    };

                    ov::Output<ov::Node> k_input, v_input;
                    if (info.inputs.size() > 1 && is_in_model_graph(info.inputs[1])) {
                        k_input = info.inputs[1];
                        std::cout << "  K: using original graph input (still in model)\n";
                    } else {
                        // info.inputs[1] is orphaned - find K through the PA node's inputs
                        // PA node inputs after transformation: [Q, ..., block_indices, ...]
                        // The K/V init state input should be accessible via the state variable
                        // Walk back from PA's input to find the encoder K projection
                        if (node->get_input_size() > 1) {
                            k_input = find_encoder_proj(node->input(1).get_source_output());
                        } else {
                            k_input = info.inputs[1]; // fallback even if orphaned
                        }
                        std::cout << "  K: using PA node input (original was orphaned)\n";
                    }
                    if (info.inputs.size() > 2 && is_in_model_graph(info.inputs[2])) {
                        v_input = info.inputs[2];
                        std::cout << "  V: using original graph input (still in model)\n";
                    } else {
                        if (node->get_input_size() > 2) {
                            v_input = find_encoder_proj(node->input(2).get_source_output());
                        } else {
                            v_input = info.inputs[2]; // fallback even if orphaned
                        }
                        std::cout << "  V: using PA node input (original was orphaned)\n";
                    }

                    // Q input for restored cross-attention SDPA:
                    // info.inputs[0] was captured BEFORE SDPAToPagedAttention.  After the
                    // transformation, the self-attention SDPA that fed Q was replaced by a PA
                    // node, making info.inputs[0] point to a dead/orphaned node whose output
                    // is garbage.  Instead, get Q from the live cross-attention PA node's
                    // input(0), which was correctly updated by ov::replace_node.
                    auto q_input = node->input(0).get_source_output();
                    std::cout << "  Q: from live PA input(0): '"
                              << q_input.get_node()->get_friendly_name() << "'\n";

                    std::shared_ptr<ov::Node> new_sdpa;
                    if (info.inputs.size() >= 4 && is_in_model_graph(info.inputs[3])) {
                        new_sdpa = std::make_shared<v13::ScaledDotProductAttention>(
                            q_input, k_input, v_input, info.inputs[3], false);
                    } else {
                        new_sdpa = std::make_shared<v13::ScaledDotProductAttention>(
                            q_input, k_input, v_input, false);
                    }
                    new_sdpa->set_friendly_name(info.friendly_name);
                    ov::replace_node(node, new_sdpa);
                    std::cout << "  Restored: " << new_sdpa->get_friendly_name() << "\n";
                    break;
                }
            }
        }
    }
    std::cout << "Selective PA transformation complete!\n\n";
}

} // namespace

namespace ov::genai {
template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

ContinuousBatchingPipeline::ContinuousBatchingImpl::ContinuousBatchingImpl(
    const std::shared_ptr<ov::Model>& model,
    const Tokenizer& tokenizer,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& properties,
    const ov::genai::GenerationConfig& generation_config,
    bool is_validation_mode_enabled) {
    m_tokenizer = tokenizer;
    m_generation_config = generation_config;
    m_is_validation_mode_enabled = is_validation_mode_enabled;

    // Reshape 2D input_ids to 1D for models like Whisper that have [batch, seq_len] inputs
    // This makes them compatible with the continuous batching infrastructure
    // COMMENTED OUT: This breaks existing Reshape nodes in the model that expect 2D input
    // The issue may be in how we set the input tensor, not in the model structure
    /*
    std::cout << "Continuous Batching Pipeline: Checking for 2D input_ids parameter to reshape to 1D...\n";
    std::cout << "Model parameters:\n";
    for (auto& param : model->get_parameters()) {
        auto shape = param->get_partial_shape();
        std::cout << "  - " << param->get_friendly_name() << ": shape=" << shape 
                  << ", rank=" << (shape.rank().is_static() ? std::to_string(shape.rank().get_length()) : "dynamic") << "\n";
    }
    
    for (auto& param : model->get_parameters()) {
        if (param->get_friendly_name() == "input_ids" || param->get_friendly_name() == "decoder_input_ids") {
            auto shape = param->get_partial_shape();
            std::cout << "Found " << param->get_friendly_name() << " parameter with shape: " << shape << "\n";
            if (shape.rank().is_static() && shape.rank().get_length() == 2) {
                std::cout << "Reshaping " << param->get_friendly_name() << " from 2D " << shape << " to 1D {?}\n";
                
                // Find all consumers of this parameter
                auto consumers = param->output(0).get_target_inputs();
                if (!consumers.empty()) {
                    // Create a Reshape node to flatten the input
                    auto shape_const = std::make_shared<ov::op::v0::Constant>(
                        ov::element::i32, ov::Shape{1}, std::vector<int32_t>{-1});
                    auto reshape = std::make_shared<ov::op::v1::Reshape>(param, shape_const, false);
                    reshape->set_friendly_name("input_ids_reshape");
                    
                    // Replace parameter usage with reshape output
                    for (auto& consumer : consumers) {
                        consumer.replace_source_output(reshape->output(0));
                    }
                    
                    std::cout << "Added Reshape node to flatten input_ids to 1D\n";
                }
            }
            break;
        }
    }
    */
    
    // Check if PA transformation should be skipped entirely for debugging
    const char* skip_pa = std::getenv("SKIP_PA_TRANSFORMATION");
    bool skip_pa_transformation = (skip_pa != nullptr && std::string(skip_pa) == "1");
    
    // Check if selective PA transformation is requested (self-attention only)
    const char* pa_self_attn_only = std::getenv("PA_SELF_ATTEN_ONLY");
    bool use_selective_pa = (pa_self_attn_only != nullptr && std::string(pa_self_attn_only) == "1");

    if (skip_pa_transformation) {
        std::cout << "SKIP_PA_TRANSFORMATION=1: Skipping PA transformation entirely for debugging\n";
        std::cout << "WARNING: This means no continuous batching - just testing if PA is the issue\n";
    } else if (use_selective_pa) {
        std::cout << "PA_SELF_ATTEN_ONLY=1 detected: applying selective PA transformation (self-attention only)\n";
        apply_selective_pa_transformation(model);
        utils::apply_gather_before_matmul_transformation(model);

                // Post-process: Remove incorrect Unsqueeze added after decoder_input_ids/input_ids for Whisper
        // The PA transformation incorrectly adds Unsqueeze [?,?] -> [?,1,?] which breaks subsequent Reshape
        std::cout << "\nPost-processing: Checking for incorrect Unsqueeze after decoder_input_ids...\n";
        for (auto& param : model->get_parameters()) {
            if (param->get_friendly_name() == "input_ids" || param->get_friendly_name() == "decoder_input_ids") {
                auto shape = param->get_partial_shape();
                std::cout << "Found parameter: " << param->get_friendly_name() << " with shape " << shape << "\n";
                
                // Check if first consumer is an Unsqueeze
                auto consumers = param->output(0).get_target_inputs();
                for (auto& consumer : consumers) {
                    auto consumer_node = consumer.get_node();
                    if (ov::is_type<ov::op::v0::Unsqueeze>(consumer_node)) {
                        std::cout << "  Found Unsqueeze node: " << consumer_node->get_friendly_name() << "\n";
                        
                        // Get the Unsqueeze node's consumers
                        auto unsqueeze_consumers = consumer_node->output(0).get_target_inputs();
                        
                        // Bypass the Unsqueeze ONLY for Reshape consumers
                        // Keep it for ShapeOf and other operations that might need the 3D shape
                        for (auto& unsqueeze_consumer : unsqueeze_consumers) {
                            auto consumer_op = unsqueeze_consumer.get_node();
                            if (ov::is_type<ov::op::v1::Reshape>(consumer_op)) {
                                unsqueeze_consumer.replace_source_output(param->output(0));
                                std::cout << "    Bypassed Unsqueeze for Reshape: " 
                                          << consumer_op->get_friendly_name() << "\n";
                            } else {
                                std::cout << "    Kept Unsqueeze for: " 
                                          << consumer_op->get_friendly_name() << " (type: " 
                                          << consumer_op->get_type_name() << ")\n";
                            }
                        }
                        
                        std::cout << "  Selectively removed Unsqueeze connections for " << param->get_friendly_name() << "\n";
                    }
                }
            }
        }
        std::cout << "Post-processing complete.\n\n";

    } else {
        std::cout << "Applying full PA transformation (all SDPA nodes)\n";
        bool is_need_per_layer_cache_control = scheduler_config.use_cache_eviction;
        bool allow_cache_rotation = scheduler_config.cache_eviction_config.apply_rotation;
        bool allow_xattention = scheduler_config.use_sparse_attention && scheduler_config.sparse_attention_config.mode == SparseAttentionMode::XATTENTION;
        bool allow_score_aggregation = true;
        bool allow_adaptive_rkv = scheduler_config.use_cache_eviction && scheduler_config.cache_eviction_config.aggregation_mode == AggregationMode::ADAPTIVE_RKV;
        auto sdpa_to_pa_successful = ov::pass::SDPAToPagedAttention(is_need_per_layer_cache_control, is_need_per_layer_cache_control, allow_score_aggregation, allow_cache_rotation, allow_xattention, allow_adaptive_rkv).run_on_model(model);
        if (sdpa_to_pa_successful) {
            std::cout << "SDPA to Paged Attention transformation applied successfully.\n";
        } else {
            std::cout << "SDPA to Paged Attention transformation was not applied successfully.\n";
        }
        utils::apply_gather_before_matmul_transformation(model);
        
        // Note: decoder_input_ids stays as [?,?] parameter
        // We handle 1D->2D conversion at runtime in ModelRunner
    }
    
    // Additional post-processing: Fix Reshape constants before PagedAttentionExtension
    // The SDPAToPagedAttention transformation creates Reshape nodes with pattern [1, -1]
    // but for continuous batching we need [-1, hidden_size] to properly flatten batch*seq_len
    std::cout << "\nPost-processing: Fixing Reshape patterns before PagedAttentionExtension...\n";
    for (auto& node : model->get_ordered_ops()) {
        if (node->get_type_name() == std::string("PagedAttentionExtension")) {
            std::cout << "Found PagedAttentionExtension: " << node->get_friendly_name() << "\n";
            
            // Check inputs 0, 1, 2 (Q, K, V)
            for (size_t input_idx = 0; input_idx < 3; ++input_idx) {
                auto input_node = node->get_input_node_shared_ptr(input_idx);
                if (ov::is_type<ov::op::v1::Reshape>(input_node)) {
                    std::cout << "  Input " << input_idx << " is Reshape: " << input_node->get_friendly_name() << "\n";
                    
                    // Get the reshape pattern (should be second input)
                    auto pattern_node = input_node->get_input_node_shared_ptr(1);
                    if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(pattern_node)) {
                        auto pattern_data = constant->cast_vector<int64_t>();
                        std::cout << "    Current pattern: [";
                        for (size_t i = 0; i < pattern_data.size(); ++i) {
                            std::cout << pattern_data[i];
                            if (i < pattern_data.size() - 1) std::cout << ", ";
                        }
                        std::cout << "]\n";
                        
                        // Debug: print each element
                        std::cout << "    Pattern analysis: size=" << pattern_data.size();
                        if (pattern_data.size() >= 1) std::cout << ", [0]=" << pattern_data[0];
                        if (pattern_data.size() >= 2) std::cout << ", [1]=" << pattern_data[1];
                        std::cout << "\n";
                        
                        // If pattern is [0, -1] or [1, -1], change to [-1, hidden_size]
                        // [0, -1] means "copy first dim from input, infer second" which is wrong for PA
                        bool needs_fix = (pattern_data.size() == 2 && 
                                         (pattern_data[0] == 0 || pattern_data[0] == 1) && 
                                         pattern_data[1] == -1);
                        std::cout << "    Pattern needs_fix: " << (needs_fix ? "YES" : "NO") << "\n";
                        
                        if (needs_fix) {
                            std::cout << "    Pattern needs fixing: [" << pattern_data[0] << ", -1] -> [-1, hidden_size]\n";
                            
                            // Get the input shape to Reshape to determine hidden_size
                            auto reshape_input_shape = input_node->get_input_partial_shape(0);
                            std::cout << "    Reshape input shape: " << reshape_input_shape << "\n";
                            std::cout << "    Reshape input rank: " << reshape_input_shape.rank() << "\n";
                            
                            // For shape [batch, seq, num_heads, head_dim] or [?, ?, num_heads, head_dim]
                            // hidden_size = num_heads * head_dim (should be static even if batch/seq are dynamic)
                            int64_t hidden_size = -1;
                            if (reshape_input_shape.rank().is_static() && reshape_input_shape.rank().get_length() == 4) {
                                std::cout << "    Reshape input is 4D\n";
                                auto num_heads_dim = reshape_input_shape[2];
                                auto head_dim = reshape_input_shape[3];
                                std::cout << "    num_heads_dim: " << num_heads_dim << ", head_dim: " << head_dim << "\n";
                                
                                if (num_heads_dim.is_static() && head_dim.is_static()) {
                                    hidden_size = num_heads_dim.get_length() * head_dim.get_length();
                                    std::cout << "    Computed hidden_size: " << hidden_size << "\n";
                                } else {
                                    std::cout << "    ERROR: num_heads or head_dim is dynamic\n";
                                }
                            } else {
                                std::cout << "    ERROR: Reshape input is not 4D or rank is dynamic\n";
                            }
                            
                            // If we couldn't compute hidden_size, try hardcoded value for Whisper
                            if (hidden_size == -1) {
                                // For Whisper large-v3-turbo, hidden_size is 1280 (20 heads * 64 head_dim)
                                // Check if this is a Whisper model by looking for 20 heads
                                if (reshape_input_shape.rank().is_static() && reshape_input_shape.rank().get_length() == 4) {
                                    auto num_heads_dim = reshape_input_shape[2];
                                    if (num_heads_dim.is_static() && num_heads_dim.get_length() == 20) {
                                        hidden_size = 1280;
                                        std::cout << "    Using hardcoded hidden_size for Whisper: " << hidden_size << "\n";
                                    }
                                }
                            }
                            
                            // Apply fix if we computed hidden_size
                            if (hidden_size > 0) {
                                    
                                    // Create new constant with pattern [-1, hidden_size]
                                    std::vector<int64_t> new_pattern = {-1, hidden_size};
                                    auto new_constant = std::make_shared<ov::op::v0::Constant>(
                                        ov::element::i64, ov::Shape{2}, new_pattern);
                                    
                                    // Replace the old constant
                                    input_node->input(1).replace_source_output(new_constant->output(0));
                                    
                                    // Verify the replacement worked
                                    auto verify_node = input_node->get_input_node_shared_ptr(1);
                                    if (auto verify_constant = ov::as_type_ptr<ov::op::v0::Constant>(verify_node)) {
                                        auto verify_pattern = verify_constant->cast_vector<int64_t>();
                                        std::cout << "    VERIFIED - New pattern: [";
                                        for (size_t i = 0; i < verify_pattern.size(); ++i) {
                                            std::cout << verify_pattern[i];
                                            if (i < verify_pattern.size() - 1) std::cout << ", ";
                                        }
                                        std::cout << "]\n";
                                    }
                                    
                                    std::cout << "    Fixed pattern: [-1, " << hidden_size << "]\n";
                            } else {
                                std::cout << "    ERROR: Could not determine hidden_size for shape " << reshape_input_shape << "\n";
                            }
                        } else {
                            std::cout << "    Pattern does not need fixing\n";
                        }
                    }
                }
            }
        }
    }
    std::cout << "Input Reshape pattern fixes complete.\n\n";
    
    // Fix OUTPUT Reshapes from PagedAttentionExtension
    // Pattern is typically [0, 1, -1, head_dim] via Concat, but should be [1, -1, num_heads, head_dim]
    std::cout << "Fixing OUTPUT Reshapes from PagedAttentionExtension...\n";
    for (auto& node : model->get_ordered_ops()) {
        if (node->get_type_name() == std::string("PagedAttentionExtension")) {
            std::cout << "Checking outputs of PA: " << node->get_friendly_name() << "\n";
            
            // Check all consumers of this PA node
            for (auto& output : node->outputs()) {
                for (auto& input : output.get_target_inputs()) {
                    auto consumer = input.get_node()->shared_from_this();
                    
                    if (ov::is_type<ov::op::v1::Reshape>(consumer)) {
                        std::cout << "  OUTPUT to Reshape: " << consumer->get_friendly_name() << "\n";
                        
                        // Get the reshape pattern source (usually a Concat node)
                        auto pattern_node = consumer->get_input_node_shared_ptr(1);
                        
                        if (ov::is_type<ov::op::v0::Concat>(pattern_node)) {
                            std::cout << "    Pattern from Concat node: " << pattern_node->get_friendly_name() << "\n";
                            
                            // The Concat typically combines [batch, seq, num_heads, head_dim]
                            // Pattern [0, 1, -1, head_dim] is wrong - should be [1, -1, num_heads, head_dim]
                            // We need to replace:
                            // - Input 0: Change constant '0' to '1' (force batch=1 instead of copy)
                            // - Input 1: Change constant '1' to '-1' (infer seq_len instead of static 1)
                            // - Input 2: Change constant '-1' to '20' (explicit num_heads for Whisper)
                            
                            for (size_t concat_input = 0; concat_input < pattern_node->get_input_size(); ++concat_input) {
                                auto concat_input_node = pattern_node->get_input_node_shared_ptr(concat_input);
                                
                                if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(concat_input_node)) {
                                    auto values = constant->cast_vector<int64_t>();
                                    if (values.size() == 1) {
                                        int64_t val = values[0];
                                        std::cout << "      Concat input " << concat_input << ": " << val;
                                        
                                        // Fix input 0: change 0 (copy) to 1 (static batch)
                                        if (concat_input == 0 && val == 0) {
                                            auto new_constant = std::make_shared<ov::op::v0::Constant>(
                                                ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
                                            pattern_node->input(concat_input).replace_source_output(new_constant->output(0));
                                            std::cout << " -> 1 (FIXED: force batch=1)\n";
                                        }
                                        // Fix input 1: change 1 (static) to -1 (infer seq_len)
                                        else if (concat_input == 1 && val == 1) {
                                            auto new_constant = std::make_shared<ov::op::v0::Constant>(
                                                ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
                                            pattern_node->input(concat_input).replace_source_output(new_constant->output(0));
                                            std::cout << " -> -1 (FIXED: infer seq_len)\n";
                                        }
                                        // Fix input 2: change -1 (infer) to 20 (explicit num_heads)
                                        else if (concat_input == 2 && val == -1) {
                                            auto new_constant = std::make_shared<ov::op::v0::Constant>(
                                                ov::element::i64, ov::Shape{1}, std::vector<int64_t>{20});
                                            pattern_node->input(concat_input).replace_source_output(new_constant->output(0));
                                            std::cout << " -> 20 (FIXED: explicit num_heads)\n";
                                        }
                                        else {
                                            std::cout << " (unchanged)\n";
                                        }
                                    }
                                }
                            }
                        } else {
                            std::cout << "    Pattern is not a Concat, skipping\n";
                        }
                    }
                }
            }
        }
    }
    std::cout << "Output Reshape fixes complete.\n\n";
    
    // Revalidate the model after all modifications
    std::cout << "Revalidating model after all transformations...\n";
    model->validate_nodes_and_infer_types();
    std::cout << "Model validation complete.\n\n";
    
    // Verify final parameter shapes
    std::cout << "Final parameter shapes after all modifications:\n";
    for (auto& param : model->get_parameters()) {
        std::cout << "  " << param->get_friendly_name() << ": " << param->get_partial_shape() << "\n";
    }
    std::cout << "\n";

    // Save the transformed model to XML for debugging
    std::cout << "Saving transformed model to pa_experimental_model.xml...\n";
    try {
        ov::pass::Serialize("./pa_experimental_model.xml", "./pa_experimental_model.bin").run_on_model(model);
        std::cout << "Model saved successfully!\n\n";
    } catch (const std::exception& e) {
        std::cout << "Failed to save model: " << e.what() << "\n\n";
    }

    initialize_pipeline(model, scheduler_config, device, properties);
}

ContinuousBatchingPipeline::ContinuousBatchingImpl::ContinuousBatchingImpl(
    const std::shared_ptr<ov::Model>& model,
    std::shared_ptr<InputsEmbedder> inputs_embedder,
    const Tokenizer& tokenizer,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& properties,
    const ov::genai::GenerationConfig& generation_config,
    bool is_validation_mode_enabled) : ContinuousBatchingImpl(model, tokenizer, scheduler_config, device, properties, generation_config, is_validation_mode_enabled){
    m_inputs_embedder = inputs_embedder;
    // Note: set_inputs_embedder also sets the embedding model internally.
    m_model_runner->set_inputs_embedder(inputs_embedder);
    m_model_input_type = ModelInputType::EMBEDDINGS;
    m_vision_registry = std::make_shared<VisionRegistry>();
}

ContinuousBatchingPipeline::ContinuousBatchingImpl::ContinuousBatchingImpl(
    const std::shared_ptr<ov::Model>& model,
    std::shared_ptr<SpeechEncoder> speech_encoder,
    const Tokenizer& tokenizer,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& properties,
    const ov::genai::GenerationConfig& generation_config,
    bool is_validation_mode_enabled) : ContinuousBatchingImpl(model, tokenizer, scheduler_config, device, properties, generation_config, is_validation_mode_enabled){
    m_speech_encoder = speech_encoder;
    // Note: set_inputs_embedder also sets the embedding model internally.
    // m_model_runner->set_inputs_embedder(inputs_embedder);
    //m_model_input_type = ModelInputType::EMBEDDINGS;
}

ContinuousBatchingPipeline::ContinuousBatchingImpl::~ContinuousBatchingImpl() {
    // manually release all blocks, which can re-initialize OpenVINO plugins during destruction
    if (m_model_runner) {
        m_model_runner->get_infer_request().get_compiled_model().release_memory();
    }

    if (m_scheduler) {
        m_scheduler->release();
    }
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::_pull_awaiting_requests() {
    std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
    m_requests.insert(m_requests.end(), m_awaiting_requests.begin(), m_awaiting_requests.end());
    m_awaiting_requests.clear();
    m_pipeline_metrics.requests = m_requests.size();
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::initialize_pipeline(
    std::shared_ptr<ov::Model> model,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& properties) {
    m_device = device;
    // apply LoRA
    auto filtered_properties = extract_adapters_from_properties(properties, &m_generation_config.adapters);
    if (m_generation_config.adapters) {
        m_generation_config.adapters->set_tensor_name_prefix("base_model.model.");
        m_adapter_controller = AdapterController(model, *m_generation_config.adapters, device);   // TODO: Make the prefix name configurable
    }
    // Extract sampler_num_threads property if exists and remove it from properties
    size_t sampler_num_threads = std::thread::hardware_concurrency();
    auto sampler_num_threads_it = filtered_properties->find("sampler_num_threads");
    if (sampler_num_threads_it != filtered_properties->end()) {
        sampler_num_threads = sampler_num_threads_it->second.as<size_t>();
        filtered_properties.fork().erase("sampler_num_threads");   // do not use iterator sampler_num_threads_it because a forked container may not be the same container
    }

    // Debug: print all model input names just before compile - this shows what set_tensor names are valid
    std::cout << "=== MODEL INPUTS BEFORE COMPILE ===\n";
    for (const auto& input : model->inputs()) {
        std::cout << "  INPUT: ";
        for (const auto& name : input.get_names()) std::cout << "'" << name << "' ";
        std::cout << "  shape=" << input.get_partial_shape() << "\n";
    }
    std::cout << "===================================\n\n";

    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(model, device, *filtered_properties);
    std::vector<std::string> execution_devices = compiled_model.get_property(ov::execution_devices);
    const bool all_gpu_device =
        std::all_of(execution_devices.begin(), execution_devices.end(), [&](const std::string& device) {
            return device.find("GPU") != std::string::npos;
        });
    OPENVINO_ASSERT(all_gpu_device || execution_devices.size() == 1,
                    "Continuous batching: execution device is expected to be single CPU / single GPU / multi GPUs");
    const std::string execution_device = execution_devices[0];

    ov::genai::utils::print_compiled_model_properties(compiled_model, "LLM with Paged Attention");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // Cache manager
    std::shared_ptr<CacheManager> cache_manager = std::make_shared<CacheManager>(infer_request);
    m_num_decoder_layers = cache_manager->get_num_decoder_layers();
    m_block_size = cache_manager->get_block_size();


    // Scheduler configuration
    SchedulerConfig normalized_config = scheduler_config;
    size_t total_mem_size;
    if (execution_device.find("GPU") != std::string::npos) {
        total_mem_size = utils::get_available_gpu_memory(execution_device, m_num_decoder_layers);
    } else {
        total_mem_size = get_available_cpu_memory();
    }
    if (normalized_config.num_kv_blocks == 0 && normalized_config.cache_size > 0) {
        size_t size_in_bytes = normalized_config.cache_size * 1024 * 1024 * 1024; // convert GBs to bytes
        OPENVINO_ASSERT(size_in_bytes <= total_mem_size, "Requested KV-cache size is larger than available memory size on the system.");
        normalized_config.num_kv_blocks = size_in_bytes / cache_manager->get_block_size_in_bytes();
    }
    if (normalized_config.num_kv_blocks > 0) {
        size_t size_in_bytes = cache_manager->get_block_size_in_bytes() * normalized_config.num_kv_blocks;
        OPENVINO_ASSERT(size_in_bytes <= total_mem_size, "Requested number of KV-blocks require more memory than available on the system.");
    }

    bool can_use_partial_preemption = true;
    if (execution_device.find("GPU") != std::string::npos && !normalized_config.dynamic_split_fuse) {
        // in case of executing a `vLLM-like` pipeline, it's better not to use partial eviction on the GPU,
        // as it may lead to performance slowdown
        can_use_partial_preemption = false;
    }

    // Scheduler and Model Runner instantiation
    bool is_use_xattention = scheduler_config.use_sparse_attention && scheduler_config.sparse_attention_config.mode == SparseAttentionMode::XATTENTION;
    bool is_use_cache_eviction = scheduler_config.use_cache_eviction;
    if (is_use_cache_eviction) {
        const auto& eviction_config = scheduler_config.cache_eviction_config;
        m_scheduler = std::make_shared<Scheduler>(m_block_size, cache_manager, normalized_config, m_num_decoder_layers, can_use_partial_preemption, eviction_config.snapkv_window_size);

        bool is_apply_rotation = eviction_config.apply_rotation;
        bool is_use_adaptive_rkv = (eviction_config.aggregation_mode == AggregationMode::ADAPTIVE_RKV);
        m_model_runner = std::make_shared<ModelRunner>(infer_request,
                                                       m_block_size,
                                                       m_num_decoder_layers,
                                                       /* collect_attention_scores = */ true,
                                                       /* is_use_per_layer_cache_control = */ true,
                                                       /* is_use_rotation_inputs = */ is_apply_rotation,
                                                       /* is_aggregate_attention_scores = */ true,
                                                       is_use_xattention,
                                                       is_use_adaptive_rkv);
        if (eviction_config.apply_rotation) {
            _prepare_rotation_data_storage(normalized_config, cache_manager->get_v_head_size(0));
        }
    } else {
        m_scheduler = std::make_shared<Scheduler>(m_block_size, cache_manager, normalized_config, m_num_decoder_layers, can_use_partial_preemption);
        m_model_runner =
            std::make_shared<ModelRunner>(infer_request, m_block_size, m_num_decoder_layers,
                                                       /* collect_attention_scores = */ false,
                                                       /* is_use_per_layer_cache_control = */ false,
                                                       /* is_use_rotation_inputs = */ false,
                                                       /* is_aggregate_attention_scores = */ false,
                                                       is_use_xattention,
                                                       /* is_use_adaptive_rkv = */ false);
    }

    m_sampler = std::make_shared<Sampler>(m_tokenizer, sampler_num_threads);
    m_sampler->set_seed(m_generation_config.rng_seed);

    // If eos_token_id was not provided, take value
    if (m_generation_config.eos_token_id == -1)
        m_generation_config.set_eos_token_id(m_tokenizer.get_eos_token_id());
};


void ContinuousBatchingPipeline::ContinuousBatchingImpl::_prepare_rotation_data_storage(const SchedulerConfig& normalized_config, size_t embedding_size) {
    m_rotation_deltas_stores.reserve(m_num_decoder_layers);
    ov::Shape rotation_deltas_store_shape{normalized_config.num_kv_blocks, 1}; // last dim can be later changed to BLOCK_SIZE for per-token granularity
    for (size_t i = 0; i < m_num_decoder_layers; i++) {
        ov::Tensor store(ov::element::i32, rotation_deltas_store_shape);
        std::memset(store.data(), 0, store.get_byte_size());
        m_rotation_deltas_stores.push_back(store);
    }

    size_t max_sequence_cache_occupation_length_in_blocks = normalized_config.max_num_batched_tokens / m_block_size  + 1;
    m_cache_rotation_calculator = std::make_shared<CacheRotationCalculator>(
        m_block_size,
        max_sequence_cache_occupation_length_in_blocks,
        embedding_size);
    auto rotation_trig_lut = ov::Tensor(ov::element::f32, ov::Shape{max_sequence_cache_occupation_length_in_blocks, embedding_size});
    float* rotation_trig_lut_data = rotation_trig_lut.data<float>();
    std::memset(rotation_trig_lut_data, 0, rotation_trig_lut.get_byte_size());

    const auto& cos_lut = m_cache_rotation_calculator->get_cos_lut();
    const auto& sin_lut = m_cache_rotation_calculator->get_sin_lut();


    for (size_t pos_idx = 0; pos_idx < max_sequence_cache_occupation_length_in_blocks; pos_idx++) {
        for (size_t embedding_pair_idx = 0; embedding_pair_idx < cos_lut[0].size(); embedding_pair_idx++) {
            rotation_trig_lut_data[pos_idx * embedding_size + embedding_pair_idx] = cos_lut[pos_idx][embedding_pair_idx];
            rotation_trig_lut_data[pos_idx * embedding_size + embedding_size / 2 + embedding_pair_idx] = sin_lut[pos_idx][embedding_pair_idx];
        }
    }

    m_model_runner->set_cache_rotation_trig_lut(std::move(rotation_trig_lut));
}

GenerationHandle
ContinuousBatchingPipeline::ContinuousBatchingImpl::add_request(
    uint64_t request_id,
    const ov::Tensor& input_ids,
    const ov::genai::GenerationConfig& sampling_params,
    std::optional<ov::Tensor> token_type_ids,
    std::optional<ov::Tensor> encoder_hidden_state) {
    auto sampling_params_copy = sampling_params;
    // If stop_token_ids were not provided, take value from default m_generation_config
    if (sampling_params_copy.stop_token_ids.empty())
        sampling_params_copy.stop_token_ids = m_generation_config.stop_token_ids;
    // If eos_token_id was not provided, take value from default m_generation_config
    if (sampling_params_copy.eos_token_id == -1)
        sampling_params_copy.set_eos_token_id(m_generation_config.eos_token_id);
    sampling_params_copy.validate();
    size_t prompt_len;
    if (input_ids.get_shape().size() > 1) {
        prompt_len = input_ids.get_shape()[1];
    } else {
        prompt_len = input_ids.get_size();
    }
    OPENVINO_ASSERT(sampling_params_copy.max_length > prompt_len, "'max_length' must be greater than the number of prompt tokens");

    std::shared_ptr<SequenceGroup> sequence_group;
    if (m_model_input_type == ModelInputType::EMBEDDINGS) {
        const auto [position_ids, rope_delta] = m_inputs_embedder->get_position_ids(input_ids.get_shape()[1], 0);
        sequence_group = std::make_shared<SequenceGroup>(request_id, 
                                                         input_ids, 
                                                         sampling_params_copy, 
                                                         m_block_size, 
                                                         token_type_ids, 
                                                         position_ids, 
                                                         rope_delta);
    }
    else {
        if (encoder_hidden_state.has_value()) {
            std::cout << "Creating SequenceGroup with encoder_hidden_state\n";
            std::cout << "  encoder_hidden_state shape: [";
            for (size_t i = 0; i < encoder_hidden_state->get_shape().size(); ++i) {
                std::cout << encoder_hidden_state->get_shape()[i];
                if (i < encoder_hidden_state->get_shape().size() - 1) std::cout << ", ";
            }
            std::cout << "], size: " << encoder_hidden_state->get_size() << "\n";
            
            sequence_group = std::make_shared<SequenceGroup>(request_id, input_ids, *encoder_hidden_state, sampling_params_copy, m_block_size);
        } else {
            std::cout << "Creating SequenceGroup WITHOUT encoder_hidden_state\n";
            sequence_group = std::make_shared<SequenceGroup>(request_id, input_ids, sampling_params_copy, m_block_size, token_type_ids);
        }    
    }

    if (m_scheduler->get_config().enable_prefix_caching) {
        m_scheduler->restore_cached_blocks(sequence_group);
    }

    {
        std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
        m_awaiting_requests.push_back(sequence_group);
    }
    std::cout << "Added request " << request_id << " with prompt length " << prompt_len << "\n";
    return std::make_shared<GenerationHandleImpl>(sequence_group->get_generation_stream(), sampling_params_copy);
}

GenerationHandle
ContinuousBatchingPipeline::ContinuousBatchingImpl::add_request(uint64_t request_id,
                                                                const std::string& prompt,
                                                                const ov::genai::GenerationConfig& sampling_params) {
    ov::Tensor inputs;
    ov::genai::VLMPerfMetrics metrics;
    if (m_model_input_type == ModelInputType::TOKENS) {
        static ManualTimer timer("tokenize");
        timer.start();
        inputs = m_tokenizer.encode(prompt).input_ids;
        timer.end();
        return add_request(request_id, inputs, sampling_params);
    } else if (m_model_input_type == ModelInputType::EMBEDDINGS) {
        return ContinuousBatchingPipeline::IContinuousBatchingPipeline::add_request(request_id, prompt, {}, sampling_params);
    } else {
        OPENVINO_THROW("Unknown model input type.");
    }

    return add_request(request_id, inputs, sampling_params);
}

GenerationHandle
ContinuousBatchingPipeline::ContinuousBatchingImpl::add_request(uint64_t request_id,
                                 const ov::Tensor& input_ids,
                                 const ov::Tensor& encoder_hidden_state,
                                 const ov::genai::WhisperGenerationConfig& sampling_params) {
    std::cout << "ContinuousBatchingPipeline::ContinuousBatchingImpl::add_request with encoder_hidden_state called\n";
    // WhisperGenerationConfig inherits from GenerationConfig, so we can pass it directly
    return add_request(request_id, input_ids, sampling_params, std::nullopt, encoder_hidden_state);
}

bool ContinuousBatchingPipeline::ContinuousBatchingImpl::has_non_finished_requests() {
    std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
    return !m_awaiting_requests.empty() || !m_requests.empty();
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::step() {
    static ManualTimer step_timer("step()");
    step_timer.start();

    _pull_awaiting_requests();

    Scheduler::Output scheduler_output;

    {
        static ManualTimer scheduling_timer("scheduling");
        scheduling_timer.start();
        scheduler_output = m_scheduler->schedule(m_requests);
        scheduling_timer.end();

        m_pipeline_metrics.scheduled_requests = scheduler_output.m_scheduled_sequence_groups_ids.size();
        m_pipeline_metrics.cache_usage = scheduler_output.m_cache_usage;
        m_pipeline_metrics.max_cache_usage = std::max(m_pipeline_metrics.max_cache_usage, scheduler_output.m_cache_usage);
        _register_step_cache_usage(scheduler_output.m_cache_usage);
        m_pipeline_metrics.avg_cache_usage = _get_current_running_average_cache_usage();

        const auto& sched_config = m_scheduler->get_config();
        if (sched_config.use_cache_eviction) {
           if (sched_config.cache_eviction_config.apply_rotation) {
                _compute_cache_rotation_data(m_requests, scheduler_output);
                m_model_runner->set_cache_rotation_data(std::move(m_current_step_rotated_block_indices_per_sequence),
                                                        std::move(m_current_step_rotation_deltas));
           }
        }

    }

    // if no tokens were scheduled, we are out of memory => free all requests and return
    if (scheduler_output.m_total_num_scheduled_tokens == 0) {
        for (size_t i = 0; i < m_requests.size(); ++i) {
            SequenceGroup::Ptr sequence_group = m_requests[i];
            if (!sequence_group->is_waiting()) {
                sequence_group->set_out_of_memory();
                sequence_group->notify_handle();
            }
        }
        _free_non_running_requests();
        return;
    }
    ov::Tensor logits;

    {
        static ManualTimer timer("forward");
        const auto infer_start = std::chrono::steady_clock::now();
        timer.start();
        logits = m_model_runner->forward(m_requests, scheduler_output);
        const auto infer_end = std::chrono::steady_clock::now();
        m_pipeline_metrics.inference_duration = PerfMetrics::get_microsec(infer_end - infer_start);
        timer.end();
    }

#ifdef DEBUG_CACHE_STATE_DUMP
    CacheStateDumper dumper(CacheStateDumper::get_run_id_for_generation_step(step_count, "before_eviction"));
    dumper.dump_cache_state(*m_scheduler, m_requests, step_count);
#endif

    // evict unimportant blocks from KV cache, if requested
    const auto& sched_config = m_scheduler->get_config();
    if (sched_config.use_cache_eviction) {
        _maybe_evict_cache_blocks(sched_config, scheduler_output);
    }

#ifdef DEBUG_CACHE_STATE_DUMP
    CacheStateDumper dumper_after(CacheStateDumper::get_run_id_for_generation_step(step_count, "eviction"));
    dumper_after.dump_cache_state(*m_scheduler, m_requests, step_count);
    step_count++;
#endif

    // process generation_config.echo parameter
    _fill_prompt_log_probs(m_requests, logits);

    // Apply Whisper-specific logit processing (suppress tokens, timestamp forcing)
    // This mirrors what process_whisper_logits() does in whisper.cpp for each decode step.
    // The logits tensor here is [total_scheduled_tokens, 1, vocab] or [1, T, vocab] — the
    // do_suppress_tokens / process_whisper_timestamp_logits helpers look at the LAST token
    // in the sequence dimension (shape[1]-1), which is exactly position T-1 (the first free
    // token position during prefill, or the single new token during decode).
    if (m_has_whisper_config) {
        for (size_t seq_group_idx = 0, offset = 0; seq_group_idx < m_requests.size(); ++seq_group_idx) {
            auto& sg = m_requests[seq_group_idx];
            if (!sg->is_scheduled() || !sg->requires_sampling())
                continue;

            const size_t num_running = sg->num_running_seqs();
            const size_t output_seq_len = sg->get_output_seq_len();
            const size_t vocab_size = logits.get_shape().back();

            // Build a view into this sequence group's portion of the logits tensor
            const float* base = logits.data<float>() + offset * vocab_size;
            ov::Tensor sg_logits(ov::element::f32,
                                 ov::Shape{num_running, output_seq_len, vocab_size},
                                 const_cast<float*>(base));

            for (size_t batch = 0; batch < num_running; ++batch) {
                // Get already-generated tokens for this sequence (empty on first step)
                auto& running_seqs = sg->get_sequences();
                std::vector<int64_t> generated_ids;
                if (batch < running_seqs.size()) {
                    generated_ids = running_seqs[batch]->get_generated_ids();
                }
                const bool initial_step = generated_ids.empty();

                // 1) Suppress begin_suppress_tokens on the initial step only
                if (initial_step && !m_whisper_gen_config.begin_suppress_tokens.empty()) {
                    ov::genai::do_suppress_tokens(sg_logits, batch, m_whisper_gen_config.begin_suppress_tokens);
                }
                // 2) Always suppress suppress_tokens
                if (!m_whisper_gen_config.suppress_tokens.empty()) {
                    ov::genai::do_suppress_tokens(sg_logits, batch, m_whisper_gen_config.suppress_tokens);
                }
                // 3) Timestamp logit processing when return_timestamps is set
                if (m_whisper_gen_config.return_timestamps) {
                    ov::genai::process_whisper_timestamp_logits(
                        sg_logits, batch, m_whisper_gen_config, generated_ids, initial_step);
                }
            }
            offset += output_seq_len * num_running;
        }
    }

    SamplerOutput sampler_output;
    {
        static ManualTimer timer("sample");
        timer.start();
        sampler_output = m_sampler->sample(m_requests, logits, m_is_validation_mode_enabled);
        m_batch_size = sampler_output.num_generated_tokens;
        timer.end();
    }

    // process sampler_output (e.g. fork or drop sequences from BlockScheduler)
    {
        static ManualTimer free_fork_timer("fork / free sequence");
        free_fork_timer.start();

        for (const auto& pair : sampler_output.m_forked_sequences) {
            uint64_t parent_id = pair.first;
            const std::list<uint64_t>& child_ids = pair.second;
            for (auto& child_id : child_ids)
                m_scheduler->fork_sequence(parent_id, child_id);
        }

        for (auto seq_id : sampler_output.m_dropped_sequences)
            m_scheduler->free_sequence(seq_id);

        free_fork_timer.end();
    }
    
    // append embeddings for generated tokens
    if (m_model_input_type == ModelInputType::EMBEDDINGS)
        m_model_runner->append_embeddings(m_requests, scheduler_output);

    // notify requests dropped by handle
    {
        static ManualTimer report_tokens_timer("notify requests dropped by handle");
        report_tokens_timer.start();
        _notify_requests_dropped_by_handle();
        report_tokens_timer.end();
    }

    // free non running requests for current step

    {
        static ManualTimer clean_up_requests_timer("free non running requests");
        clean_up_requests_timer.start();
        _free_non_running_requests();
        clean_up_requests_timer.end();
    }

    step_timer.end();
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::set_adapters(const std::optional<AdapterConfig>& adapters) {
    if (m_adapter_controller) {
        m_adapter_controller->apply(m_model_runner->get_infer_request(), adapters);
    }
}

std::vector<EncodedGenerationResult>
ContinuousBatchingPipeline::ContinuousBatchingImpl::generate(const std::vector<ov::Tensor>& input_ids,
                                                             const std::vector<GenerationConfig>& sampling_params,
                                                             const StreamerVariant& streamer,
                                                             const std::optional<std::vector<ov::Tensor>>& token_type_ids,
                                                             const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids_list) {

    _reset_cache_usage_statistics();
    ManualTimer generate_timer("generate()");
    generate_timer.start();

    OPENVINO_ASSERT(!has_non_finished_requests(), "Generate cannot be called while ContinuousBatchingPipeline is already in running state. Use ContinuousBatchingPipeline::add_request");
    OPENVINO_ASSERT(input_ids.size() == sampling_params.size());

    if (position_ids_list.has_value()) {
        OPENVINO_ASSERT((*position_ids_list).size() == input_ids.size());
    }

    auto start_time =  std::chrono::steady_clock::now();
    PerfMetrics perf_metrics;
    auto& raw_perf_counters = perf_metrics.raw_metrics;
    raw_perf_counters.m_inference_durations =  {{ MicroSeconds(0.0f) }};

    // checks that all requests has the same LoRA adapters property value
    for (size_t i = 1; i < sampling_params.size(); ++i) {
        OPENVINO_ASSERT(sampling_params[i - 1].adapters == sampling_params[i].adapters,
            "LoRA adapters value must be the same for all requests");
    }
    set_adapters(sampling_params[0].adapters);

    const auto streamer_ptr = std::make_shared<ThreadedStreamerWrapper>(streamer, m_tokenizer);

    OPENVINO_ASSERT(!streamer_ptr->has_callback() || input_ids.size() == 1 && sampling_params[0].num_return_sequences == 1 &&
        (sampling_params[0].is_greedy_decoding() || sampling_params[0].is_multinomial()),
        "Currently streaming is possible only with batch size=1 and only for greedy or multinomial decoding");

    std::vector<GenerationHandle> generations;
    for (size_t request_id = 0; request_id < input_ids.size(); ++request_id) {
        OPENVINO_ASSERT(1 == input_ids[request_id].get_shape().at(0), "Use multiple tensors to pass a batch.");
        if (position_ids_list.has_value()) {
            const auto [position_ids, rope_delta] = (*position_ids_list)[request_id];
            m_inputs_embedder->set_position_ids(position_ids);
            if (rope_delta.has_value()) {
                m_inputs_embedder->set_rope_delta(*rope_delta);
            }
        }
        bool has_valid_token = token_type_ids.has_value() && request_id < token_type_ids->size();
        generations.push_back(
            add_request(request_id, input_ids[request_id], sampling_params[request_id], has_valid_token ? std::make_optional((*token_type_ids)[request_id]) : std::nullopt)
        );
    }

    auto all_requests = get_awaiting_requests(); // we need to store all requests to get results from them once generation has finished

    GenerationHandle& generation = generations.at(0);

    streamer_ptr->start();
    m_sampler->clear_structured_output_compile_times();
    while (has_non_finished_requests()) {
        try {
            const auto infer_start = std::chrono::steady_clock::now();
            step();
            
            // During prefill step (or steps if max_batch_size < prompt_len) we don't generate new tokens,
            // but still inference took place, so we need to add this time to the total inference duration.
            raw_perf_counters.m_inference_durations[0] += MicroSeconds(m_pipeline_metrics.inference_duration);
            if (m_batch_size > 0) {
                const auto infer_end = std::chrono::steady_clock::now();
                const auto infer_ms = PerfMetrics::get_microsec(infer_end - infer_start);
                raw_perf_counters.m_token_infer_durations.emplace_back(infer_ms);
                raw_perf_counters.m_new_token_times.emplace_back(infer_end);
                raw_perf_counters.m_batch_sizes.emplace_back(m_batch_size);
            }
        } catch (...) {
            drop_requests(); // remove all requests from pipeline state in case of exception
            streamer_ptr->end();
            std::rethrow_exception(std::current_exception());
        }
        stream_tokens(streamer_ptr, generation);
    }

    auto times = m_sampler->get_structured_output_times();
    perf_metrics.grammar_compiler_init_times = times.first;
    for (const auto& t: times.second) {
        raw_perf_counters.m_grammar_compile_times.emplace_back(t);
    }

    // waiting for completion of streaming
    streamer_ptr->end();

    OPENVINO_ASSERT(m_requests.empty(), "Internal error: current request is supposed to be dropped within step() function as completed");

    std::vector<EncodedGenerationResult> results;
    results.reserve(all_requests.size());

    for (size_t request_id = 0; request_id < all_requests.size(); ++request_id) {
        const auto& request = all_requests[request_id];
        auto sampling_params = request->get_sampling_parameters();
        const auto& sequences = request->get_finished_sequences();
        size_t num_outputs = std::min(sampling_params.num_return_sequences, sequences.size());

        EncodedGenerationResult result;
        result.m_request_id = request_id;
        result.m_generation_ids.resize(num_outputs);
        result.m_scores.resize(num_outputs);
        result.m_status = request->get_generation_stream()->get_status();

        for (size_t i = 0; i < num_outputs; ++i) {
            const auto & sequence = sequences[i];
            const float score = sampling_params.is_beam_search() ? sequence->get_beam_search_score(sampling_params) : sequence->get_cumulative_log_prob();
            const auto & generated_ids = sequence->get_generated_ids();

            if (sampling_params.echo)
                result.m_generation_ids[i] = request->get_prompt_ids();
            std::copy(generated_ids.begin(), generated_ids.end(), std::back_inserter(result.m_generation_ids[i]));
            result.m_scores[i] = score;
        }

        result.m_status = generations[request_id]->get_status();

        // The same perf metrics for each sequence, only tokenization/detokenization will differ.
        perf_metrics.raw_metrics.generate_durations.clear();
        perf_metrics.raw_metrics.generate_durations.emplace_back(PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start_time));
        perf_metrics.num_input_tokens = request->get_prompt_len();
        perf_metrics.evaluate_statistics(start_time);

        result.perf_metrics = perf_metrics;
        results.push_back(std::move(result));
    }

    OPENVINO_ASSERT(results.size() == input_ids.size());

    generate_timer.end();
    
    const auto& scheduler_config = m_scheduler->get_config();
    // Clear KV-cache in case of dynamic cache allocation and no prefix caching
    if (!scheduler_config.enable_prefix_caching && scheduler_config.cache_size == 0 && scheduler_config.num_kv_blocks == 0) {
        m_scheduler->clear_kv_cache();
    }
    return results;
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::_free_non_running_requests() {
    std::vector<SequenceGroup::Ptr>::iterator requests_iterator = m_requests.begin();
    while (requests_iterator != m_requests.end()) {
        const auto& request = *requests_iterator;
        if(request->has_finished() || request->handle_stopped() || request->handle_cancelled()) {
            for (const auto& sequence: request->get_sequences()) {
                if (m_scheduler->has_block_table(sequence->get_id())) {
                    m_scheduler->free_sequence(sequence->get_id());
                }
            }
            m_sampler->clear_request_info(request->get_request_id());
            requests_iterator = m_requests.erase(requests_iterator);
        } else {
            requests_iterator++;
        }
    }
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::_notify_requests_dropped_by_handle() {
    // Notify the last time by pushing empty output
    // This causes read() to unblock by adding anything to the queue
    for (SequenceGroup::Ptr& request : m_requests) {
        if (request->handle_stopped() || request->handle_cancelled())
            request->push_empty_outputs();
    }
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::_register_step_cache_usage(float step_cache_usage) {
    if (m_previous_step_cache_usages.size() >= AVG_CACHE_USAGE_WINDOW_SIZE_IN_STEPS) {
        m_previous_step_cache_usages.pop_front();
    }
    m_previous_step_cache_usages.push_back(step_cache_usage);
}

float ContinuousBatchingPipeline::ContinuousBatchingImpl::_get_current_running_average_cache_usage() const {
    return std::accumulate(m_previous_step_cache_usages.begin(), m_previous_step_cache_usages.end(), 0.0) / m_previous_step_cache_usages.size();
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::_reset_cache_usage_statistics() {
    m_previous_step_cache_usages.clear();
    m_pipeline_metrics.max_cache_usage = 0.0;
    m_pipeline_metrics.avg_cache_usage = 0.0;
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::drop_requests() {
    for (const std::shared_ptr<ov::genai::SequenceGroup> request : m_requests) {
        for (const auto& sequence: request->get_sequences()) {
            if (m_scheduler->has_block_table(sequence->get_id())) {
                m_scheduler->free_sequence(sequence->get_id());
            }
        }
        m_sampler->clear_request_info(request->get_request_id());
    }
    m_requests.clear();
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::_compute_cache_rotation_data(const std::vector<SequenceGroup::Ptr>& sequence_groups,
        const Scheduler::Output& scheduler_output) {
    size_t num_sequence_groups = scheduler_output.m_scheduled_sequence_groups_ids.size();
    std::map<size_t, size_t> live_seq_ids_to_num_occupied_blocks;
    for (size_t i = 0; i < num_sequence_groups; ++i) {
        size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
        SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id];
        std::vector<Sequence::CPtr> running_sequences = sequence_group->get_running_sequences();
        size_t num_running_sequences = running_sequences.size();

        for (size_t i = 0; i < num_running_sequences; ++i) {
            Sequence::CPtr sequence = running_sequences[i];
            size_t num_blocks = sequence_group->get_num_logical_blocks();
            size_t seq_id = sequence->get_id();
            OPENVINO_ASSERT(live_seq_ids_to_num_occupied_blocks.find(seq_id) == live_seq_ids_to_num_occupied_blocks.end(),
                    "duplicate seq_id ", seq_id, " among sequence groups");
            live_seq_ids_to_num_occupied_blocks[seq_id] = num_blocks;
        }
    }

    // necessary since we move from these members during previous steps
    m_current_step_rotated_block_indices_per_sequence.clear();
    m_current_step_rotated_block_indices_per_sequence.resize(m_num_decoder_layers);
    m_current_step_rotation_deltas.clear();

    std::vector<size_t> num_blocks_to_rotate_for_each_layer(m_num_decoder_layers, 0);


    for (const auto& seq_id_and_evicted_blocks : m_previous_evicted_block_logical_indices_per_sequence) {
        size_t seq_id = seq_id_and_evicted_blocks.first;
        // Skip sequences that, in the meanwhile before previous step's forward execution and now,
        // have left the cache (e.g. finished or were preempted)
        if (live_seq_ids_to_num_occupied_blocks.find(seq_id) == live_seq_ids_to_num_occupied_blocks.end()) {
            continue;
        }

        const auto& logical_blocks_to_evict = seq_id_and_evicted_blocks.second;

        for (size_t layer_idx = 0; layer_idx < logical_blocks_to_evict.size(); layer_idx++) {
            if (logical_blocks_to_evict[layer_idx].empty()) {
                continue;
            }
            size_t num_blocks_before_eviction = m_previous_num_blocks_before_eviction_per_sequence[seq_id];
            auto rotation_multipliers =
                m_cache_rotation_calculator->get_rotation_data(logical_blocks_to_evict[layer_idx],
                                                                       num_blocks_before_eviction);
            for (size_t i = 0; i < rotation_multipliers.size(); i++) {
                const auto& block_rotation_data = rotation_multipliers[i];

                m_current_step_rotated_block_indices_per_sequence[layer_idx][seq_id].push_back(
                    block_rotation_data.logical_block_idx);

                size_t block_offset = num_blocks_to_rotate_for_each_layer[layer_idx];
                auto rotation_deltas_tensor_data =
                    m_rotation_deltas_stores[layer_idx].data<int32_t>() + block_offset;
                for (size_t tok_idx = 0; tok_idx < m_block_size; tok_idx++) {
                   rotation_deltas_tensor_data[tok_idx] = block_rotation_data.rotation_delta / m_block_size;
                }
                num_blocks_to_rotate_for_each_layer[layer_idx] += 1;
            }
        }
    }
    // Select the previously filled rotation coefficients from the store tensor
    for (size_t i = 0; i < m_num_decoder_layers; i++) {
        m_current_step_rotation_deltas.emplace_back(
            m_rotation_deltas_stores[i],
            ov::Coordinate{0, 0},
            ov::Coordinate{num_blocks_to_rotate_for_each_layer[i], 1});
    }
}


void ContinuousBatchingPipeline::ContinuousBatchingImpl::_maybe_evict_cache_blocks(const SchedulerConfig& sched_config, const Scheduler::Output& scheduler_output) {
    std::unordered_map<SequenceGroup::Ptr, size_t> seq_group_to_num_blocks_evicted_map;
    const auto& sequence_attention_scores = m_model_runner->get_last_attention_scores();

    OPENVINO_ASSERT(!sequence_attention_scores.empty());
    size_t num_decoder_layers = sequence_attention_scores.begin()->second.size();

    m_previous_evicted_block_logical_indices_per_sequence.clear();
    m_previous_num_blocks_before_eviction_per_sequence.clear();

    for (auto& seq_id_and_attention_scores : sequence_attention_scores) {
        auto seq_id = seq_id_and_attention_scores.first;
        const auto& attention_scores_for_all_decoder_layers = seq_id_and_attention_scores.second;
        if (m_seq_group_id_to_cache_eviction_algo_map.find(seq_id) == m_seq_group_id_to_cache_eviction_algo_map.end()) {
            constexpr size_t MAX_POOL_WINDOW_SIZE = 7;
            m_seq_group_id_to_cache_eviction_algo_map[seq_id] = CacheEvictionAlgorithm(sched_config.cache_eviction_config, m_block_size, num_decoder_layers, MAX_POOL_WINDOW_SIZE);
        }
        auto& cache_eviction_algo = m_seq_group_id_to_cache_eviction_algo_map[seq_id];
        std::set<size_t> skip_set;
        if (scheduler_output.m_apply_sparse_attention_mask) {
            const auto& skip_map = scheduler_output.m_sparse_attention_skipped_logical_blocks;
            auto it = skip_map.find(seq_id);
            if (it != skip_map.end()) {
                skip_set = it->second;
            }
        }

        if (skip_set.empty()) {
            // For now, will only register token scores from the dense attention stages
            cache_eviction_algo.register_new_token_scores(attention_scores_for_all_decoder_layers, skip_set, scheduler_output.m_score_aggregation_windows.at(seq_id));
        }

        auto seq_group_ptr_it = std::find_if(m_requests.begin(), m_requests.end(), [seq_id](const SequenceGroup::Ptr& val) { return val->has_sequence_with_id(seq_id); });
        OPENVINO_ASSERT(seq_group_ptr_it != m_requests.end(), "could not find sequence group with sequence ", seq_id);
        auto seq_group_ptr = *seq_group_ptr_it;

         if (!seq_group_ptr->can_generate_tokens()) {
             // do not evict during prefill
             continue;
         }

        if (sched_config.cache_eviction_config.aggregation_mode == AggregationMode::ADAPTIVE_RKV) {
            const auto& block_diversities = m_model_runner->get_last_block_diversities();
            auto it = block_diversities.find(seq_id);
            if (it != block_diversities.end()) {
                cache_eviction_algo.register_block_diversity(it->second);
            }
        }

        m_previous_num_blocks_before_eviction_per_sequence[seq_id] = seq_group_ptr->get_num_logical_blocks();

        auto logical_blocks_to_evict = cache_eviction_algo.evict_logical_blocks();
        m_previous_evicted_block_logical_indices_per_sequence[seq_id] = logical_blocks_to_evict;

        m_scheduler->free_blocks_from_sequence(seq_id, logical_blocks_to_evict);

        size_t num_blocks_evicted = logical_blocks_to_evict[0].size();

        if (seq_group_to_num_blocks_evicted_map.find(seq_group_ptr) != seq_group_to_num_blocks_evicted_map.end()) {
            OPENVINO_ASSERT(seq_group_to_num_blocks_evicted_map[seq_group_ptr] == num_blocks_evicted, "internal error - each sequence in the same group must have the same number of blocks evicted");
        } else {
            seq_group_to_num_blocks_evicted_map[seq_group_ptr] = num_blocks_evicted;
        }

    }

    for (const auto& seq_group_ptr_and_num_blocks_evicted : seq_group_to_num_blocks_evicted_map) {
        // Assuming that the evicted blocks are always full (since they by design are only selected from intermediate-age blocks)
        auto seq_group_ptr = seq_group_ptr_and_num_blocks_evicted.first;
        auto num_blocks_evicted = seq_group_ptr_and_num_blocks_evicted.second;
        seq_group_ptr->register_token_eviction(num_blocks_evicted * m_block_size);
    }
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::_set_adaptive_rkv_diversity_blocks(const SchedulerConfig& sched_config, const Scheduler::Output& scheduler_output) {
    // TODO(vshampor): implement
}



void ContinuousBatchingPipeline::ContinuousBatchingImpl::_fill_prompt_log_probs(std::vector<SequenceGroup::Ptr>& sequence_groups, ov::Tensor& logits) {
    const float * logits_data = logits.data<float>();
    ov::Shape logits_shape = logits.get_shape();
    OPENVINO_ASSERT(logits_shape.size() == 3);
    size_t vocab_size = logits_shape[2];
    for (size_t sequence_group_id = 0, currently_processed_tokens = 0; sequence_group_id < sequence_groups.size(); ++sequence_group_id) {
        SequenceGroup::Ptr sequence_group = sequence_groups[sequence_group_id];
        // requests not scheduled, in decoding phase or not echoing are not processed
        if (!sequence_group->is_scheduled() || sequence_group->get_context_len() > sequence_group->get_prompt_len() ||
            !sequence_group->get_sampling_parameters().echo)
            continue;

        size_t num_running_sequences = sequence_group->num_running_seqs();
        OPENVINO_ASSERT(num_running_sequences == 1);
        size_t output_seq_len = sequence_group->get_output_seq_len();

        const float * sequence_group_logits_data = logits_data + vocab_size * currently_processed_tokens;

        size_t num_prompt_tokens_processed = sequence_group->get_num_processed_tokens();
        OPENVINO_ASSERT(num_prompt_tokens_processed + output_seq_len <= sequence_group->get_prompt_len());

        // if we processed the whole prompt we don't include last logprob as it will be processed by the sampler (it's already completion)
        // otherwise we include it as it will be used in the next part of the prompt
        int exclude_last_logprob = 1;
        if (num_prompt_tokens_processed + output_seq_len < sequence_group->get_prompt_len())
            exclude_last_logprob = 0;

        // if we start processing the prompt we add "fake" log prob for the first position (begin of sequence)
        if (num_prompt_tokens_processed == 0)
            sequence_group->append_prompt_log_prob(1.0);

        for (int token_logits_offset = 0, token_id_offset = num_prompt_tokens_processed + 1;
             token_logits_offset < output_seq_len - exclude_last_logprob;
             token_logits_offset++, token_id_offset++) {

            const float* token_logits = (sequence_group_logits_data + token_logits_offset * vocab_size);
            int64_t token_id = sequence_group->get_prompt_ids()[token_id_offset];
            float token_logit = token_logits[token_id];

            // find max value for log softmax
            float max_value = -std::numeric_limits<float>::infinity();
            size_t max_index = 0;
            for (size_t i = 0; i < vocab_size; ++i) {
                if (token_logits[i] > max_value) {
                    max_value = token_logits[i];
                    max_index = i;
                }
            }

            // apply log softmax to token logit
            float log_sum = std::log(std::accumulate(
                token_logits, token_logits + vocab_size, 0.0f, [max_value](float accumulated, float to_add) {
                    return accumulated + std::exp(to_add - max_value);
            }));

            sequence_group->append_prompt_log_prob(token_logit - max_value - log_sum);
        }
        currently_processed_tokens += output_seq_len * num_running_sequences;
        // For max_new_tokens == 0, we don't reach sampling so need to notify handle separately
        if(sequence_group->get_max_new_tokens() == 0) {
            sequence_group->notify_handle_echo_only();
        }
    }
}

std::vector<SequenceGroup::Ptr> ContinuousBatchingPipeline::ContinuousBatchingImpl::get_awaiting_requests() {
    std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
    return m_awaiting_requests;
}
} // namespace ov::genai
