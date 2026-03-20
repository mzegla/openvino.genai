// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <chrono>
#include <iomanip>
#include <thread>
#include <optional>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <queue>
#include <unordered_set>
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
#include "openvino/op/multiply.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/transpose.hpp"
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

// Returns true when OPENVINO_LOG_LEVEL > 1 (cached after first call).
static bool cb_verbose() {
    static const bool val = []() -> bool {
        const char* e = std::getenv("OPENVINO_LOG_LEVEL");
        return e && std::atoi(e) > 1;
    }();
    return val;
}

// ---------------------------------------------------------------------------
// extract_cross_attn_projector()
//
// Called AFTER the ReadValue bypass has wired each cross-attn SDPA's K and V
// inputs directly to their projection subgraph outputs (MatMul → Reshape →
// Transpose chains that consume encoder_hidden_states).
//
// This function:
//  1. Collects the K/V projection chain output (kv_src) for each cross-attn layer.
//  2. Wraps those outputs in Result nodes and builds a separate projector ov::Model
//     with encoder_hidden_states as its single input.
//  3. Replaces the K/V sources in the decoder with new Parameters named
//     "cross_kv_K_{l}" / "cross_kv_V_{l}" so the decoder no longer runs the
//     K/V projection on every infer() call.
//  4. Removes encoder_hidden_states from the decoder (it has no more consumers).
//
// Returns nullptr if CROSS_KV_CACHE=0 env var is set or extraction fails.
// ---------------------------------------------------------------------------
static std::shared_ptr<ov::Model> extract_cross_attn_projector(std::shared_ptr<ov::Model>& model) {
    using namespace ov::op;

    // Honour kill-switch
    const char* kv_env = std::getenv("CROSS_KV_CACHE");
    if (kv_env && std::string(kv_env) == "0") {
        if (cb_verbose()) std::cout << "[ProjectorExtract] CROSS_KV_CACHE=0 — skipping extraction\n";
        return nullptr;
    }

    // 1. Find the encoder_hidden_states Parameter — needed before SDPA detection.
    std::shared_ptr<v0::Parameter> enc_param;
    for (const auto& param : model->get_parameters()) {
        // Check tensor names first (StatefulToStateless renames the node)
        for (const auto& tname : param->get_output_tensor(0).get_names()) {
            if (tname.find("encoder_hidden_states") != std::string::npos) {
                enc_param = param;
                break;
            }
        }
        if (enc_param) break;
        // Fallback: check friendly name
        if (param->get_friendly_name().find("encoder_hidden_states") != std::string::npos) {
            enc_param = param;
            break;
        }
    }
    if (!enc_param) {
        std::cout << "[ProjectorExtract] encoder_hidden_states not found — skipping\n";
        return nullptr;
    }

    // 2. Collect K/V source outputs for each cross-attn layer, sorted by index.
    //    Use structural connectivity: find all SDPA nodes where K (input 1) or V (input 2)
    //    is reachable from encoder_hidden_states.  This is robust to any naming convention.
    std::map<int, std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>> layer_kv;
    {
        // BFS forward from enc_param to collect all nodes reachable through its outputs.
        std::unordered_set<ov::Node*> enc_descendants;
        std::queue<ov::Node*> bfs_q;
        bfs_q.push(enc_param.get());
        while (!bfs_q.empty()) {
            ov::Node* n = bfs_q.front(); bfs_q.pop();
            if (!enc_descendants.insert(n).second) continue;  // already visited
            for (size_t oi = 0; oi < n->get_output_size(); ++oi)
                for (const auto& tgt : n->output(oi).get_target_inputs())
                    bfs_q.push(tgt.get_node());
        }

        int seq_idx = 0;  // sequential fallback when name-based extraction fails
        for (const auto& op : model->get_ordered_ops()) {
            if (!ov::is_type<v13::ScaledDotProductAttention>(op)) continue;
            if (op->get_input_size() < 3) continue;

            // The Q input (0) comes from the decoder; K (1) and V (2) from the encoder.
            bool k_from_enc = enc_descendants.count(op->input(1).get_source_output().get_node()) > 0;
            bool v_from_enc = enc_descendants.count(op->input(2).get_source_output().get_node()) > 0;
            if (!k_from_enc && !v_from_enc) continue;

            // Extract layer index from node name; fallback to sequential counter.
            int layer_idx = seq_idx;
            const std::string& name = op->get_friendly_name();
            auto pos_l = name.find("layers.");
            if (pos_l != std::string::npos) {
                try { layer_idx = std::stoi(name.substr(pos_l + 7)); } catch (...) {}
            }

            std::cout << "[ProjectorExtract] cross-attn SDPA: " << name
                      << "  layer_idx=" << layer_idx << "\n";
            layer_kv[layer_idx] = { op->input(1).get_source_output(),
                                     op->input(2).get_source_output() };
            ++seq_idx;
        }
    }

    if (layer_kv.empty()) {
        std::cout << "[ProjectorExtract] No cross-attn layers found — skipping\n";
        return nullptr;
    }
    std::cout << "[ProjectorExtract] Found " << layer_kv.size() << " cross-attn layers\n";

    // Snapshot each SDPA K/V target inputs BEFORE we add Result nodes (which would
    // also appear as consumers and must not be redirected to the new Parameters).
    std::map<int, std::pair<std::vector<ov::Input<ov::Node>>,
                            std::vector<ov::Input<ov::Node>>>> layer_targets;
    for (auto& [layer_idx, kv] : layer_kv) {
        std::vector<ov::Input<ov::Node>> K_tgts, V_tgts;
        for (auto& inp : kv.first.get_target_inputs())  K_tgts.push_back(inp);
        for (auto& inp : kv.second.get_target_inputs()) V_tgts.push_back(inp);
        layer_targets[layer_idx] = { std::move(K_tgts), std::move(V_tgts) };
    }

    // 3. Build projector model: Result nodes wrapping each K/V source.
    //
    // Implementation notes:
    //   - Clone enc_param → proj_enc_param with a STATIC shape [1, T_enc, D].
    //   - Move all K/V chain consumers (MatMul nodes) from enc_param to proj_enc_param.
    //     enc_param now has zero consumers so remove_parameter always succeeds.
    //   - Register proj_enc_param IN the decoder model too (model->add_parameters).
    //     OV CPU creates Node objects for ALL operations during compile_model (even dead
    //     ones) and validates their shapes.  If proj_enc_param is unregistered, OV uses
    //     a dynamic pseudo-shape for it, causing Reshape_229018 to fail the static-shape
    //     check at compile time.  Registering it with a concrete static shape avoids this.
    //   - At runtime, proj_enc_param (decoded name "encoder_hidden_states") receives a
    //     zero dummy tensor from model_runner; the dead K/V chain is never executed.

    // Determine the static encoder size from enc_param's partial shape.
    // StatefulToStateless may have already made dim[0] dynamic; fix it to 1.
    ov::PartialShape proj_ps = enc_param->get_partial_shape();
    if (proj_ps.rank().is_static() && proj_ps.rank().get_length() == 3)
        proj_ps[0] = 1;

    auto proj_enc_param = std::make_shared<v0::Parameter>(
        enc_param->get_element_type(), proj_ps);
    proj_enc_param->set_friendly_name("encoder_hidden_states");
    proj_enc_param->get_output_tensor(0).set_names(
        enc_param->get_output_tensor(0).get_names());

    // Move all enc_param consumers in the decoder K/V chains to proj_enc_param.
    {
        std::vector<ov::Input<ov::Node>> enc_targets;
        for (auto& tgt : enc_param->output(0).get_target_inputs())
            enc_targets.push_back(tgt);
        for (auto& tgt : enc_targets)
            tgt.replace_source_output(proj_enc_param->output(0));
    }
    // enc_param now has zero consumers → safe to remove from decoder.

    ov::ResultVector proj_results;
    for (auto& [layer_idx, kv] : layer_kv) {
        const std::string name_K = "cross_proj_K_" + std::to_string(layer_idx);
        const std::string name_V = "cross_proj_V_" + std::to_string(layer_idx);

        auto res_K = std::make_shared<v0::Result>(kv.first);
        auto res_V = std::make_shared<v0::Result>(kv.second);
        res_K->set_friendly_name(name_K);
        res_V->set_friendly_name(name_V);
        res_K->get_output_tensor(0).set_names({name_K});
        res_V->get_output_tensor(0).set_names({name_V});
        proj_results.push_back(res_K);
        proj_results.push_back(res_V);
    }

    auto projector = std::make_shared<ov::Model>(proj_results, ov::ParameterVector{proj_enc_param});
    projector->validate_nodes_and_infer_types();

    if (cb_verbose()) {
        std::cout << "[ProjectorExtract] Projector model: "
                  << proj_results.size() << " outputs (" << layer_kv.size() << " layers)\n";
        for (const auto& out : projector->outputs()) {
            for (const auto& n : out.get_names())
                std::cout << "  " << n << "  shape=" << out.get_partial_shape() << "\n";
        }
    }

    // 4. Replace K/V source outputs in the decoder with new Parameters.
    for (auto& [layer_idx, kv] : layer_kv) {
        auto K_ps = kv.first.get_partial_shape();
        auto V_ps = kv.second.get_partial_shape();
        // Make the batch (slot) dimension dynamic so the decoder accepts any N
        if (K_ps.rank().is_static()) K_ps[0] = ov::Dimension::dynamic();
        if (V_ps.rank().is_static()) V_ps[0] = ov::Dimension::dynamic();

        const std::string name_K = "cross_kv_K_" + std::to_string(layer_idx);
        const std::string name_V = "cross_kv_V_" + std::to_string(layer_idx);

        auto param_K = std::make_shared<v0::Parameter>(kv.first.get_element_type(),  K_ps);
        auto param_V = std::make_shared<v0::Parameter>(kv.second.get_element_type(), V_ps);
        param_K->set_friendly_name(name_K);
        param_V->set_friendly_name(name_V);
        param_K->get_output_tensor(0).set_names({name_K});
        param_V->get_output_tensor(0).set_names({name_V});

        // Redirect only the pre-captured SDPA targets (not the projector Result nodes)
        for (auto& tgt : layer_targets[layer_idx].first)
            tgt.replace_source_output(param_K->output(0));
        for (auto& tgt : layer_targets[layer_idx].second)
            tgt.replace_source_output(param_V->output(0));

        model->add_parameters({param_K, param_V});
        if (cb_verbose())
            std::cout << "[ProjectorExtract] Layer " << layer_idx
                      << ": decoder params " << name_K << " + " << name_V << " added\n";
    }

    // 5. Remove the original enc_param from the decoder.
    //
    // After step 3, all enc_param consumers in the decoder were moved to proj_enc_param.
    // After step 4, all SDPA K/V inputs were redirected to the fresh cross_kv_K/V params.
    // The K/V projection chain (proj_enc_param → MatMul → Reshape → Transpose → proj_results)
    // is part of the PROJECTOR model only, not the decoder model.  OV's get_ordered_ops()
    // traverses backward from decoder Results — it never reaches proj_enc_param's chain,
    // so those nodes are invisible to the decoder's compile_model and do not need explicit
    // removal.  We only need to drop enc_param itself (zero consumers now).
    //
    // Note: we intentionally do NOT call ov::replace_node(proj_enc_param, zero_const):
    //   - proj_enc_param is shared with the projector model — replacing it would break the
    //     projector's input.
    //   - The dead K/V chain is already outside the decoder model's op-set; no DCE needed.
    try {
        model->remove_parameter(enc_param);
        if (cb_verbose())
            std::cout << "[ProjectorExtract] Original encoder_hidden_states removed from decoder.\n";
    } catch (const std::exception& e) {
        std::cout << "[ProjectorExtract] WARNING: could not remove enc_param: " << e.what() << "\n";
    }

    model->validate_nodes_and_infer_types();
    if (cb_verbose())
        std::cout << "[ProjectorExtract] Done — decoder now has "
                  << 2 * layer_kv.size() << " cross-KV parameter inputs\n\n";

    return projector;
}

// Helper function to apply selective PA transformation (self-attention only)
std::shared_ptr<ov::Model> apply_selective_pa_transformation(std::shared_ptr<ov::Model> model) {
    using namespace ov::op;

    std::vector<std::shared_ptr<ov::Node>> self_attn_nodes;
    std::vector<std::shared_ptr<ov::Node>> cross_attn_nodes;

    // Find all ScaledDotProductAttention nodes and classify them
    if (cb_verbose()) std::cout << "Analyzing SDPA nodes for selective PA transformation...\n";
    for (const auto& node : model->get_ops()) {
        if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(node)) {
            std::string node_name = node->get_friendly_name();
            if (node_name.find("encoder_attn") != std::string::npos || 
                node_name.find("cross_attn") != std::string::npos) {
                cross_attn_nodes.push_back(node);
                if (cb_verbose()) std::cout << "  Cross-attention SDPA: " << node_name << "\n";
            } else {
                self_attn_nodes.push_back(node);
                if (cb_verbose()) std::cout << "  Self-attention SDPA: " << node_name << "\n";
            }
        }
    }

    if (cb_verbose()) std::cout << "Found " << self_attn_nodes.size() << " self-attention SDPA nodes\n";
    if (cb_verbose()) std::cout << "Found " << cross_attn_nodes.size() << " cross-attention SDPA nodes\n\n";

    if (self_attn_nodes.empty()) {
        if (cb_verbose()) std::cout << "No self-attention SDPA nodes to transform\n";
        return nullptr;
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

    // ROOT CAUSE FIX: Whisper exports decoder self-attention as SDPA with is_causal=false
    // plus an explicit triangular mask tensor.  SDPAToPagedAttention honours is_causal=false
    // and produces a PA node that relies on the mask, which the PA kernel does not receive
    // (and does not accept as an input).  The resulting attention is non-causal during
    // prefill, producing completely wrong ("." loop) outputs.  Replace each self-attention
    // SDPA with an is_causal=true node so the pass creates a correctly-causal PA node.
    //
    // NOTE: PagedAttentionExtension itself has no "is_causal" attribute.  Causality during
    // multi-token PROMPT prefill is applied internally by the kernel based on token
    // positions.  There is a known numerical difference between the kernel's PROMPT path
    // (M>1 tokens processed together) and GENERATE path (M=1 tokens, one at a time): due
    // to different accumulation order / block-boundary reductions, the floating-point
    // results differ by small amounts that accumulate over ~6 decode steps and eventually
    // flip the top-1 prediction from the correct timestamp to a wrong EOT.  The workaround
    // is to set max_num_batched_tokens=1 in SchedulerConfig so every token goes through the
    // numerically stable GENERATE path (see whisper_speech_recognition.cpp).
    // ROOT CAUSE FIX (revised): The previous version kept the explicit triangular mask
    // wired into the causal SDPA node.  Per the OV SDPA v13 spec, is_causal=true PLUS a
    // mask input adds the mask as an additional attention bias on top of the built-in
    // triangular mask — i.e. double-masking.  Whisper's original mask has −∞ in the upper
    // triangle, so the double-masked positions receive doubly-negative logits before
    // softmax.  This subtly changes the softmax denominator for every row vs the clean
    // is_causal-only path, producing the PROMPT-path floating-point drift that causes
    // ts_prob to drop from ~51% (correct) to ~14.6% (below the forcing threshold) with
    // M>1 batched prefill.  Fix: always drop the mask and use the 3-input form so the
    // resulting PA node sees pure causal attention with no extra masking.
    // NOTE: An is_causal=true pre-patch was attempted here to address M>1 PROMPT-path
    // failures, but all forms of ov::replace_node(sa_node, causal_sdpa) proved harmful:
    //   - Keeping original mask: double-masking → FP drift → early EOT with M>1
    //   - Dropping mask (3-input): beam_idx becomes disconnected → SDPAToPagedAttention
    //     throws "undeclared parameters: beam_idx"
    //   - Zeroed mask via Multiply: Multiply node in mask path breaks StateManagementPattern
    //     matching for ≥1 self-attention SDPA node, leaving it unconverted → loop even M=1
    // Additionally, SDPAToPagedAttention ignores is_causal entirely (confirmed from source).
    // The pre-patch is therefore a no-op at best, graph-breaking at worst.  It is removed.
    //
    // M>1 PROMPT-path failure ("."-loop) root cause: the PagedAttention PROMPT path
    // (q_cnt>1, batched GEMM) and GENERATE path (q_cnt=1, sequential GEMM) accumulate
    // floating-point in different orders, producing slightly different K/V values in the
    // KV-cache blocks.  These differences cascade over ~6 decode steps and flip the top-1
    // prediction from the correct timestamp token to wrong ones.
    //
    // SOLUTION: Set SchedulerConfig::max_num_tokens_per_prefill_step=1 to force all
    // prefill tokens through single-token steps (q_cnt=1 → GENERATE path numerics),
    // while still allowing GENERATE-phase tokens from multiple concurrent requests to be
    // batched together up to max_num_batched_tokens.  See whisper_speech_recognition.cpp.

    // Apply SDPAToPagedAttention transformation to entire model
    if (cb_verbose()) std::cout << "Applying SDPAToPagedAttention transformation...\n";
    ov::pass::SDPAToPagedAttention(false, false, false, false, false, false).run_on_model(model);
    if (cb_verbose()) std::cout << "Transformation complete!\n\n";

    // Debug: print model parameters AFTER transformation to see what StatefulToStateless created
    if (cb_verbose()) std::cout << "=== MODEL PARAMETERS AFTER SDPAToPagedAttention ===\n";
    for (const auto& param : model->get_parameters()) {
        if (cb_verbose()) std::cout << "  PARAM: " << param->get_friendly_name() 
                  << " shape=" << param->get_output_partial_shape(0) << "\n";
    }
    // Debug: print all PA nodes and their Q/K/V input sources.
    // NOTE: PagedAttentionExtension has NO is_causal attribute — causality is hardcoded
    // into the kernel (applies a lower-triangular mask for multi-token / PROMPT mode).
    // The sliding_window input (index 10) = 0 for standard causal LLMs (no sliding window).
    if (cb_verbose()) std::cout << "=== PA NODES AFTER TRANSFORMATION ===\n";
    int pa_count = 0;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_info().name == std::string("PagedAttentionExtension")) {
            ++pa_count;
            if (cb_verbose()) std::cout << "  PA[" << pa_count << "]: " << op->get_friendly_name()
                      << "  num_inputs=" << op->get_input_size() << "\n";
            for (size_t i = 0; i < std::min(op->get_input_size(), size_t(3)); ++i) {
                auto src = op->input(i).get_source_output();
                if (cb_verbose()) std::cout << "    input[" << i << "]: " << src.get_node()->get_friendly_name()
                          << " type=" << src.get_node()->get_type_info().name << "\n";
            }
            // Print sliding_window value (index 10) if it's a constant
            if (op->get_input_size() > 10) {
                auto sw_src = op->input(10).get_source_output();
                if (auto sw_const = ov::as_type_ptr<ov::op::v0::Constant>(sw_src.get_node_shared_ptr())) {
                    auto sw_val = sw_const->cast_vector<int32_t>();
                    if (cb_verbose()) std::cout << "    sliding_window (input[10]): " << (sw_val.empty() ? "empty" : std::to_string(sw_val[0])) << "\n";
                }
            }
        }
    }
    if (cb_verbose()) std::cout << "Total PA nodes: " << pa_count << "\n";
    if (cb_verbose()) std::cout << "=================================================\n\n";

    // Debug: inspect cross-attention info after transformation
    if (cb_verbose()) std::cout << "=== CROSS-ATTN INFO AFTER TRANSFORMATION ===\n";
    for (const auto& info : cross_attn_info) {
        if (cb_verbose()) std::cout << "  CrossAttn: " << info.friendly_name << "\n";
        // Is the original SDPA node still in the model?
        bool node_in_model = false;
        for (const auto& op : model->get_ops()) {
            if (op.get() == info.node.get()) { node_in_model = true; break; }
        }
        if (cb_verbose()) std::cout << "    original SDPA node still in model: " << (node_in_model ? "YES" : "NO") << "\n";
        for (size_t i = 0; i < info.inputs.size(); ++i) {
            auto* inp_node = info.inputs[i].get_node();
            bool in_model = false;
            for (const auto& op : model->get_ops()) {
                if (op.get() == inp_node) { in_model = true; break; }
            }
            if (cb_verbose()) std::cout << "    input[" << i << "]: " << inp_node->get_friendly_name()
                      << " type=" << inp_node->get_type_info().name
                      << " in_model=" << (in_model ? "YES" : "NO") << "\n";
        }
    }
    // Debug: print remaining SDPA nodes (cross-attn should still be here)
    if (cb_verbose()) std::cout << "=== SDPA NODES AFTER TRANSFORMATION ===\n";
    for (const auto& op : model->get_ops()) {
        if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(op)) {
            if (cb_verbose()) std::cout << "  SDPA: " << op->get_friendly_name() << "\n";
            for (size_t i = 0; i < op->get_input_size(); ++i) {
                auto src = op->input(i).get_source_output();
                if (cb_verbose()) std::cout << "    input[" << i << "]: " << src.get_node()->get_friendly_name()
                          << " type=" << src.get_node()->get_type_info().name << "\n";
            }
        }
    }
    if (cb_verbose()) std::cout << "============================================\n\n";

    // ---- Bypass cross-attention ReadValue/Assign state nodes ----
    // After SDPAToPagedAttention (including StatefulToStateless), the cross-attention K/V
    // nodes are now plain v6::ReadValue nodes with an init subgraph (K/V projection from
    // encoder_hidden_states). Cross-attention K/V never change during autoregression, but
    // the stateful ReadValue would read zero on step 0 (state uninitialized) and only
    // produce correct output from step 1 onwards (after the first Assign has run).
    // Fix: replace each ReadValue directly with its init subgraph output so that
    // cross-attention always computes from the live encoder_hidden_states input.
    {
        if (cb_verbose()) std::cout << "Bypassing cross-attention ReadValue/Assign state nodes...\n";
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

                if (cb_verbose()) std::cout << "  ReadValue '" << src_node->get_friendly_name()
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
            if (cb_verbose()) std::cout << "  Assign '" << sink->get_friendly_name() << "' removed\n";
        }
        if (cb_verbose()) std::cout << "  Cross-attention bypass complete ("
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
                if (cb_verbose()) std::cout << "  lm_head fix: node before logits result is NOT a MatMul ("
                          << lm_node->get_type_name() << "), skipping\n";
                break;
            }
            if (cb_verbose()) std::cout << "  lm_head fix: found MatMul '"
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
            if (cb_verbose()) std::cout << "  lm_head fix: A input from '"
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
                            if (cb_verbose()) std::cout << "  lm_head fix: WARNING – weight rank != 2, skipping\n";
                            break;
                        }
                        // Weight shape: [V, H] with transpose_b=true  →  keep [V, H], cast to f32
                        size_t V = orig_shape[0], H = orig_shape[1];
                        auto fp32_data = c->cast_vector<float>();
                        auto fp32_w = std::make_shared<ov::op::v0::Constant>(
                            ov::element::f32, orig_shape, fp32_data);
                        fp32_w->set_friendly_name(c->get_friendly_name() + "_fp32");
                        if (cb_verbose()) std::cout << "  lm_head fix: weight [" << V << "," << H << "] cast to f32\n";

                        // 3. Flatten activation: [bs, T, H] → [bs*T, H]
                        auto flatten_shape_const = std::make_shared<ov::op::v0::Constant>(
                            ov::element::i64, ov::Shape{2},
                            std::vector<int64_t>{-1, static_cast<int64_t>(H)});
                        auto A_flat = std::make_shared<ov::op::v1::Reshape>(
                            A, flatten_shape_const->output(0), /*special_zero=*/false);
                        A_flat->set_friendly_name("lm_head_A_flat");
                        if (cb_verbose()) std::cout << "  lm_head fix: flatten reshape [bs,T," << H << "] → [bs*T," << H << "]\n";

                        // 4. MatMul [bs*T, H] x [V, H]^T → [bs*T, V]  (original transpose_b=true)
                        auto new_mm = std::make_shared<ov::op::v0::MatMul>(
                            A_flat, fp32_w->output(0),
                            /*transpose_a=*/false, /*transpose_b=*/true);
                        new_mm->set_friendly_name(lm_node->get_friendly_name() + "_flat");
                        if (cb_verbose()) std::cout << "  lm_head fix: new MatMul [bs*T," << H << "] x [" << V << "," << H << "]^T\n";

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
                        if (cb_verbose()) std::cout << "  lm_head fix: unflatten reshape [bs*T," << V << "] → [bs,T," << V << "]\n";

                        // 6. Replace old lm_node with mm_unflat for all consumers
                        auto old_names = lm_node->output(0).get_tensor().get_names();
                        mm_unflat->output(0).get_tensor().set_names(old_names);
                        lm_node->output(0).get_tensor().set_names({});
                        ov::replace_node(lm_node, mm_unflat);
                        if (cb_verbose()) std::cout << "  lm_head fix: ov::replace_node → flatten/unflatten done\n";

                        new_matmul_done = true;
                        break;
                    }
                    if (wn->get_input_size() == 0) break;
                    w = wn->input(0).get_source_output();
                }
                if (!new_matmul_done)
                    if (cb_verbose()) std::cout << "  lm_head fix: WARNING – could not locate weight Constant\n";
            }
            break;
        }
        // Add deferred debug results outside the get_results() iterator
        for (auto& dr : deferred_debug_results)
            model->add_results({dr});
        // Force OV to re-validate topology and infer shapes after all graph edits
        model->validate_nodes_and_infer_types();
        if (cb_verbose()) std::cout << "  lm_head fix: model validated and shape inference refreshed\n";

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
                        if (cb_verbose()) std::cout << "  K: using original graph input (still in model)\n";
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
                        if (cb_verbose()) std::cout << "  K: using PA node input (original was orphaned)\n";
                    }
                    if (info.inputs.size() > 2 && is_in_model_graph(info.inputs[2])) {
                        v_input = info.inputs[2];
                        if (cb_verbose()) std::cout << "  V: using original graph input (still in model)\n";
                    } else {
                        if (node->get_input_size() > 2) {
                            v_input = find_encoder_proj(node->input(2).get_source_output());
                        } else {
                            v_input = info.inputs[2]; // fallback even if orphaned
                        }
                        if (cb_verbose()) std::cout << "  V: using PA node input (original was orphaned)\n";
                    }

                    // Q input for restored cross-attention SDPA:
                    // info.inputs[0] was captured BEFORE SDPAToPagedAttention.  After the
                    // transformation, the self-attention SDPA that fed Q was replaced by a PA
                    // node, making info.inputs[0] point to a dead/orphaned node whose output
                    // is garbage.  Instead, get Q from the live cross-attention PA node's
                    // input(0), which was correctly updated by ov::replace_node.
                    auto q_input = node->input(0).get_source_output();
                    if (cb_verbose()) std::cout << "  Q: from live PA input(0): '"
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
                    if (cb_verbose()) std::cout << "  Restored: " << new_sdpa->get_friendly_name() << "\n";
                    break;
                }
            }
        }
    }

    // ---- Extract cross-attention K/V projector (Option A: slot buffer) ----
    // Must run AFTER ReadValue bypass (K/V wired to projection chains) and
    // AFTER cross-attention SDPA restoration, but BEFORE segmented CA so
    // that the new cross_kv_K/V Parameters become the K/V sources in the
    // segmented SDPA nodes that are about to be created.
    std::shared_ptr<ov::Model> projector_model = extract_cross_attn_projector(model);

    // ---- Segmented cross-attention: per-sequence encoder K/V routing ----
    // With max_num_tokens_per_prefill_step=1, every scheduled sequence contributes exactly
    // 1 token per step (both PROMPT and GENERATE). So total_tokens == N (num sequences).
    // Cross-attention Q in flat batch: [1, n_heads, N, head_dim].
    // We need token i to attend to encoder K/V for sequence i, not a single broadcast state.
    //
    // Technique (no custom op):
    //   Transpose Q perm [2,1,0,3]: [1, n_heads, N, head_dim] → [N, n_heads, 1, head_dim]
    //   Now batched SDPA sees batch=N, each sequence has 1 query attending to T_enc keys.
    //   K/V projections on encoder_hidden_states [N, T_enc, D] → [N, n_heads, T_enc, head_dim].
    //   Transpose output back (same perm, self-inverse): [N, n_heads, 1, head_dim] → [1, n_heads, N, head_dim].
    //
    // model_runner.hpp builds encoder_hidden_states as [N, T_enc, D] per scheduled sequence.
    //
    // Set SKIP_SEGMENTED_CA=1 to bypass this block for comparison testing.
    const bool skip_segmented_ca = (std::getenv("SKIP_SEGMENTED_CA") != nullptr &&
                                    std::string(std::getenv("SKIP_SEGMENTED_CA")) == "1");
    if (skip_segmented_ca)
        if (cb_verbose()) std::cout << "SKIP_SEGMENTED_CA=1: skipping segmented cross-attention transform.\n";
    if (!skip_segmented_ca) {
        if (cb_verbose()) std::cout << "=== SEGMENTED CROSS-ATTENTION TRANSFORMATION ===\n";

        // 1. Snapshot live ops BEFORE changing any parameter shapes.
        //    IMPORTANT: set_partial_shape() on encoder_hidden_states must happen AFTER this
        //    snapshot and AFTER all replace_node() calls.  Calling set_partial_shape() first
        //    causes OV to immediately propagate the dynamic batch dim through every
        //    downstream node (K/V projection Reshapes, MatMuls, etc.).  Some of those nodes
        //    are Constant nodes whose data buffers OV tries to reallocate from the new
        //    (dynamic → SIZE_MAX) shape, throwing "cannot create std::vector larger than
        //    max_size()".  By deferring the shape change to just before validate(), we avoid
        //    premature propagation entirely.
        //
        //    CRITICAL: bind get_ordered_ops() to a named variable BEFORE calling .begin()
        //    and .end().  Writing:
        //        vector<...> v(model->get_ordered_ops().begin(),
        //                      model->get_ordered_ops().end())
        //    calls get_ordered_ops() TWICE, producing two different temporary NodeVectors.
        //    .begin() comes from T1 and .end() comes from T2 — iterating between them is
        //    undefined behaviour and walks off the end of T1 into garbage memory → SIGSEGV.
        auto ops_snap = model->get_ordered_ops();
        const std::vector<int64_t> perm{2, 1, 0, 3};
        int seg_count = 0;
        for (const auto& op : ops_snap) {
            if (!ov::is_type<v13::ScaledDotProductAttention>(op)) continue;
            const std::string& name = op->get_friendly_name();
            if (name.find("encoder_attn") == std::string::npos &&
                name.find("cross_attn")   == std::string::npos) continue;

            // Transpose Q: [1, n_heads, N, head_dim] → [N, n_heads, 1, head_dim]
            auto perm_pre = v0::Constant::create(ov::element::i64, ov::Shape{4}, perm);
            auto q_t = std::make_shared<v1::Transpose>(
                op->input(0).get_source_output(), perm_pre);
            q_t->set_friendly_name(name + "/q_seq_t");

            // New SDPA forwarding ALL original inputs (Q, K, V, optional mask, optional scale).
            // CRITICAL: the original cross-attention SDPA may have 5 inputs:
            //   input[3] = attention mask (ConvertLike, encoder_attn-specific)
            //   input[4] = explicit scale  (ConvertLike, shared from self_attn layers.0)
            // Dropping input[4] causes OV to fall back to 1/sqrt(head_dim) which differs
            // from the model's own scale, breaking all N including N=1.
            // Strategy: replace only Q (input 0) with q_t; pass inputs 1..N-1 unchanged.
            std::shared_ptr<v13::ScaledDotProductAttention> new_sdpa;
            const size_t n_inputs = op->get_input_size();
            if (n_inputs >= 5) {
                new_sdpa = std::make_shared<v13::ScaledDotProductAttention>(
                    q_t->output(0),
                    op->input(1).get_source_output(),
                    op->input(2).get_source_output(),
                    op->input(3).get_source_output(),  // attn_mask
                    op->input(4).get_source_output(),  // scale
                    /*causal=*/false);
            } else if (n_inputs == 4) {
                new_sdpa = std::make_shared<v13::ScaledDotProductAttention>(
                    q_t->output(0),
                    op->input(1).get_source_output(),
                    op->input(2).get_source_output(),
                    op->input(3).get_source_output(),  // attn_mask or scale
                    /*causal=*/false);
            } else {
                new_sdpa = std::make_shared<v13::ScaledDotProductAttention>(
                    q_t->output(0),
                    op->input(1).get_source_output(),
                    op->input(2).get_source_output(),
                    /*causal=*/false);
            }
            if (cb_verbose()) std::cout << "  [seg] n_inputs=" << n_inputs
                      << " -> forwarding " << (n_inputs - 1) << " non-Q inputs\n";
            new_sdpa->set_friendly_name(name + "/seg");

            // Transpose output back: [N, n_heads, 1, head_dim] → [1, n_heads, N, head_dim]
            auto perm_post = v0::Constant::create(ov::element::i64, ov::Shape{4}, perm);
            auto out_t = std::make_shared<v1::Transpose>(new_sdpa->output(0), perm_post);
            out_t->set_friendly_name(name + "/out_seq_t");

            // Transfer tensor names so downstream references still resolve
            out_t->output(0).get_tensor().set_names(
                op->output(0).get_tensor().get_names());
            op->output(0).get_tensor().set_names({});

            ov::replace_node(op, out_t);
            if (cb_verbose()) std::cout << "  [" << ++seg_count << "] Segmented: '" << name << "'\n";
        }

        // Note: the encoder_hidden_states parameter (originally named "encoder_hidden_states")
        // is renamed to "Parameter_XXXXX" by SDPAToPagedAttention / StatefulToStateless, but
        // its tensor_name remains "encoder_hidden_states".  StatefulToStateless already creates
        // it with a fully-dynamic shape [-1, -1, D], so no set_partial_shape is needed here.
        // model_runner.hpp passes [N, T_enc, D] at runtime; OV accepts any N≥1.

        // IMPORTANT: must call validate_nodes_and_infer_types() here, INSIDE the function,
        // after all replace_node() calls.  Without this, the output shape metadata on the
        // out_seq_t Transpose nodes (which replaced the original SDPA nodes) is stale/unknown.
        // The consumer Transpose_3 nodes still see the old SDPA output shape from before
        // replace_node(), and the CPU plugin uses those stale shapes during execution,
        // producing completely wrong attention outputs (timestamp-doubling loop) even for N=1.
        model->validate_nodes_and_infer_types();
        if (cb_verbose()) std::cout << "  " << seg_count << " cross-attention layer(s) segmented.\n";

        // ---- Per-layer hidden-state debug taps ----
        // For each decoder layer we add three Result nodes so model_runner can compare
        // pos[0] vs pos[1] and find where req2's hidden state diverges from req1's:
        //   DEBUG_selfattn_LN  – PA self-attention output (before residual add)
        //   DEBUG_xattn_LN     – segmented cross-attention out_seq_t (before residual add)
        //   DEBUG_ffn_LN       – layer output after FFN + residual (full layer output)
        //
        // Strategy:
        //  • out_seq_t nodes are already named "…/out_seq_t" and easy to find.
        //  • Self-attn PA nodes: locate by type + layer name.
        //  • FFN output = the Add node that is the last op of each layer before
        //    the next layer's self-attn input.  We use a simple heuristic: the Add
        //    whose direct consumer is the next layer's first LayerNorm (or the final LN).
        {
            std::vector<std::shared_ptr<ov::op::v0::Result>> dbg_results;
            // Map layer index → out_seq_t node (cross-attention output)
            std::map<int, std::shared_ptr<ov::Node>> layer_xattn;
            for (const auto& op : model->get_ordered_ops()) {
                const std::string& n = op->get_friendly_name();
                if (n.find("/out_seq_t") == std::string::npos) continue;
                // Extract layer index from name like "model.decoder.layers.2.encoder_attn/out_seq_t"
                auto pos_l = n.find("layers.");
                if (pos_l == std::string::npos) continue;
                int layer_idx = std::stoi(n.substr(pos_l + 7));
                layer_xattn[layer_idx] = op;
            }
            for (auto& [layer_idx, xattn_node] : layer_xattn) {
                // Cross-attention tap: shape [1, n_heads, N, head_dim] — already 4D
                auto dbg_xattn = std::make_shared<ov::op::v0::Result>(xattn_node->output(0));
                std::string xattn_name = "DEBUG_xattn_L" + std::to_string(layer_idx);
                dbg_xattn->set_friendly_name(xattn_name);
                dbg_xattn->get_output_tensor(0).set_names({xattn_name});
                dbg_results.push_back(dbg_xattn);

                // Cross-attention residual Add tap: walk consumers of out_seq_t to find the
                // downstream Add (the residual connection).  It's typically 1-2 hops away
                // (possibly through a Transpose that undoes the head layout before adding).
                // We find the first Add among consumers (BFS depth ≤ 4).
                std::function<std::shared_ptr<ov::Node>(std::shared_ptr<ov::Node>, int)> find_add;
                find_add = [&](std::shared_ptr<ov::Node> src, int depth) -> std::shared_ptr<ov::Node> {
                    if (depth == 0) return nullptr;
                    for (auto& out : src->outputs()) {
                        for (auto& inp : out.get_target_inputs()) {
                            auto tgt = inp.get_node()->shared_from_this();
                            if (ov::is_type<ov::op::v1::Add>(tgt)) return tgt;
                            auto found = find_add(tgt, depth - 1);
                            if (found) return found;
                        }
                    }
                    return nullptr;
                };
                auto xattn_residual_add = find_add(xattn_node, 4);
                if (xattn_residual_add) {
                    auto dbg_xattn_res = std::make_shared<ov::op::v0::Result>(xattn_residual_add->output(0));
                    std::string res_name = "DEBUG_after_xattn_L" + std::to_string(layer_idx);
                    dbg_xattn_res->set_friendly_name(res_name);
                    dbg_xattn_res->get_output_tensor(0).set_names({res_name});
                    dbg_results.push_back(dbg_xattn_res);
                    if (cb_verbose()) std::cout << "  [dbg-tap] " << res_name << " → "
                              << xattn_residual_add->get_friendly_name() << "\n";
                } else {
                    if (cb_verbose()) std::cout << "  [dbg-tap] WARNING: cross-attn residual Add not found for layer " << layer_idx << "\n";
                }
            }
            model->add_results(dbg_results);
            model->validate_nodes_and_infer_types();
            if (cb_verbose()) std::cout << "  Added " << dbg_results.size() << " per-layer hidden-state debug tap(s).\n";
        }

        // Debug: dump inferred shapes of q_seq_t and segmented SDPA for the first CA layer.
        for (const auto& op : model->get_ordered_ops()) {
            const std::string& n = op->get_friendly_name();
            if (n.find("layers.0") == std::string::npos) continue;
            if (n.find("q_seq_t") != std::string::npos || n.find("/seg") != std::string::npos) {
                if (cb_verbose()) std::cout << "  [shape-dbg] " << n.substr(n.rfind('/')+1) << ":";
                for (size_t i = 0; i < op->get_input_size(); ++i)
                    if (cb_verbose()) std::cout << " in" << i << "=" << op->get_input_partial_shape(i);
                for (size_t i = 0; i < op->get_output_size(); ++i)
                    if (cb_verbose()) std::cout << " out" << i << "=" << op->get_output_partial_shape(i);
                if (cb_verbose()) std::cout << "\n";
            }
        }
        if (cb_verbose()) std::cout << "\n";
    } // end if (!skip_segmented_ca)

    if (cb_verbose()) std::cout << "Selective PA transformation complete!\n\n";
    return projector_model;
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
    if (cb_verbose()) std::cout << "Continuous Batching Pipeline: Checking for 2D input_ids parameter to reshape to 1D...\n";
    if (cb_verbose()) std::cout << "Model parameters:\n";
    for (auto& param : model->get_parameters()) {
        auto shape = param->get_partial_shape();
        if (cb_verbose()) std::cout << "  - " << param->get_friendly_name() << ": shape=" << shape 
                  << ", rank=" << (shape.rank().is_static() ? std::to_string(shape.rank().get_length()) : "dynamic") << "\n";
    }
    
    for (auto& param : model->get_parameters()) {
        if (param->get_friendly_name() == "input_ids" || param->get_friendly_name() == "decoder_input_ids") {
            auto shape = param->get_partial_shape();
            if (cb_verbose()) std::cout << "Found " << param->get_friendly_name() << " parameter with shape: " << shape << "\n";
            if (shape.rank().is_static() && shape.rank().get_length() == 2) {
                if (cb_verbose()) std::cout << "Reshaping " << param->get_friendly_name() << " from 2D " << shape << " to 1D {?}\n";
                
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
                    
                    if (cb_verbose()) std::cout << "Added Reshape node to flatten input_ids to 1D\n";
                }
            }
            break;
        }
    }
    */
    
    // Check if PA transformation should be skipped entirely for debugging
    const char* skip_pa = std::getenv("SKIP_PA_TRANSFORMATION");
    bool skip_pa_transformation = (skip_pa != nullptr && std::string(skip_pa) == "1");

    // Auto-detect whether this is an encoder-decoder model (Whisper, etc.) by checking
    // for cross-attention SDPA nodes.  If any are found, use apply_selective_pa_transformation
    // which handles ReadValue bypass + projector extraction + segmented cross-attention.
    // PA_SELF_ATTEN_ONLY=1 keeps backward compat; PA_SELF_ATTEN_ONLY=0 forces the full path.
    bool has_cross_attn = false;
    for (const auto& op : model->get_ops()) {
        if (!ov::is_type<ov::op::v13::ScaledDotProductAttention>(op)) continue;
        const std::string& n = op->get_friendly_name();
        if (n.find("encoder_attn") != std::string::npos ||
            n.find("cross_attn")   != std::string::npos) {
            has_cross_attn = true;
            break;
        }
    }

    // Check if selective PA transformation is requested (self-attention only)
    const char* pa_self_attn_only = std::getenv("PA_SELF_ATTEN_ONLY");
    // Auto-enable for encoder-decoder models; env-var can still force either direction.
    bool use_selective_pa = has_cross_attn;
    if (pa_self_attn_only != nullptr)
        use_selective_pa = (std::string(pa_self_attn_only) == "1");
    if (has_cross_attn)
        std::cout << "[CB] Encoder-decoder model detected — using selective PA + CrossKVCache path\n";

    if (skip_pa_transformation) {
        if (cb_verbose()) std::cout << "SKIP_PA_TRANSFORMATION=1: Skipping PA transformation entirely for debugging\n";
        if (cb_verbose()) std::cout << "WARNING: This means no continuous batching - just testing if PA is the issue\n";
    } else if (use_selective_pa) {
        if (cb_verbose()) std::cout << "Applying selective PA transformation (cross-attn auto-detected or PA_SELF_ATTEN_ONLY=1)\n";
        m_projector_model = apply_selective_pa_transformation(model);
        utils::apply_gather_before_matmul_transformation(model);

                // Post-process: Remove incorrect Unsqueeze added after decoder_input_ids/input_ids for Whisper
        // The PA transformation incorrectly adds Unsqueeze [?,?] -> [?,1,?] which breaks subsequent Reshape
        if (cb_verbose()) std::cout << "\nPost-processing: Checking for incorrect Unsqueeze after decoder_input_ids...\n";
        for (auto& param : model->get_parameters()) {
            if (param->get_friendly_name() == "input_ids" || param->get_friendly_name() == "decoder_input_ids") {
                auto shape = param->get_partial_shape();
                if (cb_verbose()) std::cout << "Found parameter: " << param->get_friendly_name() << " with shape " << shape << "\n";
                
                // Check if first consumer is an Unsqueeze
                auto consumers = param->output(0).get_target_inputs();
                for (auto& consumer : consumers) {
                    auto consumer_node = consumer.get_node();
                    if (ov::is_type<ov::op::v0::Unsqueeze>(consumer_node)) {
                        if (cb_verbose()) std::cout << "  Found Unsqueeze node: " << consumer_node->get_friendly_name() << "\n";
                        
                        // Get the Unsqueeze node's consumers
                        auto unsqueeze_consumers = consumer_node->output(0).get_target_inputs();
                        
                        // Bypass the Unsqueeze ONLY for Reshape consumers
                        // Keep it for ShapeOf and other operations that might need the 3D shape
                        for (auto& unsqueeze_consumer : unsqueeze_consumers) {
                            auto consumer_op = unsqueeze_consumer.get_node();
                            if (ov::is_type<ov::op::v1::Reshape>(consumer_op)) {
                                unsqueeze_consumer.replace_source_output(param->output(0));
                                if (cb_verbose()) std::cout << "    Bypassed Unsqueeze for Reshape: " 
                                          << consumer_op->get_friendly_name() << "\n";
                            } else {
                                if (cb_verbose()) std::cout << "    Kept Unsqueeze for: " 
                                          << consumer_op->get_friendly_name() << " (type: " 
                                          << consumer_op->get_type_name() << ")\n";
                            }
                        }
                        
                        if (cb_verbose()) std::cout << "  Selectively removed Unsqueeze connections for " << param->get_friendly_name() << "\n";
                    }
                }
            }
        }
        if (cb_verbose()) std::cout << "Post-processing complete.\n\n";

    } else {
        if (cb_verbose()) std::cout << "Applying full PA transformation (all SDPA nodes)\n";
        bool is_need_per_layer_cache_control = scheduler_config.use_cache_eviction;
        bool allow_cache_rotation = scheduler_config.cache_eviction_config.apply_rotation;
        bool allow_xattention = scheduler_config.use_sparse_attention && scheduler_config.sparse_attention_config.mode == SparseAttentionMode::XATTENTION;
        bool allow_score_aggregation = true;
        bool allow_adaptive_rkv = scheduler_config.use_cache_eviction && scheduler_config.cache_eviction_config.aggregation_mode == AggregationMode::ADAPTIVE_RKV;
        auto sdpa_to_pa_successful = ov::pass::SDPAToPagedAttention(is_need_per_layer_cache_control, is_need_per_layer_cache_control, allow_score_aggregation, allow_cache_rotation, allow_xattention, allow_adaptive_rkv).run_on_model(model);
        if (sdpa_to_pa_successful) {
            if (cb_verbose()) std::cout << "SDPA to Paged Attention transformation applied successfully.\n";
        } else {
            if (cb_verbose()) std::cout << "SDPA to Paged Attention transformation was not applied successfully.\n";
        }
        utils::apply_gather_before_matmul_transformation(model);
        
        // Note: decoder_input_ids stays as [?,?] parameter
        // We handle 1D->2D conversion at runtime in ModelRunner
    }
    
    // Additional post-processing: Fix Reshape constants before PagedAttentionExtension
    // The SDPAToPagedAttention transformation creates Reshape nodes with pattern [1, -1]
    // but for continuous batching we need [-1, hidden_size] to properly flatten batch*seq_len
    if (cb_verbose()) std::cout << "\nPost-processing: Fixing Reshape patterns before PagedAttentionExtension...\n";
    for (auto& node : model->get_ordered_ops()) {
        if (node->get_type_name() == std::string("PagedAttentionExtension")) {
            if (cb_verbose()) std::cout << "Found PagedAttentionExtension: " << node->get_friendly_name() << "\n";
            
            // Check inputs 0, 1, 2 (Q, K, V)
            for (size_t input_idx = 0; input_idx < 3; ++input_idx) {
                auto input_node = node->get_input_node_shared_ptr(input_idx);
                if (ov::is_type<ov::op::v1::Reshape>(input_node)) {
                    if (cb_verbose()) std::cout << "  Input " << input_idx << " is Reshape: " << input_node->get_friendly_name() << "\n";
                    
                    // Get the reshape pattern (should be second input)
                    auto pattern_node = input_node->get_input_node_shared_ptr(1);
                    if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(pattern_node)) {
                        auto pattern_data = constant->cast_vector<int64_t>();
                        if (cb_verbose()) std::cout << "    Current pattern: [";
                        for (size_t i = 0; i < pattern_data.size(); ++i) {
                            if (cb_verbose()) std::cout << pattern_data[i];
                            if (i < pattern_data.size() - 1) std::cout << ", ";
                        }
                        if (cb_verbose()) std::cout << "]\n";
                        
                        // Debug: print each element
                        if (cb_verbose()) std::cout << "    Pattern analysis: size=" << pattern_data.size();
                        if (pattern_data.size() >= 1) std::cout << ", [0]=" << pattern_data[0];
                        if (pattern_data.size() >= 2) std::cout << ", [1]=" << pattern_data[1];
                        if (cb_verbose()) std::cout << "\n";
                        
                        // If pattern is [0, -1] or [1, -1], change to [-1, hidden_size]
                        // [0, -1] means "copy first dim from input, infer second" which is wrong for PA
                        bool needs_fix = (pattern_data.size() == 2 && 
                                         (pattern_data[0] == 0 || pattern_data[0] == 1) && 
                                         pattern_data[1] == -1);
                        if (cb_verbose()) std::cout << "    Pattern needs_fix: " << (needs_fix ? "YES" : "NO") << "\n";
                        
                        if (needs_fix) {
                            if (cb_verbose()) std::cout << "    Pattern needs fixing: [" << pattern_data[0] << ", -1] -> [-1, hidden_size]\n";
                            
                            // Get the input shape to Reshape to determine hidden_size
                            auto reshape_input_shape = input_node->get_input_partial_shape(0);
                            if (cb_verbose()) std::cout << "    Reshape input shape: " << reshape_input_shape << "\n";
                            if (cb_verbose()) std::cout << "    Reshape input rank: " << reshape_input_shape.rank() << "\n";
                            
                            // For shape [batch, seq, num_heads, head_dim] or [?, ?, num_heads, head_dim]
                            // hidden_size = num_heads * head_dim (should be static even if batch/seq are dynamic)
                            int64_t hidden_size = -1;
                            if (reshape_input_shape.rank().is_static() && reshape_input_shape.rank().get_length() == 4) {
                                if (cb_verbose()) std::cout << "    Reshape input is 4D\n";
                                auto num_heads_dim = reshape_input_shape[2];
                                auto head_dim = reshape_input_shape[3];
                                if (cb_verbose()) std::cout << "    num_heads_dim: " << num_heads_dim << ", head_dim: " << head_dim << "\n";
                                
                                if (num_heads_dim.is_static() && head_dim.is_static()) {
                                    hidden_size = num_heads_dim.get_length() * head_dim.get_length();
                                    if (cb_verbose()) std::cout << "    Computed hidden_size: " << hidden_size << "\n";
                                } else {
                                    if (cb_verbose()) std::cout << "    ERROR: num_heads or head_dim is dynamic\n";
                                }
                            } else {
                                if (cb_verbose()) std::cout << "    ERROR: Reshape input is not 4D or rank is dynamic\n";
                            }
                            
                            // If we couldn't compute hidden_size, try hardcoded value for Whisper
                            if (hidden_size == -1) {
                                // For Whisper large-v3-turbo, hidden_size is 1280 (20 heads * 64 head_dim)
                                // Check if this is a Whisper model by looking for 20 heads
                                if (reshape_input_shape.rank().is_static() && reshape_input_shape.rank().get_length() == 4) {
                                    auto num_heads_dim = reshape_input_shape[2];
                                    if (num_heads_dim.is_static() && num_heads_dim.get_length() == 20) {
                                        hidden_size = 1280;
                                        if (cb_verbose()) std::cout << "    Using hardcoded hidden_size for Whisper: " << hidden_size << "\n";
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
                                        if (cb_verbose()) std::cout << "    VERIFIED - New pattern: [";
                                        for (size_t i = 0; i < verify_pattern.size(); ++i) {
                                            if (cb_verbose()) std::cout << verify_pattern[i];
                                            if (i < verify_pattern.size() - 1) std::cout << ", ";
                                        }
                                        if (cb_verbose()) std::cout << "]\n";
                                    }
                                    
                                    if (cb_verbose()) std::cout << "    Fixed pattern: [-1, " << hidden_size << "]\n";
                            } else {
                                if (cb_verbose()) std::cout << "    ERROR: Could not determine hidden_size for shape " << reshape_input_shape << "\n";
                            }
                        } else {
                            if (cb_verbose()) std::cout << "    Pattern does not need fixing\n";
                        }
                    }
                }
            }
        }
    }
    if (cb_verbose()) std::cout << "Input Reshape pattern fixes complete.\n\n";

    // Revalidate the model after all modifications
    if (cb_verbose()) std::cout << "Revalidating model after all transformations...\n";
    model->validate_nodes_and_infer_types();
    if (cb_verbose()) std::cout << "Model validation complete.\n\n";

    // Post-processing: bypass the head-unsplit Reshape/Transpose chain left by
    // SDPAToPagedAttention after each self-attention PA output.
    //
    // SDPAToPagedAttention replaces the SDPA node with a PA node but leaves the
    // original post-SDPA chain in place:
    //   SDPA [bs,n_heads,T,head_dim] → Transpose → Transpose_3 → Reshape(0,0,H)
    // Wired to PA output it becomes:
    //   PA [N,H] → Reshape(head-unsplit) → Transpose → Transpose_3 → Reshape(0,0,H) → out_proj
    //
    // The final Reshape(0,0,H) uses special_zero=true so dim0 and dim1 come from the
    // input tensor.  Its input is shape [1,1,N,64] (from the Transpose chain), so it
    // always produces [1,1,H] — collapsing N tokens to 1.  At N=1 that accidentally
    // matches the residual [1,1,H]; at N>1 the residual is [1,N,H] and the Add fails.
    //
    // Fix: replace the entire Reshape/Transpose/Transpose/Reshape chain with a single
    // Reshape [N,H] → [1,-1,H].  "Walk" from PA output through consecutive Reshape and
    // Transpose nodes; redirect the FIRST non-Reshape/Transpose consumer (i.e. the
    // out_proj MatMul) to receive the new [1,-1,H] tensor instead.
    //
    // All remaining PA nodes are self-attention (cross-attention was restored to SDPA).
    {
        if (cb_verbose()) std::cout << "Post-processing: Bypassing PA post-SDPA head-unsplit chain...\n";
        auto ops_for_bypass = model->get_ordered_ops();
        int bypass_count = 0;
        for (const auto& op : ops_for_bypass) {
            if (op->get_type_info().name != std::string("PagedAttentionExtension")) continue;

            // H is statically known from the PA output partial shape.
            auto pa_out_shape = op->output(0).get_partial_shape();
            int64_t H_static = -1;
            if (pa_out_shape.rank().is_static() && pa_out_shape.rank().get_length() >= 2) {
                auto H_dim = pa_out_shape[pa_out_shape.rank().get_length() - 1];
                if (H_dim.is_static()) H_static = static_cast<int64_t>(H_dim.get_length());
            }
            if (H_static <= 0) {
                if (cb_verbose()) std::cout << "  Skipping (H unknown): " << op->get_friendly_name() << "\n";
                continue;
            }

            // Walk from PA output through Reshape/Transpose nodes to find the first
            // non-Reshape/Transpose consumer (the out_proj MatMul or equivalent).
            // Collect all Reshape/Transpose nodes in the chain so we can transfer names.
            //
            // Only follow single-consumer chains to avoid splitting the graph incorrectly.
            ov::Output<ov::Node> cursor = op->output(0);
            std::string chain_last_name;
            bool found_chain = false;
            for (int depth = 0; depth < 8; ++depth) {
                auto consumers = cursor.get_target_inputs();
                if (consumers.size() != 1) break;  // multiple consumers — stop
                auto next = consumers.begin()->get_node()->shared_from_this();
                const std::string& t = next->get_type_info().name;
                if (t != std::string("Reshape") && t != std::string("Transpose")) {
                    // next is the out_proj MatMul (or first non-chain node)
                    found_chain = true;
                    break;
                }
                chain_last_name = next->get_friendly_name();
                cursor = next->output(0);
            }

            if (!found_chain) {
                if (cb_verbose()) std::cout << "  No chain found for: " << op->get_friendly_name() << "\n";
                continue;
            }

            // cursor is now the output of the last Reshape/Transpose in the chain.
            // All its consumers should be redirected to the new unflatten output.
            auto unflatten_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3},
                                                                 std::vector<int64_t>{1, -1, H_static});
            auto unflatten = std::make_shared<ov::op::v1::Reshape>(
                op->output(0), unflatten_shape->output(0), /*special_zero=*/false);
            unflatten->set_friendly_name(op->get_friendly_name() + "/out_unflatten");

            // Transfer tensor names from the chain tail so downstream references resolve.
            unflatten->output(0).get_tensor().set_names(cursor.get_tensor().get_names());
            cursor.get_tensor().set_names({});

            // Collect consumers of chain tail BEFORE redirect.
            std::vector<ov::Input<ov::Node>> targets;
            for (auto& tgt : cursor.get_target_inputs())
                targets.push_back(tgt);
            for (auto& tgt : targets)
                tgt.replace_source_output(unflatten->output(0));

            if (cb_verbose()) std::cout << "  Bypassed chain → unflatten after PA: "
                      << op->get_friendly_name()
                      << "  [N," << H_static << "] → [1,N," << H_static << "]"
                      << "  (chain tail: " << chain_last_name << ")\n";
            ++bypass_count;
        }
        if (cb_verbose()) std::cout << "  Bypassed " << bypass_count << " PA head-unsplit chain(s).\n\n";
    }

    // Revalidate after bypass insertions
    model->validate_nodes_and_infer_types();

    // Verify final parameter shapes
    if (cb_verbose()) std::cout << "Final parameter shapes after all modifications:\n";
    for (auto& param : model->get_parameters()) {
        if (cb_verbose()) std::cout << "  " << param->get_friendly_name() << ": " << param->get_partial_shape() << "\n";
    }
    if (cb_verbose()) std::cout << "\n";

    // Save the transformed model to XML for debugging
    if (cb_verbose()) std::cout << "Saving transformed model to pa_experimental_model.xml...\n";
    try {
        ov::pass::Serialize("./pa_experimental_model.xml", "./pa_experimental_model.bin").run_on_model(model);
        if (cb_verbose()) std::cout << "Model saved successfully!\n\n";
    } catch (const std::exception& e) {
        if (cb_verbose()) std::cout << "Failed to save model: " << e.what() << "\n\n";
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
    // Free any still-allocated KV-cache blocks before the block-manager is destroyed.
    // This prevents the BlockManager destructor assertion (m_block_table.empty()) from
    // firing when sequences are in-flight at shutdown (e.g. incomplete generation, exception, etc.)
    if (m_scheduler) {
        try {
            _pull_awaiting_requests();
        } catch (...) {}
        for (const auto& request : m_requests) {
            for (const auto& sequence : request->get_sequences()) {
                try {
                    if (m_scheduler->has_block_table(sequence->get_id())) {
                        m_scheduler->free_sequence(sequence->get_id());
                    }
                } catch (...) {}
            }
        }
        m_requests.clear();
    }

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

void ContinuousBatchingPipeline::ContinuousBatchingImpl::_flush_pending_projections() {
    // How many requests can we admit without blocking?
    const size_t n_free = m_cross_kv_cache->n_free_slots();
    if (n_free == 0) return;

    // Take up to n_free requests from the pending queue.
    std::vector<PendingProjection> batch;
    {
        std::lock_guard<std::mutex> pl(m_pending_proj_mutex);
        if (m_pending_proj_queue.empty()) return;
        const size_t n_take = std::min(n_free, m_pending_proj_queue.size());
        batch.assign(m_pending_proj_queue.begin(),
                     m_pending_proj_queue.begin() + static_cast<ptrdiff_t>(n_take));
        m_pending_proj_queue.erase(m_pending_proj_queue.begin(),
                                   m_pending_proj_queue.begin() + static_cast<ptrdiff_t>(n_take));
    }

    const size_t N = batch.size();

    const auto& ref_shape = batch[0].encoder_hidden_states.get_shape();
    OPENVINO_ASSERT(ref_shape.size() == 3,
        "_flush_pending_projections: expected encoder_hidden_states shape [1, T_enc, D]");
    const size_t T_enc = ref_shape[1];
    const size_t D     = ref_shape[2];
    const auto hs_type = batch[0].encoder_hidden_states.get_element_type();

    std::vector<uint64_t> req_ids;
    req_ids.reserve(N);
    for (auto& p : batch) req_ids.push_back(p.req_id);

    // Stack N encoder hidden states into [N, T_enc, D] and run the projector once.
    // Batching amortises the kernel launch cost; N parallel slot writes follow.
    const size_t slice_bytes = T_enc * D * hs_type.size();
    ov::Tensor batched_hs(hs_type, {N, T_enc, D});
    uint8_t* dst = static_cast<uint8_t*>(batched_hs.data());
    for (size_t i = 0; i < N; ++i)
        std::memcpy(dst + i * slice_bytes, batch[i].encoder_hidden_states.data(), slice_bytes);

    m_proj_infer_request.set_tensor("encoder_hidden_states", batched_hs);
    const auto t0 = std::chrono::steady_clock::now();
    m_proj_infer_request.infer();
    const double proj_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - t0).count();
    m_cross_kv_cache->add_proj_infer_us(proj_us);
    m_cross_kv_cache->admit_precomputed_batch(req_ids, m_proj_infer_request);
    std::cout << "[flush] batch N=" << N << " proj=" << std::fixed
              << std::setprecision(1) << proj_us / 1000.0 << " ms\n";

    // Assign slot IDs and push sequence groups to the awaiting queue.
    {
        std::lock_guard<std::mutex> lock(m_awaiting_requests_mutex);
        for (auto& p : batch) {
            p.sequence_group->set_cross_kv_slot_id(m_cross_kv_cache->slot_of(p.req_id));
            m_awaiting_requests.push_back(p.sequence_group);
        }
    }
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
    if (cb_verbose()) std::cout << "=== MODEL INPUTS BEFORE COMPILE ===\n";
    for (const auto& input : model->inputs()) {
        if (cb_verbose()) std::cout << "  INPUT: ";
        for (const auto& name : input.get_names()) std::cout << "'" << name << "' ";
        if (cb_verbose()) std::cout << "  shape=" << input.get_partial_shape() << "\n";
    }
    if (cb_verbose()) std::cout << "===================================\n\n";

    // Audit: enumerate dynamic-output Reshape nodes just before compile.
    // With cb_verbose(), print every dynamic/suspicious Reshape and trace
    // the shape-input chain for any Reshape with "229018" in its name.
    {
        int reshape_total = 0, reshape_dynamic = 0;
        for (const auto& op : model->get_ordered_ops()) {
            if (!ov::is_type<ov::op::v1::Reshape>(op)) continue;
            ++reshape_total;
            const auto& out_ps = op->get_output_partial_shape(0);
            const bool is_static = out_ps.rank().is_static() && out_ps.is_static();
            if (is_static) continue;
            ++reshape_dynamic;
            const std::string& name = op->get_friendly_name();
            const bool is_229018 = name.find("229018") != std::string::npos;
            if (cb_verbose() || is_229018) {
                std::cout << "[DecoderAudit] Reshape '" << name
                          << "'  out_shape=" << out_ps
                          << (is_229018 ? "  <<< FOUND 229018" : "") << "\n";
            }
            if (is_229018) {
                for (size_t pi = 0; pi < op->get_input_size(); ++pi) {
                    auto src = op->input(pi).get_source_output();
                    std::cout << "  [229018]  input[" << pi << "] from '"
                              << src.get_node()->get_friendly_name()
                              << "' idx=" << src.get_index()
                              << "  shape=" << src.get_partial_shape() << "\n";
                }
                for (const auto& tgt : op->output(0).get_target_inputs())
                    std::cout << "  [229018]  consumed by '"
                              << tgt.get_node()->get_friendly_name()
                              << "' port=" << tgt.get_index() << "\n";
            }
        }
        std::cout << "[DecoderAudit] Reshape nodes: " << reshape_total
                  << " total, " << reshape_dynamic << " dynamic-output\n";
    }

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

    // ---- Cross-attention K/V slot cache (Option A) ----
    // Compile the projector model extracted by apply_selective_pa_transformation() and
    // instantiate a CrossKVCache so every admitted request runs K/V projection once.
    if (m_projector_model) {
        const char* kv_env = std::getenv("CROSS_KV_CACHE");
        const bool cross_kv_enabled = !(kv_env && std::string(kv_env) == "0");
        if (cross_kv_enabled) {
            // Make the batch dimension dynamic so _flush_pending_projections() can
            // run the projector with [N, T_enc, D] input for N ≥ 1.
            {
                std::map<std::string, ov::PartialShape> dyn_batch_map;
                for (auto& inp : m_projector_model->inputs()) {
                    auto ps = inp.get_partial_shape();
                    // Always make batch dimension (dim 0) dynamic, regardless of
                    // whether other dims are already dynamic (e.g. T_enc is already ?).
                    if (ps.size() >= 1)
                        ps[0] = ov::Dimension();  // dynamic batch
                    dyn_batch_map[inp.get_any_name()] = ps;
                }
                m_projector_model->reshape(dyn_batch_map);
            }
            // Compile projector and stash the InferRequest.
            // CrossKVCache is constructed lazily on the first admit() call once we have
            // a real encoder_hidden_state and can read concrete output shapes.
            if (cb_verbose()) std::cout << "[CrossKV] Compiling projector model (dynamic batch)...\n";
            ov::CompiledModel proj_compiled =
                utils::singleton_core().compile_model(m_projector_model, device, *filtered_properties);
            m_proj_infer_request = proj_compiled.create_infer_request();
            std::cout << "[CrossKV] Projector compiled — CrossKVCache will be initialised on first request\n";
        } else {
            std::cout << "[CrossKV] CROSS_KV_CACHE=0 — running without slot cache (slow path)\n";
            m_projector_model = nullptr;  // disable fast path gate
        }
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
            if (cb_verbose()) {
                std::cout << "Creating SequenceGroup with encoder_hidden_state\n";
                std::cout << "  encoder_hidden_state shape: [";
                for (size_t i = 0; i < encoder_hidden_state->get_shape().size(); ++i) {
                    std::cout << encoder_hidden_state->get_shape()[i];
                    if (i < encoder_hidden_state->get_shape().size() - 1) std::cout << ", ";
                }
                std::cout << "], size: " << encoder_hidden_state->get_size() << "\n";
            }
            sequence_group = std::make_shared<SequenceGroup>(request_id, input_ids, *encoder_hidden_state, sampling_params_copy, m_block_size);
        } else {
            if (cb_verbose()) std::cout << "Creating SequenceGroup WITHOUT encoder_hidden_state\n";
            sequence_group = std::make_shared<SequenceGroup>(request_id, input_ids, sampling_params_copy, m_block_size, token_type_ids);
        }    
    }

    if (m_scheduler->get_config().enable_prefix_caching) {
        m_scheduler->restore_cached_blocks(sequence_group);
    }

    // Cross-KV slot admission: run projector once per request so K/V for every
    // cross-attention layer is cached before the first decode step fires.
    if (encoder_hidden_state.has_value() && m_projector_model) {
        // Lazy init of CrossKVCache on first admit(): run projector once with a real
        // encoder_hidden_state so we can read concrete output shapes (T_enc, n_heads,
        // head_dim) from the actual inferred tensors — avoids relying on OV partial shape
        // propagation which leaves all dims dynamic in this graph.
        if (!m_cross_kv_cache) {
            m_proj_infer_request.set_tensor("encoder_hidden_states", *encoder_hidden_state);
            m_proj_infer_request.infer();

            // Read concrete geometry from layer-0 K output
            ov::Tensor K0 = m_proj_infer_request.get_tensor("cross_proj_K_0");
            const auto& K0_shape = K0.get_shape();
            OPENVINO_ASSERT(K0_shape.size() == 4,
                "cross_proj_K_0 must be rank-4 [1, n_heads, T_enc, head_dim]");
            const size_t n_heads  = K0_shape[1];
            const size_t T_enc    = K0_shape[2];
            const size_t head_dim = K0_shape[3];
            ov::element::Type elem_type = K0.get_element_type();

            // Count layers: projector has cross_proj_K_l + cross_proj_V_l per layer
            const size_t n_layers = m_proj_infer_request.get_compiled_model().outputs().size() / 2;

            size_t max_slots = 16;
            const char* env_slots = std::getenv("CROSS_KV_MAX_SLOTS");
            if (env_slots) max_slots = static_cast<size_t>(std::atoi(env_slots));

            m_cross_kv_cache = std::make_unique<CrossKVCache>(
                m_proj_infer_request, max_slots,
                n_layers, n_heads, T_enc, head_dim, elem_type);
            m_model_runner->set_cross_kv_cache(m_cross_kv_cache.get());

            std::cout << "[CrossKV] Slot cache initialised: max_slots=" << max_slots
                      << " layers=" << n_layers << " heads=" << n_heads
                      << " T_enc=" << T_enc << " head_dim=" << head_dim << "\n";

            // Slot 0 is already projected from the infer() above — write it directly
            // into the buffer rather than re-running infer() inside admit().
            m_cross_kv_cache->admit_precomputed(request_id, m_proj_infer_request);
            sequence_group->set_cross_kv_slot_id(m_cross_kv_cache->slot_of(request_id));
            {
                std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
                m_awaiting_requests.push_back(sequence_group);
            }
            std::cout << "[request " << request_id << "] added, prompt_len=" << prompt_len << "\n";
        } else {
            // Projection is batched in _flush_pending_projections() called from step().
            // Push to pending queue so the step loop handles it without blocking add_request().
            {
                std::lock_guard<std::mutex> pl(m_pending_proj_mutex);
                m_pending_proj_queue.push_back({request_id, *encoder_hidden_state, sequence_group});
            }
            // sequence_group is pushed to m_awaiting_requests by _flush_pending_projections()
            // after projection completes — do NOT push it here.
        }
    } else {
        {
            std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
            m_awaiting_requests.push_back(sequence_group);
        }
        std::cout << "[request " << request_id << "] added, prompt_len=" << prompt_len << "\n";
    }
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
    if (cb_verbose()) std::cout << "ContinuousBatchingPipeline::ContinuousBatchingImpl::add_request with encoder_hidden_state called\n";
    // Persist the Whisper config so step()-time logit processing has access to it.
    m_whisper_gen_config = sampling_params;
    m_has_whisper_config = true;
    // WhisperGenerationConfig inherits from GenerationConfig, so we can pass it directly
    return add_request(request_id, input_ids, sampling_params, std::nullopt, encoder_hidden_state);
}

bool ContinuousBatchingPipeline::ContinuousBatchingImpl::has_non_finished_requests() {
    std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
    if (!m_awaiting_requests.empty() || !m_requests.empty())
        return true;
    // Also check for requests whose projection hasn't run yet.
    std::lock_guard<std::mutex> pl(m_pending_proj_mutex);
    return !m_pending_proj_queue.empty();
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::step() {
    static ManualTimer step_timer("step()");
    static size_t step_idx = 0;
    step_timer.start();

    // Batch-project any pending encoder hidden states and admit them to the
    // CrossKVCache slot buffer.  Done before _pull_awaiting_requests() so that
    // requests whose K/V is now ready get pulled into the scheduler this step.
    if (m_cross_kv_cache)
        _flush_pending_projections();

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

    // Always-visible per-step summary
    std::cout << "[step " << step_idx++ << "] "
              << scheduler_output.m_scheduled_sequence_groups_ids.size() << " req(s), "
              << scheduler_output.m_total_num_scheduled_tokens << " token(s)\n";

    ov::Tensor logits;

    {
        static ManualTimer timer("forward");
        const auto infer_start = std::chrono::steady_clock::now();
        timer.start();

        // Three-way split: PROMPT / TRANSITION / GENERATE forward() passes.
        //
        // Phase definitions (context_len = processed + scheduled, prompt_len = #SOT tokens):
        //   PROMPT     – !requires_sampling()              (context_len < prompt_len)
        //   TRANSITION – requires_sampling() && !can_generate_tokens()  (context_len == prompt_len,
        //                  processing the last SOT token; produces the first generated token)
        //   GENERATE   – can_generate_tokens()             (context_len > prompt_len)
        //
        // Root causes for mixing phases in one forward():
        //   1. PROMPT + GENERATE/TRANSITION: sampler offset corruption + KV contamination.
        //      PROMPT groups have output_seq_len=0 so the sampler offset logic is wrong when
        //      they remain is_scheduled()==true.  PagedAttention also writes fresh PROMPT KV
        //      entries while simultaneously reading stale GENERATE KV entries in the same batch.
        //   2. TRANSITION + GENERATE: TRANSITION computes its first token logit in a batch
        //      that includes GENERATE queries.  Those GENERATE queries have accumulated KV
        //      context from prior generation steps; TRANSITION has not.  The segmented
        //      cross-attention and self-attention interact over an inconsistent KV context,
        //      corrupting the TRANSITION group's first-token logit (wrong text tokens or
        //      hallucinated timestamp-repeat loops).
        //
        // Solution: three separate forward() passes, one per phase.
        //   Pass 1 (PROMPT):     write KV, discard logits, finish_iteration() immediately.
        //   Pass 2 (TRANSITION): solo forward, deep-copy logits (forward() returns a VIEW
        //                        into the infer-request buffer; Pass 3 would alias it).
        //   Pass 3 (GENERATE):   solo forward, use logits directly (last forward call).
        //
        // After Passes 2+3, logit rows are assembled into one tensor in ascending sg_id
        // order (= sampler iteration order over m_requests) using byte-accurate memcpy
        // keyed on elem_type.size() so both f32 and bf16 models are handled correctly.
        std::vector<uint64_t> prompt_ids, trans_ids, gen_ids;
        for (uint64_t sg_id : scheduler_output.m_scheduled_sequence_groups_ids) {
            auto& sg = m_requests[sg_id];
            if      (!sg->requires_sampling())   prompt_ids.push_back(sg_id);
            else if (!sg->can_generate_tokens()) trans_ids.push_back(sg_id);
            else                                 gen_ids.push_back(sg_id);
        }
        const bool has_prompt = !prompt_ids.empty();
        const bool has_trans  = !trans_ids.empty();
        const bool has_gen    = !gen_ids.empty();

        if (cb_verbose()) std::cout << "[THREE_SPLIT] prompt=" << prompt_ids.size()
                                     << " trans=" << trans_ids.size()
                                     << " gen=" << gen_ids.size() << "\n";

        // Helper: build a sub-Scheduler::Output restricted to the given group IDs.
        auto make_sub_sched = [&](const std::vector<uint64_t>& ids) {
            Scheduler::Output sub = scheduler_output;
            sub.m_scheduled_sequence_groups_ids = ids;
            sub.m_total_num_scheduled_tokens = 0;
            for (uint64_t id : ids)
                sub.m_total_num_scheduled_tokens +=
                    m_requests[id]->get_num_scheduled_tokens() * m_requests[id]->num_running_seqs();
            return sub;
        };

        // needs_any_split: true when any two different phases are co-scheduled,
        // OR (for encoder-decoder / Whisper) when PROMPT groups have different past_lens.
        //
        // Whisper-specific PROMPT-solo rule:
        //   The PagedAttention kernel with N>1 PROMPT sequences is reliable only when all
        //   sequences have the same past_len (i.e. were injected at the same step and have
        //   advanced in lockstep).  When past_lens differ (e.g. [1,0] from staggered
        //   injection) the self-attention write for the later-context sequence goes to a
        //   wrong physical KV slot, corrupting all future generation for that request.
        //   Scenario 2 (A@0+B@0) passes because past_lens are always symmetric;
        //   scenario 3 (A@0+B@1) fails at the first P+P step where past_lens=[1,0].
        //   Fix: run each PROMPT group with a unique past_len in its own solo pass.
        bool prompt_needs_solo = false;
        if (m_has_whisper_config && prompt_ids.size() > 1) {
            size_t ref_past = m_requests[prompt_ids[0]]->get_num_processed_tokens();
            for (size_t pi = 1; pi < prompt_ids.size(); ++pi) {
                if (m_requests[prompt_ids[pi]]->get_num_processed_tokens() != ref_past) {
                    prompt_needs_solo = true;
                    break;
                }
            }
        }

        // TRANSITION sequences (last SOT token → first generated token) are safe to batch
        // with GENERATE sequences: PA handles heterogeneous past_lens by design, and the
        // cross-attention slot assignment is per-request regardless of decode position.
        // Merging eliminates one solo forward() per request entering the GENERATE pool.
        // Separation is only required between PROMPT and non-PROMPT (sampler offset + KV
        // write/read ordering) and between staggered PROMPT groups (Whisper past_len rule).
        const bool needs_any_split = (has_prompt && (has_trans || has_gen))
                                      || prompt_needs_solo;

        // Helper: record last model-runner timings into cumulative pipeline metrics.
        auto accum_runner_timings = [&]() {
            auto lt = m_model_runner->get_last_timings();
            m_pipeline_metrics.cross_attn_assembly_us_total += lt.assembly_us;
            m_pipeline_metrics.ov_infer_us_total            += lt.infer_us;
            if (m_cross_kv_cache)
                m_pipeline_metrics.cross_kv_proj_us_total =
                    m_cross_kv_cache->get_proj_infer_us();
        };

        if (needs_any_split) {
            // ── Pass 1: PROMPT ─────────────────────────────────────────────────
            // For Whisper with mixed-past_len PROMPT groups, run each solo to avoid
            // PA kernel corruption.  For symmetric groups (same past_len, e.g. scenario 2)
            // or non-Whisper models, batch them together in one pass as before.
            if (has_prompt) {
                const auto t_p0 = std::chrono::steady_clock::now();
                if (prompt_needs_solo) {
                    for (uint64_t sg_id : prompt_ids) {
                        m_model_runner->forward(m_requests, make_sub_sched({sg_id}));
                        accum_runner_timings();
                        m_requests[sg_id]->finish_iteration();
                    }
                } else {
                    m_model_runner->forward(m_requests, make_sub_sched(prompt_ids));
                    accum_runner_timings();
                    for (uint64_t sg_id : prompt_ids)
                        m_requests[sg_id]->finish_iteration();
                }
                m_pipeline_metrics.prompt_phase_us_total += std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - t_p0).count();
            }

            // ── Pass 2: TRANSITION + GENERATE (batched together) ─────────────
            // Both phases produce one logit row per sequence; PA handles heterogeneous
            // past_lens by design.  Batching them eliminates one forward() call per
            // request that transitions from prompt processing into decode.
            std::vector<uint64_t> non_prompt_ids;
            non_prompt_ids.reserve(trans_ids.size() + gen_ids.size());
            for (uint64_t sg_id : scheduler_output.m_scheduled_sequence_groups_ids)
                if (m_requests[sg_id]->requires_sampling())
                    non_prompt_ids.push_back(sg_id);

            if (!non_prompt_ids.empty()) {
                const auto t_g0 = std::chrono::steady_clock::now();
                logits = m_model_runner->forward(m_requests, make_sub_sched(non_prompt_ids));
                accum_runner_timings();
                const double dt_us = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - t_g0).count();
                // All time goes to GENERATE; TRANSITION is now batched with it
                // and is no longer a separately-reported phase.
                m_pipeline_metrics.generate_phase_us_total += dt_us;
            }

            // Update scheduler_output so the Whisper logit loop and sampler see only
            // the non-prompt groups (PROMPT groups already have finish_iteration() done).
            scheduler_output.m_scheduled_sequence_groups_ids = std::move(non_prompt_ids);
        } else {
            // Single-phase step (no split needed) — classify by which phase is present.
            const auto t_fwd0 = std::chrono::steady_clock::now();
            logits = m_model_runner->forward(m_requests, scheduler_output);
            accum_runner_timings();
            const double dt_us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t_fwd0).count();
            if      (has_gen)   m_pipeline_metrics.generate_phase_us_total    += dt_us;
            else if (has_trans) m_pipeline_metrics.transition_phase_us_total  += dt_us;
            else                m_pipeline_metrics.prompt_phase_us_total      += dt_us;
        }

        const auto infer_end = std::chrono::steady_clock::now();
        m_pipeline_metrics.inference_duration = PerfMetrics::get_microsec(infer_end - infer_start);
        timer.end();

        // Update cumulative step counters.
        // TRANSITION is now batched with GENERATE so count both in the generate metrics.
        m_pipeline_metrics.total_steps++;
        if (has_gen || has_trans) {
            m_pipeline_metrics.generate_steps++;
            m_pipeline_metrics.generate_batch_token_sum += gen_ids.size() + trans_ids.size();
        }
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

    // When only PROMPT groups were scheduled (e.g. Whisper prompt_needs_solo step),
    // logits was never assigned in the needs_any_split branch — it's still a
    // default-constructed ov::Tensor with shape {}.  Downstream routines
    // (_fill_prompt_log_probs, sampler) unconditionally call logits.get_shape() and
    // OPENVINO_ASSERT(shape.size() == 3) before checking is_scheduled(), which would
    // throw and leave requests with allocated KV blocks → BlockManager destructor
    // assertion.  Provide a valid-but-empty 3-D tensor; no logit data is read because
    // all PROMPT groups have is_scheduled()==false after finish_iteration().
    if (!logits) {
        logits = ov::Tensor(ov::element::f32, ov::Shape{1, 0, 1});
    }

    // process generation_config.echo parameter
    _fill_prompt_log_probs(m_requests, logits);

    // Apply Whisper-specific logit processing (suppress tokens, timestamp forcing)
    // This mirrors what process_whisper_logits() does in whisper.cpp for each decode step.
    //
    // CRITICAL: iterate in scheduler_output.m_scheduled_sequence_groups_ids order, NOT
    // m_requests order.  The model output logits are laid out in the scheduler's token order
    // (which is also the order model_runner fills input_ids and encoder_hidden_states).
    // Iterating m_requests instead produces logit/request misalignment when the scheduler
    // reorders sequences (e.g. PROMPT-first policy), causing late-arriving requests (req3/4)
    // to receive wrong logits → premature EOT.
    if (m_has_whisper_config) {
        for (size_t sched_idx = 0, offset = 0;
             sched_idx < scheduler_output.m_scheduled_sequence_groups_ids.size();
             ++sched_idx) {
            size_t sg_id = scheduler_output.m_scheduled_sequence_groups_ids[sched_idx];
            auto& sg = m_requests[sg_id];

            const size_t num_running = sg->num_running_seqs();
            const size_t output_seq_len = sg->get_output_seq_len();
            const size_t vocab_size = logits.get_shape().back();

            // Always advance offset so subsequent sequences read from the correct position.
            // output_seq_len is 0 for non-sampling PROMPT steps (matmul_gathering filters
            // them out), so the advance is a no-op for those steps.
            if (!sg->requires_sampling()) {
                if (cb_verbose()) std::cout << "[LOGIT_OFFSET] sched[" << sched_idx << "] sg_id=" << sg_id
                                             << " req=" << sg->get_request_id()
                                             << " PROMPT(skip) offset=" << offset << " output_seq_len=" << output_seq_len << "\n";
                offset += output_seq_len * num_running;
                continue;
            }

            if (cb_verbose()) std::cout << "[LOGIT_OFFSET] sched[" << sched_idx << "] sg_id=" << sg_id
                                         << " req=" << sg->get_request_id()
                                         << " offset=" << offset << " output_seq_len=" << output_seq_len << "\n";

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
                    // Debug: show the state before timestamp logit processing (verbose only)
                    if (cb_verbose()) {
                        const size_t vocab_size_dbg = sg_logits.get_shape().back();
                        const size_t seq_offset = (sg_logits.get_shape()[1] - 1) * vocab_size_dbg;
                        const float* ldata = sg_logits.data<float>() + batch * sg_logits.get_shape()[1] * vocab_size_dbg + seq_offset;
                        const size_t ts_begin = m_whisper_gen_config.no_timestamps_token_id + 1;
                        const int64_t eos_id = m_whisper_gen_config.eos_token_id;
                        float eos_logit = (eos_id >= 0 && (size_t)eos_id < vocab_size_dbg) ? ldata[eos_id] : -1e9f;
                        // compute max over all tokens
                        float max_logit = *std::max_element(ldata, ldata + vocab_size_dbg);
                        // compute log-sum-exp of timestamps
                        float ts_max = -1e30f;
                        for (size_t i = ts_begin; i < vocab_size_dbg; ++i) ts_max = std::max(ts_max, ldata[i]);
                        float ts_sum = 0.f;
                        // use regular softmax max for stability
                        float all_max = max_logit;
                        float denom = 0.f;
                        for (size_t i = 0; i < vocab_size_dbg; ++i) denom += std::exp(ldata[i] - all_max);
                        for (size_t i = ts_begin; i < vocab_size_dbg; ++i) ts_sum += std::exp(ldata[i] - all_max);
                        float ts_prob = ts_sum / denom;
                        // argmax
                        size_t argmax = (size_t)(std::max_element(ldata, ldata + vocab_size_dbg) - ldata);
                        // top-5 tokens
                        std::vector<std::pair<float,size_t>> top;
                        for (size_t ti = 0; ti < vocab_size_dbg; ++ti) top.push_back({ldata[ti], ti});
                        std::partial_sort(top.begin(), top.begin()+5, top.end(),
                            [](auto& a, auto& b){ return a.first > b.first; });
                        std::cout << "[WHISPER_LOGIT_DBG] req=" << sg->get_request_id()
                                  << " gen_len=" << generated_ids.size()
                                  << " initial=" << initial_step
                                  << " argmax=" << argmax << " max_logit=" << max_logit
                                  << " eos(" << eos_id << ")=" << eos_logit
                                  << " ts_prob=" << ts_prob
                                  << " top5=[";
                        for (int ti = 0; ti < 5; ++ti)
                            std::cout << top[ti].second << ":" << top[ti].first << (ti<4?",":"");
                        std::cout << "] last3_gen=[";
                        size_t start = generated_ids.size() > 3 ? generated_ids.size()-3 : 0;
                        for (size_t gi = start; gi < generated_ids.size(); ++gi) std::cout << generated_ids[gi] << (gi+1<generated_ids.size()?",":"");
                        std::cout << "]\n";
                    }
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
            // Release cross-KV slot (swap-to-end compaction, ~245 MB, once per request)
            if (m_cross_kv_cache)
                m_cross_kv_cache->release(request->get_request_id());
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
