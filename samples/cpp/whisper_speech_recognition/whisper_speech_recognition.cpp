// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <regex>

#include "audio_utils.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"

auto get_config_for_cache() {
    ov::AnyMap config;
    config.insert({ov::cache_dir("whisper_cache")});
    return config;
}

int main(int argc, char* argv[]) try {
    //std::string model_path = "./whisper-large-v3-turbo/openvino_decoder_model.xml";
    //std::string model_path = "../model_server/models/Qwen/Qwen3-4B/openvino_model.xml";
    //ov::Core core;
    //std::shared_ptr<ov::Model> decoder_model = core.read_model(model_path);
    //ov::pass::StatefulToStateless().run_on_model(decoder_model);
    //ov::save_model(decoder_model, "./whisper-large-v3-turbo/openvino_decoder_with_past_model.xml", false);
    //return 0;

     /*
    // Standalone test start
    std::string model_path = "./whisper-large-v3-turbo/openvino_decoder_model.xml";
    // std::string model_path = "../model_server/models/Qwen/Qwen3-4B/openvino_model.xml";
    ov::Core core;
    std::shared_ptr<ov::Model> decoder_model = core.read_model(model_path);
    std::cout << "Loaded decoder model. Model inputs:" << std::endl;
    auto inputs_sdpa = decoder_model->inputs();
    for (auto& input : inputs_sdpa) {
        try {
            std::cout << "Input: " << input.get_any_name() << "; Type: " << input.get_element_type() << "; Shape: " << input.get_partial_shape() << std::endl;
        } catch (const std::exception& e) {
            std::cout << "\nError while printing input information: " << e.what() << std::endl;
        }
    }
    std::cout << "Model outputs: " << std::endl;
    auto outputs_sdpa = decoder_model->outputs();
    for (auto& output : outputs_sdpa) {
        try {
            std::cout << "Output: " << output.get_any_name() << "; Type: " << output.get_element_type() << "; Shape: " << output.get_partial_shape() << std::endl;
        } catch (const std::exception& e) {
            std::cout << "\nError while printing output information: " << e.what() << std::endl;
        }
    }

    auto infer_request_sdpa = core.compile_model(decoder_model, "CPU").create_infer_request();
    std::cout << "Running empty inference on SDPA model..." << std::endl;
    infer_request_sdpa.infer();
    std::cout << "Inference completed successfully." << std::endl;


    std::cout << "Running statful to stateless transformation..." << std::endl;
    ov::pass::StatefulToStateless().run_on_model(decoder_model);
    std::cout << "Model inputs:" << std::endl;
    auto inputs_stateless = decoder_model->inputs();
    for (auto& input : inputs_stateless) {
        try {
            std::cout << "Input: " << input.get_any_name() << "; Type: " << input.get_element_type() << "; Shape: " << input.get_partial_shape() << std::endl;
        } catch (const std::exception& e) {
            std::cout << "\nError while printing input information: " << e.what() << std::endl;
        }
    }

    std::cout << "Model outputs: " << std::endl;
    auto outputs_stateless = decoder_model->outputs();
    for (auto& output : outputs_stateless) {
        try {
            std::cout << "Output: " << output.get_any_name() << "; Type: " << output.get_element_type() << "; Shape: " << output.get_partial_shape() << std::endl;
        } catch (const std::exception& e) {
            std::cout << "\nError while printing output information: " << e.what() << std::endl;
        }
    }

    ov::AnyMap properties_stateless{{"KV_CACHE_PRECISION", "u8"}};
    auto infer_request_stateless = core.compile_model(decoder_model, "CPU", properties_stateless).create_infer_request();

    std::cout << "Running inference..." << std::endl;
    try {
        infer_request_stateless.infer();
    } catch (const std::exception& e) {
        std::cout << "Error while running standalone infer: " << e.what() << std::endl;
    }
    std::cout << "Inference completed successfully." << std::endl;

    // return 0;

    bool use_per_layer_block_indices_inputs = false;
    bool use_score_outputs = false;
    bool allow_score_aggregation = true;
    bool allow_cache_rotation = false;
    bool allow_xattention = false;
    ov::pass::SDPAToPagedAttention(use_per_layer_block_indices_inputs, use_score_outputs, allow_score_aggregation, allow_cache_rotation, allow_xattention).run_on_model(decoder_model);
    std::cout << "SDPA to PA transformation finished. Model inputs:" << std::endl;
    auto inputs_pa = decoder_model->inputs();
    for (auto& input : inputs_pa) {
        try {
            std::cout << "Input: " << input.get_any_name() << "; Type: " << input.get_element_type() << "; Shape: " << input.get_partial_shape() << std::endl;
        } catch (const std::exception& e) {
            std::cout << "\nError while printing input information: " << e.what() << std::endl;
        }
    }

    std::cout << "Model outputs: " << std::endl;
    auto outputs_pa = decoder_model->outputs();
    for (auto& output : outputs_pa) {
        try {
            std::cout << "Output: " << output.get_any_name() << "; Type: " << output.get_element_type() << "; Shape: " << output.get_partial_shape() << std::endl;
        } catch (const std::exception& e) {
            std::cout << "\nError while printing output information: " << e.what() << std::endl;
        }
    }

    ov::AnyMap properties{{"KV_CACHE_PRECISION", "u8"}};
    auto infer_request_pa = core.compile_model(decoder_model, "CPU", properties).create_infer_request();
    std::cout << "Running empty inference on PA model. Inputs set on infer requests:" << std::endl;
    for (auto& input : inputs_pa) {
        try {
            auto names = input.get_names();
            std::cout << "> possible names: ";
            for (const auto& n : names) {
                std::cout << n << ", ";
            }
            std::cout << "\n";
            std::cout << "Input: " << input.get_any_name() << "; Type: " << infer_request_pa.get_tensor(input.get_any_name()).get_element_type() << "; Shape: " << infer_request_pa.get_tensor(input.get_any_name()).get_shape() << std::endl;
        } catch (const std::exception& e) {
            std::cout << "\nError while printing input information: " << e.what() << std::endl;
        }
    }
    std::cout << "Running inference..." << std::endl;
    try {
        infer_request_pa.infer();
    } catch (const std::exception& e) {
        std::cout << "Error while running standalone infer: " << e.what() << std::endl;
    }
    std::cout << "Inference completed successfully." << std::endl;
*/
    // Standalone test end 
    // 
    // GenAI embedded test start 

    if (argc < 4 || argc > 5) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <WAV_FILE_1> <WAV_FILE_2> [DEVICE]");
    }
    std::filesystem::path models_path = argv[1];
    std::string wav_file_path1 = argv[2];
    std::string wav_file_path2 = argv[3];
    std::string device = (argc == 5) ? argv[4] : "CPU";  // Default to CPU if no device is provided

    ov::AnyMap ov_config;
    if (device == "NPU" ||
        device.find("GPU") != std::string::npos) {  // need to handle cases like "GPU", "GPU.0" and "GPU.1"
        // Cache compiled models on disk for GPU and NPU to save time on the
        // next run. It's not beneficial for CPU.
        ov_config = get_config_for_cache();
    }

    // Word timestamps require decomposition of cross-attention decoder SDPA layers,
    // so word_timestamps must be passed to the pipeline constructor (not just in generation config)
    ov_config.insert(ov::genai::word_timestamps(true));
    ov::genai::WhisperPipeline pipeline(models_path.parent_path() / (models_path.filename().string() + "-stateful"), device, ov_config);

    ov::genai::WhisperGenerationConfig config = pipeline.get_generation_config();
    // 'task' and 'language' parameters are supported for multilingual models only
    config.language = "<|en|>";  // can switch to <|zh|> for Chinese language
    config.task = "transcribe";
    config.return_timestamps = true;
    config.word_timestamps = true;

    // Pipeline expects normalized audio with Sample Rate of 16kHz
    ov::genai::RawSpeechInput raw_speech1 = utils::audio::read_wav(wav_file_path1);
    ov::genai::RawSpeechInput raw_speech2 = utils::audio::read_wav(wav_file_path2);


    /* FULL SDPA CB TEST
    auto result = pipeline.generate(raw_speech, config);
    std::cout << "Result from standard pipeline implementation: " << result << "\n";

    std::cout << "Pipelines warmup...\n";
    ov::genai::WhisperPipelinePoc poc_pipeline_cb(models_path, device);
    poc_pipeline_cb.experimental_generate_with_continuous_batching(raw_speech, config);
    ov::genai::WhisperPipelinePoc poc_pipeline_static_batch(models_path, device);
    poc_pipeline_static_batch.generate(raw_speech, true, config);

    std::cout << "Pipelines warmup completed. Running timed generations in preferential scenario...\n";
     std::cout << "Checking experimental continuous batching pipeline...\n";
    auto start_cb = std::chrono::high_resolution_clock::now();
    poc_pipeline_cb.experimental_generate_with_continuous_batching(raw_speech, config);
    auto end_cb = std::chrono::high_resolution_clock::now();
    auto duration_cb = std::chrono::duration_cast<std::chrono::milliseconds>(end_cb - start_cb);
    std::cout << "POC pipeline experimental continuous batching mode check over. Checking static batching pipeline...\n";
    auto start_static = std::chrono::high_resolution_clock::now();
    poc_pipeline_static_batch.generate(raw_speech, true, config);
    auto end_static = std::chrono::high_resolution_clock::now();
    auto duration_static = std::chrono::duration_cast<std::chrono::milliseconds>(end_static - start_static);
    std::cout << "POC pipeline experimental continuous batching mode execution time: " << duration_cb.count() / 1000.0 << " seconds\n";
    std::cout << "POC pipeline static batching mode execution time: " << duration_static.count() / 1000.0 << " seconds\n";

    */

    /* Static version 
    std::cout << "Loading pipelines\n";
    bool batched_mode = true;
    ov::genai::WhisperPipelinePoc poc_pipeline_seq(models_path, device);
    ov::genai::WhisperPipelinePoc poc_pipeline_batched(models_path, device);
    std::cout << "Running warmup generations...\n";
    poc_pipeline_seq.generate(raw_speech, !batched_mode, config);
    poc_pipeline_batched.generate(raw_speech, batched_mode, config);
    std::cout << "Warmup generation completed. Running timed generations...\n";
    std::cout << "Checking sequential mode...\n";
    auto start = std::chrono::high_resolution_clock::now();
    poc_pipeline_seq.generate(raw_speech, !batched_mode, config);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "POC pipeline sequential mode execution time: " << duration.count() / 1000.0 << " seconds\n";
    std::cout << "POC pipeline sequential mode check over. Checking batched mode...\n";
    auto start_batched = std::chrono::high_resolution_clock::now();
    poc_pipeline_batched.generate(raw_speech, batched_mode, config);
    auto end_batched = std::chrono::high_resolution_clock::now();
    auto duration_batched = std::chrono::duration_cast<std::chrono::milliseconds>(end_batched - start_batched);
    std::cout << "POC pipeline batched mode execution time: " << duration_batched.count() / 1000.0 << " seconds\n";
    */

    /* Baseline flow test */
    std::cout << "Checking baseline pipeline (audio 1)...\n";
    auto result1 = pipeline.generate(raw_speech1, config);
    std::cout << "Baseline result 1: " << result1 << "\n\n";

    std::cout << "Checking baseline pipeline (audio 2)...\n";
    auto result2 = pipeline.generate(raw_speech2, config);
    std::cout << "Baseline result 2: " << result2 << "\n\n";

    //std::cout << std::fixed << std::setprecision(2);
    //for (auto& chunk : *result.chunks) {
    //    std::cout << "timestamps: [" << chunk.start_ts << ", " << chunk.end_ts << "] text: " << chunk.text << "\n";
    //}


// ── CB flow test ────────────────────────────────────────────────────────────
//
// Scenario matrix — with max_num_tokens_per_prefill_step=1, each new Whisper
// request goes through exactly 3 PROMPT steps (one SOT token each) before its
// TRANSITION step, then enters pure GENERATE.  The phase of a request injected
// at step S is therefore:
//
//   steps S..S+2 : PROMPT   (!requires_sampling)
//   step  S+3    : TRANSITION (requires_sampling && !can_generate_tokens)
//   steps S+4+   : GENERATE  (requires_sampling  &&  can_generate_tokens)
//
// The problematic pipeline configurations are:
//   P+G  : PROMPT   of new request coexists with GENERATE of existing one
//   T+G  : TRANSITION of new request coexists with GENERATE of existing one
//   T+T  : Two requests hit TRANSITION at the same step
//   G+G  : Multiple requests in pure GENERATE (should already work)
//
// We cover these by controlling injection offsets precisely:
//
//   Scenario 1 (solo A / solo B)  : baseline — single request only
//   Scenario 2 (A@0, B@0)         : T+T at step 3, G+G from step 4
//   Scenario 3 (A@0, B@1)         : B's T at step 4, A already in G → T+G
//   Scenario 4 (A@0, B@3)         : B's P@3 coincides with A's G@3 → P+G starts at step 3
//                                   B's T at step 6 while A still in G → T+G
//   Scenario 5 (A@0, B@6)         : B deep into P+G overlap, wider G+G window
//   Scenario 6 (A@0, B@0, A@0)    : triple simultaneous start, two T+T at step 3
//   Scenario 7 (A@0,B@0,A@3,B@6,A@9,B@12) : continuous staggered arrival (stress)

    ov::AnyMap pipe_properties{
        {ov::hint::kv_cache_precision.name(), ov::element::u8},
    };
    ov::genai::SchedulerConfig sched_cfg;
    sched_cfg.max_num_batched_tokens = 100;
    sched_cfg.max_num_tokens_per_prefill_step = 1;

    auto make_pipe = [&]() {
        return ov::genai::ContinuousBatchingPipeline(
            models_path.parent_path() / (models_path.filename().string() + "-stateful"),
            sched_cfg,
            "CPU", pipe_properties);
    };

    // ── helpers ──────────────────────────────────────────────────────────────

    // Drain a handle and return all generated token IDs.
    auto collect_tokens = [](ov::genai::GenerationHandle& h) {
        std::vector<int64_t> toks;
        if (!h) return toks;
        while (h->can_read())
            for (auto& [_, out] : h->read())
                toks.insert(toks.end(), out.generated_ids.begin(), out.generated_ids.end());
        return toks;
    };

    // ── Step 1: establish exact-token baselines via solo CB runs ─────────────
    //
    // Using the CB pipeline (not WhisperPipeline) as the reference avoids any
    // discrepancy between the two pipeline stacks.  A solo run has no
    // interleaved-request interactions, so its output is the ground truth.
    std::cout << "\n=== ESTABLISHING BASELINES (solo CB runs) ===\n";
    std::vector<int64_t> baseline_toks_A, baseline_toks_B;
    {
        auto ref_pipe = make_pipe();
        auto ha = ref_pipe.add_request(0, raw_speech1, config);
        while (ref_pipe.has_non_finished_requests()) ref_pipe.step();
        baseline_toks_A = collect_tokens(ha);
    }
    {
        auto ref_pipe = make_pipe();
        auto hb = ref_pipe.add_request(0, raw_speech2, config);
        while (ref_pipe.has_non_finished_requests()) ref_pipe.step();
        baseline_toks_B = collect_tokens(hb);
    }

    ov::AnyMap decode_params = {ov::genai::skip_special_tokens(false)};
    ov::AnyMap decode_text_params = {ov::genai::skip_special_tokens(true)};
    // Single shared pipe used just for the tokenizer (no model inference).
    auto tok_pipe = make_pipe();
    auto decode = [&](const std::vector<int64_t>& toks) -> std::string {
        if (toks.empty()) return "<empty>";
        return tok_pipe.get_tokenizer().decode(toks, decode_params);
    };
    // strip_timestamps: remove Whisper timestamp tokens of the form <|N.NN|> that
    // the OV tokenizer does NOT elide even with skip_special_tokens=true.
    // Without this, a token[24] shift of <|25.00|>→<|24.50|> (BF16 rounding) would
    // make match_text fail even though actual speech content is identical.
    auto strip_timestamps = [](std::string s) -> std::string {
        static const std::regex ts_re("<\\|[0-9]+(?:\\.[0-9]+)?\\|>");
        return std::regex_replace(s, ts_re, "");
    };
    auto decode_text = [&](const std::vector<int64_t>& toks) -> std::string {
        if (toks.empty()) return "<empty>";
        return strip_timestamps(tok_pipe.get_tokenizer().decode(toks, decode_text_params));
    };
    // Reuse a single persistent pipe for all scenarios (KV cache is freed after
    // all requests finish, so subsequent add_request calls start clean).
    auto pipe = make_pipe();

    std::cout << "Baseline A (" << baseline_toks_A.size() << " tokens): "
              << decode(baseline_toks_A) << "\n";
    std::cout << "Baseline B (" << baseline_toks_B.size() << " tokens): "
              << decode(baseline_toks_B) << "\n";

    // ── Step 2: scenario runner ───────────────────────────────────────────────

    // Each request spec: {inject_step, audio_id ('A' or 'B'), label}
    using Audio = char;  // 'A' or 'B'
    struct ReqSpec { int inject_step; Audio audio; std::string label; };

    // Per-request and per-scenario result records (collected for end-of-run summary).
    struct ReqResult {
        std::string label;
        bool ok = true;
        bool text_ok = true;       // decoded text (skip_special_tokens) matches
        size_t expected_toks = 0;
        size_t got_toks = 0;
        size_t diverge_idx = 0;    // only valid when !ok
        int64_t diverge_exp = -1;  // only valid when !ok
        int64_t diverge_got = -1;  // only valid when !ok
        std::string decoded_got;   // only valid when !ok (full decode with specials)
    };
    struct ScenResult {
        std::string name;
        bool ok = true;       // exact token match
        bool text_ok = true;  // text content match (ignores timestamp values)
        std::vector<ReqResult> reqs;
    };

    // Runs a scenario, prints verbose per-step/per-request logs, and returns a
    // ScenResult for inclusion in the end-of-run compact summary.
    auto run_scenario = [&](const std::string& name,
                             const std::vector<ReqSpec>& specs) -> ScenResult {
        std::cout << "\n══════════════════════════════════════════════════════\n";
        std::cout << "SCENARIO: " << name << "\n";
        std::cout << "══════════════════════════════════════════════════════\n";

        std::vector<ov::genai::GenerationHandle> handles(specs.size());
        const int max_steps = 2000;

        for (int step = 0; step < max_steps; ++step) {
            for (size_t ri = 0; ri < specs.size(); ++ri) {
                if (specs[ri].inject_step == step) {
                    std::cout << "  [step " << step << "] injecting "
                              << specs[ri].label << "\n";
                    const auto& speech = (specs[ri].audio == 'A') ? raw_speech1 : raw_speech2;
                    handles[ri] = pipe.add_request(
                        static_cast<uint64_t>(ri + 1), speech, config);
                }
            }
            pipe.step();
            if (!pipe.has_non_finished_requests()) break;
        }

        ScenResult result;
        result.name    = name;
        result.ok      = true;
        result.text_ok = true;

        auto show20 = [](const std::vector<int64_t>& v) {
            std::string s = "[";
            for (size_t i = 0; i < std::min<size_t>(20, v.size()); ++i)
                s += std::to_string(v[i]) + (i + 1 < std::min<size_t>(20, v.size()) ? "," : "");
            if (v.size() > 20) s += ",...";
            return s + "]";
        };

        for (size_t ri = 0; ri < specs.size(); ++ri) {
            const auto& ref_toks = (specs[ri].audio == 'A') ? baseline_toks_A : baseline_toks_B;
            auto got_toks = collect_tokens(handles[ri]);
            const bool match_exact = (got_toks == ref_toks);
            // Text-only match: decode without special tokens (timestamps, SOT, EOT etc.)
            // A mismatch that is text_ok means only timestamp values differ (BF16 rounding).
            const bool match_text  = (decode_text(got_toks) == decode_text(ref_toks));
            result.ok      = result.ok      && match_exact;
            result.text_ok = result.text_ok && match_text;

            ReqResult rr;
            rr.label         = specs[ri].label;
            rr.ok            = match_exact;
            rr.text_ok       = match_text;
            rr.expected_toks = ref_toks.size();
            rr.got_toks      = got_toks.size();

            std::cout << "\n  " << specs[ri].label << "\n";
            std::cout << "    expected " << ref_toks.size()
                      << " tokens, got " << got_toks.size() << " tokens\n";
            if (!match_exact) {
                size_t diverge = 0;
                size_t cmp_len = std::min(ref_toks.size(), got_toks.size());
                while (diverge < cmp_len && ref_toks[diverge] == got_toks[diverge])
                    ++diverge;
                rr.diverge_idx = diverge;
                rr.diverge_exp = (diverge < ref_toks.size() ? ref_toks[diverge] : -1LL);
                rr.diverge_got = (diverge < got_toks.size() ? got_toks[diverge] : -1LL);
                rr.decoded_got = decode(got_toks);
                std::cout << "    FIRST MISMATCH at token[" << diverge << "]: "
                          << "expected " << rr.diverge_exp
                          << ", got "    << rr.diverge_got
                          << "\n";
                std::cout << "    text_ok: " << (match_text ? "YES (timestamp-only drift)" : "NO (real text divergence)") << "\n";
                std::cout << "    expected: " << show20(ref_toks)  << "\n";
                std::cout << "    got     : " << show20(got_toks)  << "\n";
                std::cout << "    decoded : " << rr.decoded_got    << "\n";
            }
            std::cout << (match_exact ? "    [OK]\n" : (match_text ? "    [TIMESTAMP-DRIFT]\n" : "    [MISMATCH]\n"));
            result.reqs.push_back(std::move(rr));
        }
        if (result.ok)           std::cout << "  => PASS\n";
        else if (result.text_ok) std::cout << "  => FAIL (timestamp-drift only)\n";
        else                     std::cout << "  => FAIL\n";
        return result;
    };

    // ── Step 3: run all scenarios ─────────────────────────────────────────────

    bool grand_ok = true;
    std::vector<ScenResult> all_results;
    // Wrapper: runs a scenario and stores its result for the end summary.
    auto run = [&](const std::string& name, const std::vector<ReqSpec>& specs) {
        auto r = run_scenario(name, specs);
        grand_ok &= r.ok;
        all_results.push_back(std::move(r));
    };

    // Scenario 1a: solo A — sanity check that solo CB matches baseline
    run("1a: Solo A (sanity)", {
        {0, 'A', "req1 A@step0"},
    });

    // Scenario 1b: solo B
    run("1b: Solo B (sanity)", {
        {0, 'B', "req1 B@step0"},
    });

    // Scenario 2: A and B both injected at step 0.
    // Both go through PROMPT together (steps 0-2), both hit TRANSITION at step 3
    // (T+T), both enter GENERATE at step 4 (G+G).
    run("2: A@0 + B@0 (T+T at step 3, then G+G)", {
        {0, 'A', "req1 A@step0"},
        {0, 'B', "req2 B@step0"},
    });

    // Scenario 3: A at step 0, B at step 1.
    // A's TRANSITION: step 2 (last SOT).  B's TRANSITION: step 4 — while A is at GENERATE[1].
    // → T+P at step 2, G+T at step 3, then G+G with context-diff=1 from step 4.
    run("3: A@0 + B@1 (B's TRANSITION while A in GENERATE)", {
        {0, 'A', "req1 A@step0"},
        {1, 'B', "req2 B@step1"},
    });

    // Scenario 3b: A at step 0, B at step 2.
    // B's TRANSITION: step 5.  G+G starts with context-diff=2.
    // Discriminator: if 3b passes but 3 fails, the bug is specific to context-diff=1 in G+G.
    // If 3b also fails, the bug is structural to B-injected-during-A's-PROMPT phase.
    run("3b: A@0 + B@2 (G+G starts with context-diff=2)", {
        {0, 'A', "req1 A@step0"},
        {2, 'B', "req2 B@step2"},
    });

    // Scenario 3c: A at step 0, B at step 3.
    // B's TRANSITION: step 6.  G+G starts with context-diff=3 (same as scenario 4 start).
    // If 3b fails but 3c passes, the threshold is exactly context-diff=2.
    run("3c: A@0 + B@3 (same as scenario 4 but A pre-generates 3 tokens before B transitions)", {
        {0, 'A', "req1 A@step0"},
        {3, 'B', "req2 B@step3"},
    });

    // Scenario 3d: REVERSED - B(audio2)@0, A(audio1)@1.
    // Structurally identical to scenario 3 but with audio roles swapped.
    // If the LATER-injected request always fails → expect A to fail at token[2].
    // If the failure is content-specific to audio B → still B fails (but B is now the early one
    // and passes, while A is the late one and may or may not fail).
    run("3d: B@0 + A@1 (reversed injection order vs scenario 3)", {
        {0, 'B', "req1 B@step0"},
        {1, 'A', "req2 A@step1"},
    });

    // Scenario 3e: A@0 + A@1 (both use same audio, staggered by 1 step).
    // If the same-audio case passes: the failure in scenario 3 is audio-content-specific
    //   (cross-audio contamination in segmented cross-attention).
    // If same-audio also fails at token[2]: structural positional issue, not content-dependent.
    run("3e: A@0 + A@1 (same audio, staggered by 1 step)", {
        {0, 'A', "req1 A@step0"},
        {1, 'A', "req2 A@step1"},
    });

    // Scenario 4: A at step 0, B at step 3.
    // A: TRANSITION@3, GENERATE from step 4.
    // B: PROMPT@3-5, TRANSITION@6, GENERATE from step 7.
    // → P+G at steps 3-5, T+G at step 6.
    run("4: A@0 + B@3 (P+G then T+G)", {
        {0, 'A', "req1 A@step0"},
        {3, 'B', "req2 B@step3"},
    });

    // Scenario 5: A at step 0, B at step 6.
    // A well into GENERATE when B starts PROMPT.
    // → P+G at steps 6-8, T+G at step 9, then G+G.
    run("5: A@0 + B@6 (B enters P while A deep in GENERATE)", {
        {0, 'A', "req1 A@step0"},
        {6, 'B', "req2 B@step6"},
    });

    // Scenario 6: Three simultaneous — A, B, A all at step 0.
    // All three hit TRANSITION at step 3 simultaneously (T+T+T).
    run("6: A@0 + B@0 + A@0 (triple simultaneous, T+T+T at step 3)", {
        {0, 'A', "req1 A@step0"},
        {0, 'B', "req2 B@step0"},
        {0, 'A', "req3 A@step0"},
    });

    // Scenario 7: Continuous staggered arrival — new request every 3 steps.
    // Each new request starts its PROMPT exactly when the previous hits TRANSITION.
    // → Maximum overlap of all phase combinations, similar to benchmark high-load.
    run("7: Staggered A@0,B@3,A@6,B@9 (continuous P+T+G overlap)", {
        {0, 'A', "req1 A@step0"},
        {3, 'B', "req2 B@step3"},
        {6, 'A', "req3 A@step6"},
        {9, 'B', "req4 B@step9"},
    });

    // Scenario 8: mirrors the benchmark's high-load pattern
    run("8: Benchmark-style A@0,B@0,A@5,B@10,A@15,B@20", {
        {0,  'A', "req1 A@step0"},
        {0,  'B', "req2 B@step0"},
        {5,  'A', "req3 A@step5"},
        {10, 'B', "req4 B@step10"},
        {15, 'A', "req5 A@step15"},
        {20, 'B', "req6 B@step20"},
    });

    // ── Compact end-of-run summary ────────────────────────────────────────────
    size_t pass_count = 0, text_pass_count = 0;
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║                      SUMMARY                        ║\n";
    std::cout << "╠══════════════════════════════════════════════════════╣\n";
    for (const auto& sr : all_results) {
        if (sr.ok)           ++pass_count;
        if (sr.text_ok)      ++text_pass_count;
        const char* scen_tag = sr.ok ? "PASS" : (sr.text_ok ? "DRIFT" : "FAIL");
        std::cout << "║ " << scen_tag << "  " << sr.name << "\n";
        for (const auto& rr : sr.reqs) {
            if (rr.ok) {
                std::cout << "║      [OK]  " << rr.label
                          << "  (" << rr.got_toks << " tok)\n";
            } else if (rr.text_ok) {
                std::cout << "║      [TIMESTAMP-DRIFT]  " << rr.label
                          << "  exp=" << rr.expected_toks << "tok"
                          << "  got=" << rr.got_toks << "tok"
                          << "  diverge@[" << rr.diverge_idx << "]:"
                          << " exp=" << rr.diverge_exp
                          << " got=" << rr.diverge_got << "  (text OK)\n";
            } else {
                std::cout << "║      [MISMATCH]  " << rr.label
                          << "  exp=" << rr.expected_toks << "tok"
                          << "  got=" << rr.got_toks << "tok"
                          << "  diverge@[" << rr.diverge_idx << "]:"
                          << " exp=" << rr.diverge_exp
                          << " got=" << rr.diverge_got << "\n";
                std::cout << "║           decoded: " << rr.decoded_got << "\n";
            }
        }
    }
    std::cout << "╠══════════════════════════════════════════════════════╣\n";
    std::cout << "║ OVERALL: " << (grand_ok ? "PASS" : "FAIL")
              << "  exact=" << pass_count << "/" << all_results.size()
              << "  text=" << text_pass_count << "/" << all_results.size() << "\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n";
    // ── CB flow test end ─────────────────────────────────────────────────────

} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
}
// */
