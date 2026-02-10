// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_utils.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
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

    if (argc < 3 || argc > 4) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<WAV_FILE_PATH>\" <DEVICE>");
    }
    std::filesystem::path models_path = argv[1];
    std::string wav_file_path = argv[2];
    std::string device = (argc == 4) ? argv[3] : "CPU";  // Default to CPU if no device is provided

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
    ov::genai::RawSpeechInput raw_speech = utils::audio::read_wav(wav_file_path);
    // Broadcast raw_speech n times
    size_t n = 1;
    ov::genai::RawSpeechInput extended_raw_speech;
    extended_raw_speech.reserve(raw_speech.size() * n);
    for (int i = 0; i < n; ++i) {
        extended_raw_speech.insert(extended_raw_speech.end(), raw_speech.begin(), raw_speech.end());
    }
    raw_speech = std::move(extended_raw_speech);

    auto result = pipeline.generate(raw_speech, config);
    std::cout << "Result from standard pipeline implementation: " << result << "\n";

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


    /* Baseline flow test
    std::cout << "Checking baseline pipeline...\n";
    auto start_baseline = std::chrono::high_resolution_clock::now();
    auto result = pipeline.generate(raw_speech, config);
    auto end_baseline = std::chrono::high_resolution_clock::now();
    auto duration_baseline = std::chrono::duration_cast<std::chrono::milliseconds>(end_baseline - start_baseline);
    std::cout << "Baseline pipeline execution time: " << duration_baseline.count() / 1000.0 << " seconds\n";
    std::cout << "Result from current demo implementation: " << result << "\n";
    */

    //std::cout << std::fixed << std::setprecision(2);
    //for (auto& chunk : *result.chunks) {
    //    std::cout << "timestamps: [" << chunk.start_ts << ", " << chunk.end_ts << "] text: " << chunk.text << "\n";
    //}

    /*
    // CB flow test
    std::cout << "Creating ContinuousBatchingPipeline...\n";
  //  ov::AnyMap properties{{"KV_CACHE_PRECISION", "U8"}};
    auto pipe = ov::genai::ContinuousBatchingPipeline(
        models_path,
        ov::genai::SchedulerConfig{},
        "CPU",
        properties
    );
    std::cout << "ContinuousBatchingPipeline created. Adding request\n";

    auto handle = pipe.add_request(1, raw_speech, config);

    std::cout << "Processing requests...\n";

    for (int i = 0; i < 1; ++i) {
        std::cout << "Step " << i + 1 << "\n";
        pipe.step();
        if (!pipe.has_non_finished_requests()) {
            break;
        }
    }

    std::cout << "Reading outputs...\n";
    auto outputs = handle->read_all();
    std::vector<int64_t> generated_ids = outputs[0].generated_ids;
    std::cout << "Generated token IDs: ";
    for (const auto& id : generated_ids) {
        std::cout << id << " ";
    }
    std::cout << "\n";
    */
    // GenAI embedded test end

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
