// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_utils.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"

auto get_config_for_cache() {
    ov::AnyMap config;
    config.insert({ov::cache_dir("whisper_cache")});
    return config;
}

int main(int argc, char* argv[]) try {

    // Standalone test start

    std::string model_path = "./whisper-large-v3-turbo/openvino_decoder_model.xml";
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

    ov::pass::SDPAToPagedAttention().run_on_model(decoder_model);
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

    return 0;
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
    // Standalone test end 

    /* GenAI embedded test start 
    if (argc < 3 || argc > 4) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<WAV_FILE_PATH>\" <DEVICE>");
    }

    std::filesystem::path models_path = argv[1];
    std::string wav_file_path = argv[2];
    std::string device = (argc == 4) ? argv[3] : "CPU";  // Default to CPU if no device is provided

    ov::AnyMap ov_config;
    if (device == "NPU" || device.find("GPU") != std::string::npos) {  // need to handle cases like "GPU", "GPU.0" and "GPU.1"
        // Cache compiled models on disk for GPU and NPU to save time on the
        // next run. It's not beneficial for CPU.
        ov_config = get_config_for_cache();
    }

    ov::genai::WhisperPipeline pipeline(models_path, device, ov_config);

    ov::genai::WhisperGenerationConfig config = pipeline.get_generation_config();
    // 'task' and 'language' parameters are supported for multilingual models only
    config.language = "<|en|>";  // can switch to <|zh|> for Chinese language
    config.task = "transcribe";
    config.return_timestamps = true;

    // Pipeline expects normalized audio with Sample Rate of 16kHz
    ov::genai::RawSpeechInput raw_speech = utils::audio::read_wav(wav_file_path);
    auto result = pipeline.generate(raw_speech, config);

    std::cout << result << "\n";

    std::cout << std::fixed << std::setprecision(2);
    for (auto& chunk : *result.chunks) {
        std::cout << "timestamps: [" << chunk.start_ts << ", " << chunk.end_ts << "] text: " << chunk.text << "\n";
    }

    // CB flow test
    std::cout << "Creating ContinuousBatchingPipeline...\n";
    auto pipe = ov::genai::ContinuousBatchingPipeline(
        models_path,
        ov::genai::SchedulerConfig{},
        "CPU");
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
     GenAI embedded test end */
