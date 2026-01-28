// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/whisper_pipeline.hpp"
#include "utils.hpp"
#include "whisper/config.hpp"
#include "whisper/feature_extractor.hpp"
#include "whisper/context_tokens.hpp"
#include "whisper/models/decoder.hpp"
#include "whisper/logit_processor.hpp"

namespace ov {
namespace genai {

static constexpr int NUM_LAYERS = 12;
class WhisperSpeechEncoder {
    // TODO: use pool of infer requests for better performance in multithreaded scenarios
    ov::InferRequest m_encoder;
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
        auto compiled_model = core.compile_model(model, device, properties);
        m_encoder = compiled_model.create_infer_request();
        // For now stateful decoder to create init input_ids 
        m_decoder = WhisperDecoder::from_path(models_path, device, properties, m_encoder.get_compiled_model().output("last_hidden_state").get_partial_shape());
    }

    // Encode raw speech into decoder inputs (input_ids and encoder_hidden_state) for each chunk
    std::vector<std::pair<ov::Tensor, ov::Tensor>> encode(const RawSpeechInput& raw_speech, const WhisperContextTokens& context_tokens, const ov::genai::WhisperGenerationConfig& config) {
        std::cout << "Starting SpeechEncoder::encode\n";
        auto input_features = m_feature_extractor.extract(raw_speech);
        std::cout << "Extracted features: " << input_features.n_frames << " frames\n";
        size_t segment_offset = 0;

        OPENVINO_ASSERT(m_feature_extractor.sampling_rate != 0, "Sampling Rate for Feature Extractor is 0");
        const float frame_length_in_seconds =
            static_cast<float>(m_feature_extractor.hop_length) / m_feature_extractor.sampling_rate;

        std::vector<std::pair<ov::Tensor, ov::Tensor>> decoder_inputs;
        for (size_t chunk_offset = 0; chunk_offset < input_features.n_frames; chunk_offset += segment_offset) {
            const float chunk_time_offset = chunk_offset * frame_length_in_seconds;

            auto input_features_chunk = input_features.get_data_with_offset(chunk_offset, m_feature_extractor.nb_max_frames);
            OPENVINO_ASSERT(input_features_chunk.size() == m_feature_extractor.feature_size * m_feature_extractor.nb_max_frames,
                            "Mel spectrogram required size: ",
                            m_feature_extractor.feature_size,
                            " * ",
                            m_feature_extractor.nb_max_frames,
                            ". Actual size: ",
                            input_features_chunk.size(),
                            ".");
            ov::Tensor input_tensor(ov::element::f32, {1, m_feature_extractor.feature_size, m_feature_extractor.nb_max_frames}, input_features_chunk.data());
            m_encoder.set_tensor("input_features", input_tensor);
            std::cout << "Running inference for chunk at offset " << chunk_offset << "\n";
            m_encoder.infer();
            std::cout << "Encoder inference completed\n";

            // reset input tensor
            auto devices = m_encoder.get_compiled_model().get_property(ov::execution_devices);
            OPENVINO_ASSERT(devices.size() > 0, "No execution devices found!");
            size_t batch_size = (devices[0] == "NPU") ? 1 : 0;
            m_encoder.set_tensor("input_features", ov::Tensor(ov::element::f32, {batch_size, m_feature_extractor.feature_size, m_feature_extractor.nb_max_frames}));

            ov::Tensor encoder_output = m_encoder.get_tensor("last_hidden_state");
            ov::Tensor encoder_hidden_state(encoder_output.get_element_type(), encoder_output.get_shape());
            encoder_output.copy_to(encoder_hidden_state);

            ov::genai::RawPerfMetrics raw_metrics;
            auto init_tokens = prepare_init_tokens(encoder_hidden_state, m_decoder, config, false, raw_metrics);
            std::vector<int64_t> chunk_init_tokens = ov::genai::get_prompt_tokens(context_tokens, config, chunk_offset);
            chunk_init_tokens.insert(chunk_init_tokens.end(), init_tokens.begin(), init_tokens.end());
            
            std::cout << "chunk_init_tokens: ";
            for (const auto& token : chunk_init_tokens) {
                std::cout << token << " ";
            }
            std::cout << std::endl;

            // Limit to first 2 tokens only
            //size_t num_tokens = std::min(chunk_init_tokens.size(), size_t(2));
            ov::Tensor input_ids = ov::Tensor{ov::element::i64, {1, chunk_init_tokens.size()}, const_cast<int64_t*>(chunk_init_tokens.data())};

            decoder_inputs.emplace_back(std::make_pair(input_ids, encoder_hidden_state));
            segment_offset = std::min(input_features.n_frames, m_feature_extractor.nb_max_frames);
        }
        std::cout << "SpeechEncoder::encode completed, returning " << decoder_inputs.size() << " chunks\n";
        return decoder_inputs;
    }

    // Merge multiple decoder inputs into single batch tensors
    std::pair<ov::Tensor, ov::Tensor> merge_decoder_inputs(const std::vector<std::pair<ov::Tensor, ov::Tensor>>& decoder_inputs) {
        std::cout << "Starting SpeechEncoder::merge_decoder_inputs\n";
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

    float m_load_time_ms = 0;

    WhisperPipelinePocImpl(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties)
        : m_generation_config(utils::from_config_json_if_exists<WhisperGenerationConfig>(models_path)),
          m_tokenizer{models_path},
          m_speech_encoder{models_path, device, properties},
          m_model_config{models_path / "config.json"} {
            ov::Core core = utils::singleton_core();
            auto decoder_model = core.read_model(models_path / "openvino_decoder_model.xml");
            auto compiled_decoder_model = core.compile_model(decoder_model, device, properties);
            m_request_decoder = compiled_decoder_model.create_infer_request();

            auto decoder_with_past_model = core.read_model(models_path / "openvino_decoder_with_past_model.xml");
            auto compiled_decoder_with_past_model = core.compile_model(decoder_with_past_model, device, properties);
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
        std::cout << "Decoder first inference call\n";
        // Print input tensor information before setting
        std::cout << "Setting encoder_hidden_states - Shape: " << encoder_hidden_state.get_shape() 
              << ", Type: " << encoder_hidden_state.get_element_type() << std::endl;
        std::cout << "Setting input_ids - Shape: " << input_ids.get_shape() 
              << ", Type: " << input_ids.get_element_type() << std::endl;
        
        // Set inputs
        // Create copies to ensure data persistence
        ov::Tensor encoder_hidden_state_copy(encoder_hidden_state.get_element_type(), encoder_hidden_state.get_shape());
        encoder_hidden_state.copy_to(encoder_hidden_state_copy);
        ov::Tensor input_ids_copy(input_ids.get_element_type(), input_ids.get_shape());
        input_ids.copy_to(input_ids_copy);

        // Set the copied tensors
        m_request_decoder.set_tensor("encoder_hidden_states", encoder_hidden_state_copy);
        m_request_decoder.set_tensor("input_ids", input_ids_copy);

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
        std::cout << "Decoder next inference call with last_token_id: " << last_token_id << "\n";
        ov::Tensor input_ids{ov::element::i64, {1, 1}}; // single token input
        //std::cout << "Input_ids tensor shape set to {1, 1}\n";
        // Update input_ids tensor with last_token_id
        int64_t* input_ids_data = input_ids.data<int64_t>();
        input_ids_data[0] = last_token_id;
        m_request_decoder_with_past.set_tensor("input_ids", input_ids);
        //std::cout << "Set input_ids tensor - Shape: " << input_ids.get_shape() 
        //          << ", Type: " << input_ids.get_element_type() << std::endl;

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
        //m_request_decoder = m_request_decoder.get_compiled_model().create_infer_request();
        //m_request_decoder_with_past = m_request_decoder_with_past.get_compiled_model().create_infer_request();
    }

    std::map<size_t, int64_t> batch_decode_first(ov::Tensor& encoder_hidden_state, ov::Tensor& input_ids) {
        std::cout << "Batch Decoder first inference call\n";
        // Set inputs
        m_request_decoder.set_tensor("encoder_hidden_states", encoder_hidden_state);
        m_request_decoder.set_tensor("input_ids", input_ids);

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

WhisperPipelinePoc::~WhisperPipelinePoc() = default;

}  // namespace genai
}  // namespace ov
