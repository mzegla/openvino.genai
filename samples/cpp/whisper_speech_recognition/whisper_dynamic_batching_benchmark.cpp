// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_utils.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "openvino/genai/tokenizer.hpp"
#include <iostream>
#include <regex>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <filesystem>
#include <unistd.h>
#include <fcntl.h>

struct RequestMetrics {
    size_t request_id;
    std::chrono::steady_clock::time_point arrival_time;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point completion_time;
    double queue_time_ms;       // Time waiting in queue
    double processing_time_ms;   // Time being processed
    double total_latency_ms;     // Total time from arrival to completion
    double encode_time_ms = 0.0;        // Wall time of add_request() (CB mode; encoder + PA setup)
    double encoder_infer_ms = 0.0;       // Encoder model infer() time only (both modes)
    double feature_extract_ms = 0.0;     // MEL feature extraction time (both modes)
    double sot_tokens_ms = 0.0;          // prepare_sot_tokens time (both modes)
    std::string text;  // Output text for correctness validation (native mode)
    std::vector<int64_t> tokens;  // Output tokens for correctness validation (CB mode)
};

struct PendingRequest {
    size_t request_id;
    ov::genai::RawSpeechInput audio;
    std::chrono::steady_clock::time_point arrival_time;
};

class DynamicBenchmark {
private:
    std::string model_path;
    std::string device;
    ov::genai::RawSpeechInput base_audio;
    ov::genai::WhisperGenerationConfig config;
    bool is_native_mode;
    ov::genai::Tokenizer tokenizer;
    ov::genai::PipelineMetrics cb_final_metrics;  // captured at end of process_requests_cb

    std::queue<PendingRequest> request_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    
    std::atomic<bool> stop_flag{false};
    std::atomic<size_t> next_request_id{0};
    std::atomic<size_t> scheduled_requests{0};
    std::atomic<size_t> completed_requests{0};
    
    std::vector<RequestMetrics> metrics;
    std::mutex metrics_mutex;
    
public:
    DynamicBenchmark(const std::string& model_path, const std::string& device, 
                     const ov::genai::RawSpeechInput& audio,
                     const ov::genai::WhisperGenerationConfig& config,
                     bool is_native)
        : model_path(model_path), device(device), base_audio(audio), config(config),
          is_native_mode(is_native), tokenizer(model_path) {}
    
    void request_generator(double request_rate, double duration_seconds) {
        auto start_time = std::chrono::steady_clock::now();
        double inter_arrival_time_ms = 1000.0 / request_rate;  // Time between requests in milliseconds
        
        auto next_request_time = start_time;
        
        while (true) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration<double>(now - start_time).count();
            
            if (elapsed >= duration_seconds) {
                break;
            }
            
            // Wait until it's time for the next request
            if (now < next_request_time) {
                std::this_thread::sleep_until(next_request_time);
            }
            
            // Schedule next request
            PendingRequest req;
            req.request_id = next_request_id++;
            req.audio = base_audio;  // Copy audio data
            req.arrival_time = std::chrono::steady_clock::now();
            
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                request_queue.push(req);
                scheduled_requests++;
            }
            queue_cv.notify_one();
            
            // Calculate next request time
            next_request_time += std::chrono::milliseconds(static_cast<int>(inter_arrival_time_ms));
        }
        
        stop_flag = true;
        queue_cv.notify_all();
    }
    
    void request_processor(std::unique_ptr<ov::genai::WhisperPipeline> native_pipeline,
                            std::unique_ptr<ov::genai::ContinuousBatchingPipeline> cb_pipe) {
        if (is_native_mode) {
            process_requests_sequential(native_pipeline.get());
        } else {
            process_requests_cb(std::move(cb_pipe));
        }
    }
    
    // Native mode: process requests sequentially (blocking)
    void process_requests_sequential(ov::genai::WhisperPipeline* pipeline) {
        while (true) {
            PendingRequest req;
            
            // Get next request from queue
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [this] { 
                    return !request_queue.empty() || stop_flag.load(); 
                });
                
                if (request_queue.empty() && stop_flag.load()) {
                    break;
                }
                
                if (request_queue.empty()) {
                    continue;
                }
                
                req = request_queue.front();
                request_queue.pop();
            }
            
            // Process request (blocking)
            auto start_time = std::chrono::steady_clock::now();
            
            try {
                auto result = pipeline->generate(req.audio, config);
                
                auto completion_time = std::chrono::steady_clock::now();
                
                // Record metrics
                RequestMetrics metrics_entry;
                metrics_entry.request_id = req.request_id;
                metrics_entry.arrival_time = req.arrival_time;
                metrics_entry.start_time = start_time;
                metrics_entry.completion_time = completion_time;
                
                metrics_entry.queue_time_ms = std::chrono::duration<double, std::milli>(
                    start_time - req.arrival_time).count();
                metrics_entry.processing_time_ms = std::chrono::duration<double, std::milli>(
                    completion_time - start_time).count();
                metrics_entry.total_latency_ms = std::chrono::duration<double, std::milli>(
                    completion_time - req.arrival_time).count();
                
                // Extract text for correctness validation
                metrics_entry.text = result.texts.empty() ? "" : result.texts[0];

                // Compute encoder component timings from perf_metrics
                {
                    const auto& raw = result.perf_metrics.raw_metrics;
                    double total_infer_ms = raw.m_inference_durations.empty() ? 0.0
                        : raw.m_inference_durations[0].count() / 1000.0;
                    double decoder_infer_ms = 0.0;
                    for (const auto& d : raw.m_token_infer_durations)
                        decoder_infer_ms += d.count() / 1000.0;
                    metrics_entry.encoder_infer_ms = total_infer_ms - decoder_infer_ms;

                    const auto& wraw = result.perf_metrics.whisper_raw_metrics;
                    if (!wraw.features_extraction_durations.empty())
                        metrics_entry.feature_extract_ms = wraw.features_extraction_durations[0].count() / 1000.0;
                }
                
                {
                    std::lock_guard<std::mutex> lock(metrics_mutex);
                    metrics.push_back(metrics_entry);
                }
                
                completed_requests++;
                
            } catch (const std::exception& e) {
                std::cerr << "Error processing request " << req.request_id << ": " << e.what() << "\n";
            }
        }
    }
    
    // Called from run() before the generator starts — builds and warms up the CB pipeline.
    std::unique_ptr<ov::genai::ContinuousBatchingPipeline> setup_cb_pipeline() {
        ov::AnyMap pipe_properties{{ov::hint::kv_cache_precision.name(), ov::element::bf16}};
        ov::genai::SchedulerConfig sched_cfg;
        sched_cfg.max_num_batched_tokens = 100;
        sched_cfg.max_num_tokens_per_prefill_step = 1;
        auto cb_pipe = std::make_unique<ov::genai::ContinuousBatchingPipeline>(
            std::filesystem::path(model_path + "-stateful"),
            sched_cfg, device, pipe_properties);
        {
            auto warmup_cfg = static_cast<ov::genai::WhisperGenerationConfig>(config);
            auto warmup_handle = cb_pipe->add_request(static_cast<uint64_t>(0xFFFFFFFFFFFFFFFFull), base_audio, warmup_cfg);
            while (cb_pipe->has_non_finished_requests())
                cb_pipe->step();
        }
        return cb_pipe;
    }

    // Continuous batching mode: single-threaded step loop using ContinuousBatchingPipeline
    void process_requests_cb(std::unique_ptr<ov::genai::ContinuousBatchingPipeline> cb_pipe) {

        // Per-request tracking
        struct ActiveRequest {
            size_t request_id;
            std::chrono::steady_clock::time_point arrival_time;
            std::chrono::steady_clock::time_point start_time;
            double encode_time_ms = 0.0;       // wall time of add_request() (encoder + PA setup)
            double encoder_infer_ms = 0.0;      // encoder model infer() only
            double feature_extract_ms = 0.0;    // MEL feature extraction
            double sot_tokens_ms = 0.0;         // prepare_sot_tokens
            ov::genai::GenerationHandle handle;
            std::vector<int64_t> tokens;
        };
        std::vector<ActiveRequest> active;

        // Requests that have been encoded and are ready to add to the CB pipeline.
        // Written by encoding threads, drained by the step loop thread.
        struct EncodedRequest {
            size_t request_id;
            std::chrono::steady_clock::time_point arrival_time;
            std::chrono::steady_clock::time_point encode_start_time;
            double encode_time_ms     = 0.0;
            double encoder_infer_ms   = 0.0;
            double feature_extract_ms = 0.0;
            double sot_tokens_ms      = 0.0;
            ov::Tensor input_ids;
            ov::Tensor encoder_hidden_states;
        };
        std::vector<EncodedRequest> encoded_ready;
        std::mutex encoded_ready_mutex;
        std::atomic<int> encoding_threads_in_flight{0};

        // Kick off an encoding background thread per pending raw request.
        // Only encode_speech() is called from the thread — fully thread-safe.
        // The resulting EncodedRequest is queued; the step loop thread calls
        // the fast add_request(input_ids, encoder_hs, config) without concurrency.
        auto launch_encoding = [&](std::vector<PendingRequest>& pending) {
            for (auto& req : pending) {
                encoding_threads_in_flight++;
                std::thread([&cb_pipe, &encoded_ready, &encoded_ready_mutex,
                             &encoding_threads_in_flight, req, cfg = config]() mutable {
                    EncodedRequest er;
                    er.request_id        = req.request_id;
                    er.arrival_time      = req.arrival_time;
                    er.encode_start_time = std::chrono::steady_clock::now();

                    const auto t0 = std::chrono::steady_clock::now();
                    // encode_speech() is serialized internally by m_speech_encoder_mutex
                    auto [input_ids, encoder_hs] = cb_pipe->encode_speech(
                        req.audio, static_cast<ov::genai::WhisperGenerationConfig>(cfg));
                    er.encode_time_ms     = std::chrono::duration<double, std::milli>(
                                               std::chrono::steady_clock::now() - t0).count();
                    er.encoder_infer_ms   = cb_pipe->get_last_encoder_infer_ms();
                    er.feature_extract_ms = cb_pipe->get_last_feature_extract_ms();
                    er.sot_tokens_ms      = cb_pipe->get_last_sot_tokens_ms();
                    er.input_ids          = std::move(input_ids);
                    er.encoder_hidden_states = std::move(encoder_hs);

                    {
                        std::lock_guard<std::mutex> lk(encoded_ready_mutex);
                        encoded_ready.push_back(std::move(er));
                    }
                    encoding_threads_in_flight--;
                }).detach();
            }
        };

        // Drain the raw-request queue → launch encoding threads.
        auto ingest_queue = [&]() {
            std::vector<PendingRequest> pending;
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                while (!request_queue.empty()) {
                    pending.push_back(request_queue.front());
                    request_queue.pop();
                }
            }
            launch_encoding(pending);
        };

        // Move encoded-and-ready requests into 'active' via the fast add_request.
        // Called exclusively from the step loop thread → no concurrency with step().
        auto drain_ready = [&]() {
            std::vector<EncodedRequest> batch;
            {
                std::lock_guard<std::mutex> lk(encoded_ready_mutex);
                batch.swap(encoded_ready);
            }
            for (auto& er : batch) {
                const auto t_add0 = std::chrono::steady_clock::now();
                auto req_cfg = static_cast<ov::genai::WhisperGenerationConfig>(config);
                // Fast path: no encoding, just schedules the pre-computed tensors.
                auto handle = cb_pipe->add_request(
                    static_cast<uint64_t>(er.request_id),
                    er.input_ids, er.encoder_hidden_states, req_cfg);
                const double add_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t_add0).count();

                ActiveRequest ar;
                ar.request_id         = er.request_id;
                ar.arrival_time       = er.arrival_time;
                ar.start_time         = er.encode_start_time;
                ar.encode_time_ms     = er.encode_time_ms;
                ar.encoder_infer_ms   = er.encoder_infer_ms;
                ar.feature_extract_ms = er.feature_extract_ms;
                ar.sot_tokens_ms      = er.sot_tokens_ms;
                ar.handle             = std::move(handle);
                std::cout << "[+] Request #" << ar.request_id
                          << " added  (active: " << active.size() + 1
                          << ", enc: " << std::fixed << std::setprecision(0)
                          << ar.encode_time_ms << " ms, add: " << add_ms << " ms)\n";
                active.push_back(std::move(ar));
            }
        };

        auto collect_finished = [&]() {
            for (auto it = active.begin(); it != active.end(); ) {
                while (it->handle->can_read()) {
                    auto outputs = it->handle->read();
                    for (auto& [seq_id, output] : outputs)
                        it->tokens.insert(it->tokens.end(),
                            output.generated_ids.begin(), output.generated_ids.end());
                }
                if (it->handle->get_status() == ov::genai::GenerationStatus::FINISHED) {
                    auto completion_time = std::chrono::steady_clock::now();
                    RequestMetrics m;
                    m.request_id         = it->request_id;
                    m.arrival_time       = it->arrival_time;
                    m.start_time         = it->start_time;
                    m.completion_time    = completion_time;
                    m.queue_time_ms      = std::chrono::duration<double, std::milli>(
                                               it->start_time - it->arrival_time).count();
                    m.processing_time_ms = std::chrono::duration<double, std::milli>(
                                               completion_time - it->start_time).count();
                    m.total_latency_ms   = std::chrono::duration<double, std::milli>(
                                               completion_time - it->arrival_time).count();
                    m.encode_time_ms     = it->encode_time_ms;
                    m.encoder_infer_ms   = it->encoder_infer_ms;
                    m.feature_extract_ms = it->feature_extract_ms;
                    m.sot_tokens_ms      = it->sot_tokens_ms;
                    m.tokens = std::move(it->tokens);
                    {
                        std::lock_guard<std::mutex> lock(metrics_mutex);
                        metrics.push_back(std::move(m));
                    }
                    completed_requests++;
                    it = active.erase(it);
                } else {
                    ++it;
                }
            }
        };

        // Main step loop: ingest new requests, step, collect
        while (true) {
            ingest_queue();
            drain_ready();

            if (active.empty()) {
                if (stop_flag.load() && encoding_threads_in_flight.load() == 0) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            std::cout << "[Step] Active: " << active.size() << " request(s)\n";
            cb_pipe->step();
            collect_finished();
        }

        // Flush any requests that arrived right as stop_flag was raised
        ingest_queue();
        // Wait for all encoding threads to complete before final step loop
        while (encoding_threads_in_flight.load() > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        drain_ready();
        while (!active.empty()) {
            std::cout << "[Step] Active: " << active.size() << " request(s)\n";
            cb_pipe->step();
            collect_finished();
        }
        cb_final_metrics = cb_pipe->get_metrics();
    }
    
    void run(double request_rate, double duration_seconds) {
        std::cout << "\n========================================\n";
        std::cout << "Dynamic Batching Benchmark\n";
        std::cout << "Mode: " << (is_native_mode ? "NATIVE (sequential)" : "CONTINUOUS BATCHING") << "\n";
        std::cout << "========================================\n";
        std::cout << "Request rate: " << request_rate << " req/s\n";
        std::cout << "Duration: " << duration_seconds << " s\n";
        std::cout << "Expected requests: " << static_cast<int>(request_rate * duration_seconds) << "\n";
        std::cout << "========================================\n\n";

        // Build and warm up the pipeline BEFORE the generator starts so that
        // no requests pile up in the queue during model loading / warmup.
        std::unique_ptr<ov::genai::WhisperPipeline>            native_pipeline;
        std::unique_ptr<ov::genai::ContinuousBatchingPipeline> cb_pipe;
        if (is_native_mode) {
            native_pipeline = std::make_unique<ov::genai::WhisperPipeline>(model_path + "-stateful", device);
            native_pipeline->generate(base_audio, config);
        } else {
            cb_pipe = setup_cb_pipeline();
        }

        auto start_time = std::chrono::steady_clock::now();

        // Start processor thread (pipeline already warm)
        std::thread processor_thread(&DynamicBenchmark::request_processor, this,
                                     std::move(native_pipeline), std::move(cb_pipe));

        // Run generator in main thread
        request_generator(request_rate, duration_seconds);

        // Wait for all requests to complete
        processor_thread.join();
        
        auto end_time = std::chrono::steady_clock::now();
        auto total_duration = std::chrono::duration<double>(end_time - start_time).count();
        
        // Print summary
        print_summary(total_duration);
    }

    // Burst mode: all N requests are injected at once before processing begins.
    // For CB this gives the maximum possible batch size from the first step;
    // for native it simply queues everything and processes sequentially.
    void run_burst(size_t n_requests) {
        std::cout << "\n========================================\n";
        std::cout << "Dynamic Batching Benchmark\n";
        std::cout << "Mode: " << (is_native_mode ? "NATIVE (sequential)" : "CONTINUOUS BATCHING") << "\n";
        std::cout << "BURST: " << n_requests << " requests, all added simultaneously\n";
        std::cout << "========================================\n\n";

        std::unique_ptr<ov::genai::WhisperPipeline>            native_pipeline;
        std::unique_ptr<ov::genai::ContinuousBatchingPipeline> cb_pipe;
        if (is_native_mode) {
            native_pipeline = std::make_unique<ov::genai::WhisperPipeline>(model_path + "-stateful", device);
            native_pipeline->generate(base_audio, config);
        } else {
            cb_pipe = setup_cb_pipeline();
        }

        // Enqueue all requests at time zero before starting the processor.
        auto burst_time = std::chrono::steady_clock::now();
        for (size_t i = 0; i < n_requests; ++i) {
            PendingRequest req;
            req.request_id  = next_request_id++;
            req.audio       = base_audio;
            req.arrival_time = burst_time;
            request_queue.push(req);
            scheduled_requests++;
        }
        // Signal that no further requests will arrive.
        stop_flag = true;

        auto start_time = std::chrono::steady_clock::now();

        std::thread processor_thread(&DynamicBenchmark::request_processor, this,
                                     std::move(native_pipeline), std::move(cb_pipe));
        // Unblock the queue_cv wait in the sequential processor.
        queue_cv.notify_all();
        processor_thread.join();

        auto end_time = std::chrono::steady_clock::now();
        double total_duration = std::chrono::duration<double>(end_time - start_time).count();

        print_summary(total_duration);
    }

    void print_summary(double total_duration_seconds) {
        std::lock_guard<std::mutex> lock(metrics_mutex);

        // Strip Whisper timestamp tokens (<|N.NN|>) from a decoded string so that
        // BF16 rounding shifts in timestamp values don't count as text mismatches.
        auto strip_timestamps = [](const std::string& s) -> std::string {
            static const std::regex re("<\\|[0-9]+(?:\\.[0-9]+)?\\|>");
            return std::regex_replace(s, re, "");
        };

        if (metrics.empty()) {
            std::cout << "\nNo requests completed!\n";
            return;
        }
        
        // Calculate statistics
        std::vector<double> latencies;
        std::vector<double> queue_times;
        std::vector<double> processing_times;
        std::vector<double> encode_times;
        std::vector<double> encoder_infer_times;
        std::vector<double> feature_extract_times;
        std::vector<double> sot_tokens_times;
        
        for (const auto& m : metrics) {
            latencies.push_back(m.total_latency_ms);
            queue_times.push_back(m.queue_time_ms);
            processing_times.push_back(m.processing_time_ms);
            if (m.encode_time_ms > 0.0)
                encode_times.push_back(m.encode_time_ms);
            if (m.encoder_infer_ms > 0.0)
                encoder_infer_times.push_back(m.encoder_infer_ms);
            if (m.feature_extract_ms > 0.0)
                feature_extract_times.push_back(m.feature_extract_ms);
            if (m.sot_tokens_ms > 0.0)
                sot_tokens_times.push_back(m.sot_tokens_ms);
        }
        
        std::sort(latencies.begin(), latencies.end());
        std::sort(queue_times.begin(), queue_times.end());
        std::sort(processing_times.begin(), processing_times.end());
        std::sort(encode_times.begin(), encode_times.end());
        
        auto percentile = [](const std::vector<double>& sorted_data, double p) {
            size_t idx = static_cast<size_t>(sorted_data.size() * p);
            if (idx >= sorted_data.size()) idx = sorted_data.size() - 1;
            return sorted_data[idx];
        };
        
        double avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double avg_queue = std::accumulate(queue_times.begin(), queue_times.end(), 0.0) / queue_times.size();
        double avg_processing = std::accumulate(processing_times.begin(), processing_times.end(), 0.0) / processing_times.size();
        double avg_encode = encode_times.empty() ? 0.0
            : std::accumulate(encode_times.begin(), encode_times.end(), 0.0) / encode_times.size();
        double avg_encoder_infer = encoder_infer_times.empty() ? 0.0
            : std::accumulate(encoder_infer_times.begin(), encoder_infer_times.end(), 0.0) / encoder_infer_times.size();
        double avg_feature_extract = feature_extract_times.empty() ? 0.0
            : std::accumulate(feature_extract_times.begin(), feature_extract_times.end(), 0.0) / feature_extract_times.size();
        double avg_sot_tokens = sot_tokens_times.empty() ? 0.0
            : std::accumulate(sot_tokens_times.begin(), sot_tokens_times.end(), 0.0) / sot_tokens_times.size();
        
        double throughput = metrics.size() / total_duration_seconds;

        // Correctness validation first — tokenizer.decode() emits [OV-DBG] lines to
        // stdout, so run this section before printing the summary numbers so the
        // stats always appear at the bottom of the output.
        std::cout << "\n--- Correctness Validation ---\n";
        if (!metrics.empty()) {
            size_t correct_count = 0;
            size_t mismatches = 0;
            
            // Check if we're in native mode (text) or CB mode (tokens)
            bool is_text_mode = !metrics[0].text.empty();
            bool is_token_mode = !metrics[0].tokens.empty();
            
            if (is_text_mode) {
                // Native mode: compare text
                const auto& reference_text = metrics[0].text;
                const std::string reference_text_stripped = strip_timestamps(reference_text);
                std::cout << "Mode: Native (text comparison)\n";
                std::cout << "Reference text (request 0): \"" << reference_text << "\"\n";
                std::cout << "Reference text (stripped):  \"" << reference_text_stripped << "\"\n\n";

                size_t text_match_count = 0;
                for (const auto& m : metrics) {
                    if (m.text == reference_text) {
                        correct_count++;
                    } else {
                        mismatches++;
                        if (mismatches <= 5) {
                            std::cout << "  Mismatch in request " << m.request_id
                                      << ": got \"" << m.text.substr(0, 60)
                                      << (m.text.size() > 60 ? "..." : "") << "\"\n";
                        }
                    }
                    if (strip_timestamps(m.text) == reference_text_stripped)
                        text_match_count++;
                }

                double exact_rate = 100.0 * correct_count / metrics.size();
                double text_rate  = 100.0 * text_match_count  / metrics.size();
                std::cout << "\nExact match:  " << correct_count    << " / " << metrics.size()
                          << " (" << exact_rate << "%)\n";
                std::cout << "Text match:   " << text_match_count  << " / " << metrics.size()
                          << " (" << text_rate  << "%)  [timestamps stripped]\n";

                if (exact_rate == 100.0) {
                    std::cout << "✓ All transcriptions identical (exact)\n";
                } else if (text_rate == 100.0) {
                    std::cout << "~ All text identical, timestamp values differ (BF16 drift)\n";
                } else {
                    std::cout << "✗ Correctness check FAILED: " << mismatches << " exact mismatches, "
                              << (metrics.size() - text_match_count) << " text mismatches\n";
                }
            } else if (is_token_mode) {
                // Continuous batching mode: compare token sequences
                const ov::AnyMap decode_params = {ov::genai::skip_special_tokens(false)};
                const ov::AnyMap decode_text_params = {ov::genai::skip_special_tokens(true)};

                // The OV tokenizer emits [OV-DBG] lines directly to stdout fd 1.
                // Suppress them by temporarily redirecting fd 1 to /dev/null.
                auto silent_decode = [&](const std::vector<int64_t>& toks, const ov::AnyMap& params) -> std::string {
                    std::cout.flush();
                    int saved = dup(STDOUT_FILENO);
                    int devnull = open("/dev/null", O_WRONLY);
                    dup2(devnull, STDOUT_FILENO);
                    close(devnull);
                    std::string result = tokenizer.decode(toks, params);
                    fflush(stdout);
                    dup2(saved, STDOUT_FILENO);
                    close(saved);
                    return result;
                };

                const auto& reference_tokens = metrics[0].tokens;
                std::string reference_decoded = silent_decode(reference_tokens, decode_params);
                std::string reference_text_stripped = strip_timestamps(
                    silent_decode(reference_tokens, decode_text_params));

                std::cout << "Mode: Continuous Batching (token comparison)\n";
                std::cout << "Reference (request 0): " << reference_tokens.size() << " tokens\n";
                std::cout << "  Tokens: [";
                for (size_t i = 0; i < std::min(reference_tokens.size(), size_t(20)); ++i) {
                    std::cout << reference_tokens[i];
                    if (i < std::min(reference_tokens.size(), size_t(20)) - 1) std::cout << ", ";
                }
                if (reference_tokens.size() > 20) std::cout << " ... (" << reference_tokens.size() - 20 << " more)";
                std::cout << "]\n";
                std::cout << "  Decoded (with timestamps): \"" << reference_decoded << "\"\n";
                std::cout << "  Decoded (text only):       \"" << reference_text_stripped << "\"\n\n";

                size_t text_match_count = 0;
                for (const auto& m : metrics) {
                    if (m.tokens == reference_tokens) {
                        correct_count++;
                    } else {
                        mismatches++;
                        if (mismatches <= 5) {
                            std::string decoded = silent_decode(m.tokens, decode_params);
                            std::cout << "  Mismatch in request " << m.request_id
                                      << ": expected " << reference_tokens.size()
                                      << " tokens, got " << m.tokens.size() << " tokens\n";
                            std::cout << "    Tokens: [";
                            for (size_t i = 0; i < std::min(m.tokens.size(), size_t(20)); ++i) {
                                std::cout << m.tokens[i];
                                if (i < std::min(m.tokens.size(), size_t(20)) - 1) std::cout << ", ";
                            }
                            std::cout << "]\n";
                            std::cout << "    Decoded: \"" << decoded << "\"\n";
                        }
                    }
                    std::string got_stripped = strip_timestamps(
                        silent_decode(m.tokens, decode_text_params));
                    if (got_stripped == reference_text_stripped)
                        text_match_count++;
                }

                double exact_rate = 100.0 * correct_count   / metrics.size();
                double text_rate  = 100.0 * text_match_count / metrics.size();
                std::cout << "\nExact match:  " << correct_count    << " / " << metrics.size()
                          << " (" << exact_rate << "%)\n";
                std::cout << "Text match:   " << text_match_count  << " / " << metrics.size()
                          << " (" << text_rate  << "%)  [timestamps stripped]\n";

                if (exact_rate == 100.0) {
                    std::cout << "✓ All token sequences identical (exact)\n";
                } else if (text_rate == 100.0) {
                    std::cout << "~ All text identical, timestamp values differ (BF16 drift)\n";
                } else {
                    std::cout << "✗ Correctness check FAILED: " << mismatches << " exact mismatches, "
                              << (metrics.size() - text_match_count) << " text mismatches\n";
                }
            } else {
                std::cout << "No validation data available\n";
            }
        } else {
            std::cout << "No requests completed for validation\n";
        }

        // Print the main stats after correctness validation so they are visible
        // at the bottom of the output (not buried by [OV-DBG] tokenizer lines).
        std::cout << "\n========================================\n";
        std::cout << "Benchmark Results (" << (is_native_mode ? "NATIVE" : "CONTINUOUS BATCHING") << ")\n";
        std::cout << "========================================\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Total duration: " << total_duration_seconds << " s\n";
        std::cout << "Scheduled requests: " << scheduled_requests.load() << "\n";
        std::cout << "Completed requests: " << completed_requests.load() << "\n";
        std::cout << "Success rate: " << (100.0 * completed_requests / scheduled_requests) << "%\n";
        std::cout << "Throughput: " << throughput << " req/s\n";
        std::cout << "\n--- Latency Statistics (ms) ---\n";
        std::cout << "Average: " << avg_latency << " ms\n";
        std::cout << "Median (p50): " << percentile(latencies, 0.50) << " ms\n";
        std::cout << "p90: " << percentile(latencies, 0.90) << " ms\n";
        std::cout << "p95: " << percentile(latencies, 0.95) << " ms\n";
        std::cout << "p99: " << percentile(latencies, 0.99) << " ms\n";
        std::cout << "Min: " << latencies.front() << " ms\n";
        std::cout << "Max: " << latencies.back() << " ms\n";
        std::cout << "\n--- Queue Time (ms) ---\n";
        std::cout << "Average: " << avg_queue << " ms\n";
        std::cout << "p50: " << percentile(queue_times, 0.50) << " ms\n";
        std::cout << "p95: " << percentile(queue_times, 0.95) << " ms\n";
        std::cout << "\n--- Processing Time (ms) ---\n";
        std::cout << "Average: " << avg_processing << " ms\n";
        std::cout << "p50: " << percentile(processing_times, 0.50) << " ms\n";
        std::cout << "p95: " << percentile(processing_times, 0.95) << " ms\n";

        if (!encode_times.empty()) {
            std::cout << "\n--- Encoding / add_request Time (ms) ---\n";
            std::cout << "Average: " << avg_encode << " ms\n";
            std::cout << "p50: " << percentile(encode_times, 0.50) << " ms\n";
            std::cout << "p95: " << percentile(encode_times, 0.95) << " ms\n";
            std::cout << "(note: includes encoder inference + PA slot allocation)\n";
        }

        if (!encoder_infer_times.empty()) {
            std::sort(encoder_infer_times.begin(), encoder_infer_times.end());
            std::cout << "\n--- Encoder Breakdown (avg ms) ---\n";
            std::cout << "  Feature extraction (MEL/FFT): " << avg_feature_extract << " ms\n";
            std::cout << "  Encoder model infer():        " << avg_encoder_infer    << " ms\n";
            std::cout << "  prepare_sot_tokens():         " << avg_sot_tokens       << " ms\n";
            if (avg_feature_extract + avg_encoder_infer + avg_sot_tokens > 0.0) {
                std::cout << "  Total accounted:              "
                          << avg_feature_extract + avg_encoder_infer + avg_sot_tokens << " ms\n";
            }
        }

        // --- CB Step Timing Breakdown ---
        // Only printed for CB mode; cb_final_metrics is zero for native runs.
        if (!is_native_mode && cb_final_metrics.total_steps > 0) {
            const double total_fwd_us = cb_final_metrics.prompt_phase_us_total
                                      + cb_final_metrics.generate_phase_us_total;
            const double infer_us     = cb_final_metrics.ov_infer_us_total;
            const double assembly_us  = cb_final_metrics.cross_attn_assembly_us_total;
            const double proj_us      = cb_final_metrics.cross_kv_proj_us_total;
            auto pct = [](double num, double den) -> double {
                return den > 0.0 ? 100.0 * num / den : 0.0;
            };

            const double avg_gen_batch = cb_final_metrics.generate_steps > 0
                ? static_cast<double>(cb_final_metrics.generate_batch_token_sum)
                  / cb_final_metrics.generate_steps
                : 0.0;

            std::cout << std::fixed << std::setprecision(1);
            std::cout << "\n--- CB Step Timing Breakdown ---\n";
            std::cout << "Total forward steps:   " << cb_final_metrics.total_steps << "\n";
            std::cout << "  GENERATE steps:      " << cb_final_metrics.generate_steps
                      << "  (avg batch: " << std::setprecision(2) << avg_gen_batch << " seqs/step)\n";
            std::cout << std::setprecision(1);
            std::cout << "\nCumulative forward wall time:  " << total_fwd_us / 1000.0 << " ms\n";
            std::cout << "  PROMPT  phase:       " << cb_final_metrics.prompt_phase_us_total / 1000.0
                      << " ms  (" << pct(cb_final_metrics.prompt_phase_us_total, total_fwd_us) << "%)\n";
            std::cout << "  GENERATE phase:      " << cb_final_metrics.generate_phase_us_total / 1000.0
                      << " ms  (" << pct(cb_final_metrics.generate_phase_us_total, total_fwd_us) << "%)\n";
            std::cout << "\nOV kernel (infer):     " << infer_us / 1000.0
                      << " ms  (" << pct(infer_us, total_fwd_us) << "% of fwd time)\n";
            std::cout << "Cross-attn assembly:   " << assembly_us / 1000.0
                      << " ms  (" << pct(assembly_us, total_fwd_us) << "% of fwd time,  "
                      << pct(assembly_us, infer_us) << "% of infer)\n";
            std::cout << "Other overhead:        "
                      << (total_fwd_us - infer_us - assembly_us) / 1000.0
                      << " ms\n";
            if (proj_us > 0.0) {
                const size_t n_reqs = completed_requests.load();
                std::cout << "\nCrossKV projector:     " << proj_us / 1000.0
                          << " ms total  (avg " << std::setprecision(1)
                          << (n_reqs > 0 ? proj_us / 1000.0 / n_reqs : 0.0)
                          << " ms/req × " << n_reqs << " reqs)\n";
                std::cout << "  (this runs in background after add_request(); not included in\n"
                             "   forward wall time above — but delays scheduler visibility)\n";
            }
            std::cout << "\n[KEY METRIC] Cross-attn assembly / OV infer ratio: "
                      << std::setprecision(1) << pct(assembly_us, infer_us) << "%\n";
            std::cout << "  (If this is >10% at N>1 concurrent requests, K/V projection\n"
                         "   caching (Option 1 in whisper_cb_copilot_recommendation.md)\n"
                         "   is the right next step.)\n";
        }

        std::cout << "========================================\n";
    }
};

int main(int argc, char* argv[]) try {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <audio_file> <request_rate> <duration_seconds> [device] [--is-native] [--burst N]\n";
        std::cerr << "  model_path: Path to the Whisper model directory\n";
        std::cerr << "  audio_file: Path to the WAV audio file\n";
        std::cerr << "  request_rate: Number of requests per second (e.g., 2.0)\n";
        std::cerr << "  duration_seconds: Benchmark duration in seconds (e.g., 60)\n";
        std::cerr << "  device: Optional device (default: CPU)\n";
        std::cerr << "  --is-native: Use native WhisperPipeline (sequential) instead of continuous batching\n";
        std::cerr << "  --burst N: Add all N requests at once (ignores request_rate and duration_seconds)\n";
        std::cerr << "\nExamples:\n";
        std::cerr << "  Continuous batching: " << argv[0] << " ./whisper-base-openvino ./sample.wav 2.0 30\n";
        std::cerr << "  Native sequential:   " << argv[0] << " ./whisper-base-openvino ./sample.wav 2.0 30 CPU --is-native\n";
        std::cerr << "  Burst (CB):          " << argv[0] << " ./whisper-base-openvino ./sample.wav 0 0 CPU --burst 10\n";
        return EXIT_FAILURE;
    }
    
    std::filesystem::path models_path = argv[1];
    std::string wav_file_path = argv[2];
    double request_rate = std::stod(argv[3]);
    double duration_seconds = std::stod(argv[4]);
    
    // Parse optional arguments
    std::string device = "CPU";
    bool is_native_mode = false;
    int burst_count = -1;  // -1 means not set
    
    for (int i = 5; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--is-native") {
            is_native_mode = true;
        } else if (arg == "--burst") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --burst requires an integer argument\n";
                return EXIT_FAILURE;
            }
            burst_count = std::stoi(argv[++i]);
            if (burst_count <= 0) {
                std::cerr << "Error: --burst count must be positive\n";
                return EXIT_FAILURE;
            }
        } else if (i == 5 && arg[0] != '-') {  // First optional positional arg is device
            device = arg;
        }
    }

    if (burst_count < 0 && (request_rate <= 0 || duration_seconds <= 0)) {
        std::cerr << "Error: request_rate and duration must be positive (or use --burst N)\n";
        return EXIT_FAILURE;
    }
    
    std::cout << "Loading audio file: " << wav_file_path << "\n";
    ov::genai::RawSpeechInput raw_speech = utils::audio::read_wav(wav_file_path);
    std::cout << "Audio loaded: " << raw_speech.size() << " samples\n";
    
    // Setup generation config — use a local scope so the WhisperPipeline is
    // fully destroyed before DynamicBenchmark (and its CB pipeline) is created.
    // Keeping two compiled CPU models alive simultaneously can cause OV thread-pool
    // contention that corrupts CB inference when concurrent requests are in flight.
    ov::genai::WhisperGenerationConfig config;
    {
        ov::genai::WhisperPipeline config_pipeline(
            models_path.parent_path() / (models_path.filename().string() + "-stateful"), device);
        config = config_pipeline.get_generation_config();
    }
    config.language = "<|en|>";
    config.task = "transcribe";
    config.return_timestamps = true;
    config.word_timestamps = false;  // CB pipeline doesn't produce word-level timestamps
    config.max_new_tokens = 50;

    // Run benchmark
    DynamicBenchmark benchmark(models_path, device, raw_speech, config, is_native_mode);
    if (burst_count > 0) {
        benchmark.run_burst(static_cast<size_t>(burst_count));
    } else {
        benchmark.run(request_rate, duration_seconds);
    }
    
    return EXIT_SUCCESS;
    
} catch (const std::exception& error) {
    try {
        std::cerr << "Error: " << error.what() << '\n';
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
