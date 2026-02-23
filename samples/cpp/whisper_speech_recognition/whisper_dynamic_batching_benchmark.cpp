// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_utils.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <algorithm>
#include <numeric>
#include <iomanip>

struct RequestMetrics {
    size_t request_id;
    std::chrono::steady_clock::time_point arrival_time;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point completion_time;
    double queue_time_ms;  // Time waiting in queue
    double processing_time_ms;  // Time being processed
    double total_latency_ms;  // Total time from arrival to completion
    std::string text;  // Output text for correctness validation (native mode)
    std::vector<int64_t> tokens;  // Output tokens for correctness validation (CB mode)
};

struct PendingRequest {
    size_t request_id;
    ov::genai::RawSpeechInput audio;
    std::chrono::steady_clock::time_point arrival_time;
};

struct InFlightRequest {
    size_t request_id;
    std::chrono::steady_clock::time_point arrival_time;
    std::chrono::steady_clock::time_point submission_time;
    std::future<std::vector<int64_t>> result_future;
};

class DynamicBenchmark {
private:
    std::string model_path;
    std::string device;
    ov::genai::RawSpeechInput base_audio;
    ov::genai::WhisperGenerationConfig config;
    bool is_native_mode;
    ov::genai::Tokenizer tokenizer;
    
    std::queue<PendingRequest> request_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    
    std::queue<InFlightRequest> inflight_queue;
    std::mutex inflight_mutex;
    std::condition_variable inflight_cv;
    
    std::atomic<bool> stop_flag{false};
    std::atomic<bool> submitter_finished{false};
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
        
        std::cout << "Request generator started: " << request_rate << " req/s, duration: " 
                  << duration_seconds << "s, inter-arrival: " << inter_arrival_time_ms << "ms\n";
        
        auto next_request_time = start_time;
        
        while (true) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration<double>(now - start_time).count();
            
            if (elapsed >= duration_seconds) {
                std::cout << "Request generation completed. Total scheduled: " << scheduled_requests.load() << "\n";
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
    
    void request_processor() {
        std::cout << "Request processor started (" 
                  << (is_native_mode ? "NATIVE sequential mode" : "CONTINUOUS BATCHING mode") 
                  << ")\n";
        
        // Create pipeline based on mode
        std::unique_ptr<ov::genai::WhisperPipeline> native_pipeline;
        std::unique_ptr<ov::genai::WhisperPipelinePoc> cb_pipeline;
        
        if (is_native_mode) {
            native_pipeline = std::make_unique<ov::genai::WhisperPipeline>(model_path + "-stateful", device);
            std::cout << "Warming up native pipeline...\n";
            native_pipeline->generate(base_audio, config);
            std::cout << "Warmup completed\n";
            
            // Native mode: sequential processing
            process_requests_sequential(native_pipeline.get());
            
        } else {
            cb_pipeline = std::make_unique<ov::genai::WhisperPipelinePoc>(model_path, device);
            std::cout << "Warming up continuous batching pipeline...\n";
            cb_pipeline->experimental_generate_with_continuous_batching(base_audio, config);
            
            // Start async continuous batching mode
            std::cout << "Starting async continuous batching...\n";
            cb_pipeline->start_continuous_batching(32);
            std::cout << "Warmup completed\n";
            
            // Continuous batching mode: async submission + result collection
            // Both run in separate threads for true async operation
            std::thread submitter(&DynamicBenchmark::submit_requests_async, this, cb_pipeline.get());
            std::thread result_collector(&DynamicBenchmark::collect_results, this);
            
            submitter.join();
            result_collector.join();
            
            // Cleanup
            std::cout << "Stopping continuous batching...\n";
            cb_pipeline->stop_continuous_batching();
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
                    std::cout << "Sequential processor exiting, no more requests\n";
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
                
                {
                    std::lock_guard<std::mutex> lock(metrics_mutex);
                    metrics.push_back(metrics_entry);
                }
                
                completed_requests++;
                
                if (completed_requests % 10 == 0) {
                    std::cout << "Completed: " << completed_requests.load() 
                              << " / " << scheduled_requests.load() 
                              << " (latency: " << std::fixed << std::setprecision(1) 
                              << metrics_entry.total_latency_ms << " ms)\n";
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Error processing request " << req.request_id << ": " << e.what() << "\n";
            }
        }
    }
    
    // Continuous batching mode: submit requests without blocking
    void submit_requests_async(ov::genai::WhisperPipelinePoc* pipeline) {
        std::cout << "Async request submitter started\n";
        
        while (true) {
            PendingRequest req;
            
            // Get next request from queue
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [this] { 
                    return !request_queue.empty() || stop_flag.load(); 
                });
                
                if (request_queue.empty() && stop_flag.load()) {
                    std::cout << "Request submitter exiting, no more requests\n";
                    break;
                }
                
                if (request_queue.empty()) {
                    continue;
                }
                
                req = request_queue.front();
                request_queue.pop();
            }
            
            // Submit request (NON-BLOCKING)
            auto submission_time = std::chrono::steady_clock::now();
            
            std::cout << "[Submitter] Request " << req.request_id << " - starting generate_async()\n";
            
            try {
                auto future = pipeline->generate_async(req.audio, config);
                
                auto after_submission = std::chrono::steady_clock::now();
                auto submission_duration_ms = std::chrono::duration<double, std::milli>(
                    after_submission - submission_time).count();
                
                std::cout << "[Submitter] Request " << req.request_id 
                          << " - generate_async() returned (took " << std::fixed << std::setprecision(1)
                          << submission_duration_ms << " ms)\n";
                
                // Add to in-flight queue for result collection
                InFlightRequest inflight;
                inflight.request_id = req.request_id;
                inflight.arrival_time = req.arrival_time;
                inflight.submission_time = submission_time;
                inflight.result_future = std::move(future);
                
                {
                    std::lock_guard<std::mutex> lock(inflight_mutex);
                    inflight_queue.push(std::move(inflight));
                    std::cout << "[Submitter] Request " << req.request_id 
                              << " - added to inflight queue (queue size: " << inflight_queue.size() << ")\n";
                }
                inflight_cv.notify_one();
                
            } catch (const std::exception& e) {
                std::cerr << "Error submitting request " << req.request_id << ": " << e.what() << "\n";
            }
        }
        
        // Signal that submitter has finished
        std::cout << "Request submitter finished all submissions\n";
        submitter_finished = true;
        inflight_cv.notify_all();
    }
    
    // Continuous batching mode: collect results from futures
    void collect_results() {
        std::cout << "Result collector started\n";
        size_t expected_requests = 0;
        
        while (true) {
            InFlightRequest inflight;
            
            // Get next in-flight request
            {
                std::unique_lock<std::mutex> lock(inflight_mutex);
                inflight_cv.wait(lock, [this] { 
                    return !inflight_queue.empty() || submitter_finished.load(); 
                });
                
                // Check if we're done: submitter finished AND no more in-flight requests
                if (inflight_queue.empty() && submitter_finished.load()) {
                    std::cout << "Result collector exiting: all requests completed\n";
                    break;
                }
                
                if (inflight_queue.empty()) {
                    continue;
                }
                
                inflight = std::move(inflight_queue.front());
                inflight_queue.pop();
            }
            
            std::cout << "[Collector] Request " << inflight.request_id << " - waiting for result...\n";
            
            // Wait for result (THIS is where blocking happens, but queue can fill meanwhile)
            try {
                auto tokens = inflight.result_future.get();
                
                auto completion_time = std::chrono::steady_clock::now();
                
                // Record metrics
                RequestMetrics metrics_entry;
                metrics_entry.request_id = inflight.request_id;
                metrics_entry.arrival_time = inflight.arrival_time;
                metrics_entry.start_time = inflight.submission_time;  // Approximation
                metrics_entry.completion_time = completion_time;
                
                metrics_entry.queue_time_ms = std::chrono::duration<double, std::milli>(
                    inflight.submission_time - inflight.arrival_time).count();
                metrics_entry.processing_time_ms = std::chrono::duration<double, std::milli>(
                    completion_time - inflight.submission_time).count();
                metrics_entry.total_latency_ms = std::chrono::duration<double, std::milli>(
                    completion_time - inflight.arrival_time).count();
                
                // Store tokens for validation in continuous batching mode
                metrics_entry.tokens = tokens;
                
                {
                    std::lock_guard<std::mutex> lock(metrics_mutex);
                    metrics.push_back(metrics_entry);
                }
                
                completed_requests++;
                
                if (completed_requests % 10 == 0) {
                    std::cout << "Completed: " << completed_requests.load() 
                              << " / " << scheduled_requests.load() 
                              << " (latency: " << std::fixed << std::setprecision(1) 
                              << metrics_entry.total_latency_ms << " ms)\n";
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Error collecting result for request " << inflight.request_id 
                          << ": " << e.what() << "\n";
            }
        }
    }
    
    void run(double request_rate, double duration_seconds) {
        auto start_time = std::chrono::steady_clock::now();
        
        std::cout << "\n========================================\n";
        std::cout << "Dynamic Batching Benchmark\n";
        std::cout << "Mode: " << (is_native_mode ? "NATIVE (sequential)" : "CONTINUOUS BATCHING") << "\n";
        std::cout << "========================================\n";
        std::cout << "Request rate: " << request_rate << " req/s\n";
        std::cout << "Duration: " << duration_seconds << " s\n";
        std::cout << "Expected requests: " << static_cast<int>(request_rate * duration_seconds) << "\n";
        std::cout << "========================================\n\n";
        
        // Start processor thread
        std::thread processor_thread(&DynamicBenchmark::request_processor, this);
        
        // Run generator in main thread
        request_generator(request_rate, duration_seconds);
        
        // Wait for all requests to complete
        std::cout << "\nWaiting for pending requests to complete...\n";
        processor_thread.join();
        
        auto end_time = std::chrono::steady_clock::now();
        auto total_duration = std::chrono::duration<double>(end_time - start_time).count();
        
        // Print summary
        print_summary(total_duration);
    }
    
    void print_summary(double total_duration_seconds) {
        std::lock_guard<std::mutex> lock(metrics_mutex);
        
        if (metrics.empty()) {
            std::cout << "\nNo requests completed!\n";
            return;
        }
        
        // Calculate statistics
        std::vector<double> latencies;
        std::vector<double> queue_times;
        std::vector<double> processing_times;
        
        for (const auto& m : metrics) {
            latencies.push_back(m.total_latency_ms);
            queue_times.push_back(m.queue_time_ms);
            processing_times.push_back(m.processing_time_ms);
        }
        
        std::sort(latencies.begin(), latencies.end());
        std::sort(queue_times.begin(), queue_times.end());
        std::sort(processing_times.begin(), processing_times.end());
        
        auto percentile = [](const std::vector<double>& sorted_data, double p) {
            size_t idx = static_cast<size_t>(sorted_data.size() * p);
            if (idx >= sorted_data.size()) idx = sorted_data.size() - 1;
            return sorted_data[idx];
        };
        
        double avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double avg_queue = std::accumulate(queue_times.begin(), queue_times.end(), 0.0) / queue_times.size();
        double avg_processing = std::accumulate(processing_times.begin(), processing_times.end(), 0.0) / processing_times.size();
        
        double throughput = metrics.size() / total_duration_seconds;
        
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
        
        // Correctness validation
        std::cout << "\n--- Correctness Validation (exact match) ---\n";
        if (!metrics.empty()) {
            size_t correct_count = 0;
            size_t mismatches = 0;
            
            // Check if we're in native mode (text) or CB mode (tokens)
            bool is_text_mode = !metrics[0].text.empty();
            bool is_token_mode = !metrics[0].tokens.empty();
            
            if (is_text_mode) {
                // Native mode: compare text
                const auto& reference_text = metrics[0].text;
                std::cout << "Mode: Native (text comparison)\n";
                std::cout << "Reference text (request 0): \"" << reference_text << "\"\n\n";
                
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
                }
                
                double correctness_rate = 100.0 * correct_count / metrics.size();
                std::cout << "\nCorrect outputs: " << correct_count << " / " << metrics.size() 
                          << " (" << correctness_rate << "%)\n";
                
                if (correctness_rate == 100.0) {
                    std::cout << "✓ All transcriptions identical\n";
                } else {
                    std::cout << "✗ Correctness check FAILED: " << mismatches << " mismatches detected\n";
                }
            } else if (is_token_mode) {
                // Continuous batching mode: compare token sequences
                const auto& reference_tokens = metrics[0].tokens;
                std::string reference_text = tokenizer.decode(reference_tokens, {ov::genai::skip_special_tokens(false)});
                
                std::cout << "Mode: Continuous Batching (token comparison)\n";
                std::cout << "Reference (request 0): " << reference_tokens.size()  << " tokens\n";
                std::cout << "  Tokens: [";
                for (size_t i = 0; i < std::min(reference_tokens.size(), size_t(20)); ++i) {
                    std::cout << reference_tokens[i];
                    if (i < std::min(reference_tokens.size(), size_t(20)) - 1) std::cout << ", ";
                }
                if (reference_tokens.size() > 20) std::cout << " ... (" << reference_tokens.size() - 20 << " more)";
                std::cout << "]\n";
                std::cout << "  Decoded text: \"" << reference_text << "\"\n\n";
                
                for (const auto& m : metrics) {
                    if (m.tokens == reference_tokens) {
                        correct_count++;
                    } else {
                        mismatches++;
                        if (mismatches <= 5) {
                            std::string decoded_text = tokenizer.decode(m.tokens, {ov::genai::skip_special_tokens(false)});
                            std::cout << "  Mismatch in request " << m.request_id 
                                      << ": expected " << reference_tokens.size() 
                                      << " tokens, got " << m.tokens.size() << " tokens\n";
                            std::cout << "    Tokens: [";
                            for (size_t i = 0; i < std::min(m.tokens.size(), size_t(20)); ++i) {
                                std::cout << m.tokens[i];
                                if (i < std::min(m.tokens.size(), size_t(20)) - 1) std::cout << ", ";
                            }
                            std::cout << "]\n";
                            std::cout << "    Decoded: \"" << decoded_text << "\"\n";
                        }
                    }
                }
                
                double correctness_rate = 100.0 * correct_count / metrics.size();
                std::cout << "\nCorrect outputs: " << correct_count << " / " << metrics.size() 
                          << " (" << correctness_rate << "%)\n";
                
                if (correctness_rate == 100.0) {
                    std::cout << "✓ All token sequences identical\n";
                } else {
                    std::cout << "✗ Correctness check FAILED: " << mismatches << " mismatches detected\n";
                }
            } else {
                std::cout << "No validation data available\n";
            }
        } else {
            std::cout << "No requests completed for validation\n";
        }
        
        std::cout << "========================================\n";
    }
};

int main(int argc, char* argv[]) try {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <audio_file> <request_rate> <duration_seconds> [device] [--is-native]\n";
        std::cerr << "  model_path: Path to the Whisper model directory\n";
        std::cerr << "  audio_file: Path to the WAV audio file\n";
        std::cerr << "  request_rate: Number of requests per second (e.g., 2.0)\n";
        std::cerr << "  duration_seconds: Benchmark duration in seconds (e.g., 60)\n";
        std::cerr << "  device: Optional device (default: CPU)\n";
        std::cerr << "  --is-native: Use native WhisperPipeline (sequential) instead of continuous batching\n";
        std::cerr << "\nExamples:\n";
        std::cerr << "  Continuous batching: " << argv[0] << " ./whisper-base-openvino ./sample.wav 2.0 30\n";
        std::cerr << "  Native sequential:   " << argv[0] << " ./whisper-base-openvino ./sample.wav 2.0 30 CPU --is-native\n";
        return EXIT_FAILURE;
    }
    
    std::filesystem::path models_path = argv[1];
    std::string wav_file_path = argv[2];
    double request_rate = std::stod(argv[3]);
    double duration_seconds = std::stod(argv[4]);
    
    // Parse optional arguments
    std::string device = "CPU";
    bool is_native_mode = false;
    
    for (int i = 5; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--is-native") {
            is_native_mode = true;
        } else if (i == 5) {  // First optional arg is device if not a flag
            device = arg;
        }
    }
    
    if (request_rate <= 0 || duration_seconds <= 0) {
        std::cerr << "Error: request_rate and duration must be positive\n";
        return EXIT_FAILURE;
    }
    
    std::cout << "Loading audio file: " << wav_file_path << "\n";
    ov::genai::RawSpeechInput raw_speech = utils::audio::read_wav(wav_file_path);
    std::cout << "Audio loaded: " << raw_speech.size() << " samples\n";
    
    // Setup generation config
    ov::genai::WhisperPipeline pipeline(models_path.parent_path() / (models_path.filename().string() + "-stateful"), device);

    ov::genai::WhisperGenerationConfig config = pipeline.get_generation_config();
    config.language = "<|en|>";  // Use language code, not token format
    config.task = "transcribe";
    config.return_timestamps = false;
    config.word_timestamps = false;
    
    // Run benchmark
    DynamicBenchmark benchmark(models_path, device, raw_speech, config, is_native_mode);
    benchmark.run(request_rate, duration_seconds);
    
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
