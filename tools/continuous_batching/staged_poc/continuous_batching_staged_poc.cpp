// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// POC tool: runs the same workload on the regular continuous batching engine and on the staged engine,
// then compares generated token ids and timings.

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <cxxopts.hpp>

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "openvino/genai/tokenizer.hpp"

namespace {

using Clock = std::chrono::steady_clock;

const std::vector<std::string> BASE_PROMPTS = {
    "Explain how a refrigerator keeps food cold.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "What are the main differences between TCP and UDP?",
    "Describe the water cycle to a ten year old.",
    "Give me a recipe for a simple vegetable soup and explain each step.",
    "Why is the sky blue during the day and red at sunset?",
    "Summarize the plot of a typical detective novel in a few sentences.",
    "List some practical tips for learning a new programming language quickly.",
};

struct Request {
    std::string prompt;
    ov::genai::GenerationConfig config;
    // offset from the start of the run
    double arrival_s = 0.0;
};

struct RunResult {
    std::vector<std::vector<int64_t>> tokens;
    double wall_s = 0.0;
    // time the engine had no requests and waited for the next arrival
    double idle_s = 0.0;
    double forward_s = 0.0;
    size_t num_steps = 0;
    std::vector<double> ttft_ms;
    std::vector<double> inter_token_ms;

    size_t num_tokens() const {
        size_t total = 0;
        for (const auto& request_tokens : tokens) {
            total += request_tokens.size();
        }
        return total;
    }

    double active_s() const {
        return wall_s - idle_s;
    }
};

double to_ms(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

double mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    return sum / values.size();
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    return values[static_cast<size_t>(fraction * (values.size() - 1))];
}

std::vector<Request> make_requests(size_t num_requests,
                                   const ov::genai::GenerationConfig& base_config,
                                   size_t prompt_repeat,
                                   bool varied_lengths,
                                   double request_rate) {
    // fixed seed: every engine gets the same prompts, lengths and arrival times
    std::mt19937 rng(42);
    std::vector<Request> requests(num_requests);
    double arrival_s = 0.0;
    for (size_t request_id = 0; request_id < num_requests; ++request_id) {
        Request& request = requests[request_id];
        const size_t repeat =
            varied_lengths ? std::uniform_int_distribution<size_t>(1, prompt_repeat)(rng) : prompt_repeat;
        // the prefix makes every prompt unique, so mixed up requests show as a mismatch
        request.prompt = "Request " + std::to_string(request_id) + ".";
        for (size_t i = 0; i < repeat; ++i) {
            request.prompt += " " + BASE_PROMPTS[(request_id + i) % BASE_PROMPTS.size()];
        }

        request.config = base_config;
        request.config.rng_seed = request_id;
        if (varied_lengths) {
            const size_t max_new_tokens = base_config.max_new_tokens;
            request.config.max_new_tokens =
                std::uniform_int_distribution<size_t>(std::max<size_t>(max_new_tokens / 4, 1), max_new_tokens)(rng);
        }

        if (request_rate > 0.0 && request_id > 0) {
            arrival_s += std::exponential_distribution<double>(request_rate)(rng);
        }
        request.arrival_s = arrival_s;
    }
    return requests;
}

// With use_arrival_times a second thread submits each request at its arrival time while this thread steps the
// pipeline, otherwise every request is submitted before the first step.
RunResult run_workload(ov::genai::ContinuousBatchingPipeline& pipe,
                       const std::vector<Request>& requests,
                       bool use_arrival_times) {
    const size_t num_requests = requests.size();
    RunResult result;
    result.tokens.resize(num_requests);
    std::vector<ov::genai::GenerationHandle> handles(num_requests);
    std::vector<Clock::time_point> last_token_time(num_requests);

    std::mutex mutex;
    std::condition_variable cv;
    // Guarded by mutex. Requests are submitted in id order, entries below num_submitted belong to this thread.
    size_t num_submitted = 0;
    bool submitter_finished = false;
    bool stop_submitter = false;
    std::exception_ptr submitter_error;

    const auto start = Clock::now();
    const auto submit = [&](size_t request_id) {
        const auto submit_time = Clock::now();
        auto handle = pipe.add_request(request_id, requests[request_id].prompt, requests[request_id].config);
        {
            std::lock_guard<std::mutex> lock(mutex);
            handles[request_id] = std::move(handle);
            last_token_time[request_id] = submit_time;
            ++num_submitted;
        }
        cv.notify_all();
    };

    std::thread submitter;
    // stops and joins the submitter on every exit path, including a failing step()
    struct SubmitterJoiner {
        std::thread& thread;
        std::mutex& mutex;
        std::condition_variable& cv;
        bool& stop;

        ~SubmitterJoiner() {
            if (!thread.joinable()) {
                return;
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                stop = true;
            }
            cv.notify_all();
            thread.join();
        }
    } submitter_joiner{submitter, mutex, cv, stop_submitter};

    if (use_arrival_times) {
        submitter = std::thread([&] {
            try {
                for (size_t request_id = 0; request_id < num_requests; ++request_id) {
                    const auto arrival = start + std::chrono::duration_cast<Clock::duration>(
                                                     std::chrono::duration<double>(requests[request_id].arrival_s));
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        if (cv.wait_until(lock, arrival, [&] { return stop_submitter; })) {
                            break;
                        }
                    }
                    submit(request_id);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex);
                submitter_error = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                submitter_finished = true;
            }
            cv.notify_all();
        });
    } else {
        for (size_t request_id = 0; request_id < num_requests; ++request_id) {
            submit(request_id);
        }
        submitter_finished = true;
    }

    const auto drain = [&] {
        size_t num_readable = 0;
        {
            std::lock_guard<std::mutex> lock(mutex);
            num_readable = num_submitted;
        }
        for (size_t request_id = 0; request_id < num_readable; ++request_id) {
            while (handles[request_id]->can_read()) {
                const auto outputs = handles[request_id]->read();
                const auto now = Clock::now();
                for (const auto& [sequence_id, output] : outputs) {
                    if (output.generated_ids.empty()) {
                        continue;
                    }
                    auto& request_tokens = result.tokens[request_id];
                    const double elapsed_ms = to_ms(now - last_token_time[request_id]);
                    if (request_tokens.empty()) {
                        result.ttft_ms.push_back(elapsed_ms);
                    } else {
                        const size_t num_new_tokens = output.generated_ids.size();
                        result.inter_token_ms.insert(result.inter_token_ms.end(),
                                                     num_new_tokens,
                                                     elapsed_ms / num_new_tokens);
                    }
                    request_tokens.insert(request_tokens.end(),
                                          output.generated_ids.begin(),
                                          output.generated_ids.end());
                    last_token_time[request_id] = now;
                }
            }
        }
    };

    while (true) {
        // Taken before asking the pipeline: a request counted here is already visible to it.
        size_t submitted_snapshot = 0;
        bool finished_snapshot = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            submitted_snapshot = num_submitted;
            finished_snapshot = submitter_finished;
        }
        if (pipe.has_non_finished_requests()) {
            pipe.step();
            ++result.num_steps;
            result.forward_s += pipe.get_metrics().inference_duration / 1e6;
            drain();
            continue;
        }
        if (finished_snapshot) {
            break;
        }
        const auto idle_start = Clock::now();
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] {
                return submitter_finished || num_submitted != submitted_snapshot;
            });
        }
        result.idle_s += to_ms(Clock::now() - idle_start) / 1e3;
    }
    drain();
    result.wall_s = to_ms(Clock::now() - start) / 1e3;

    if (submitter.joinable()) {
        submitter.join();
    }
    if (submitter_error) {
        std::rethrow_exception(submitter_error);
    }
    return result;
}

size_t count_matching_requests(const RunResult& reference, const RunResult& result, const std::string& engine) {
    size_t num_matching = 0;
    for (size_t request_id = 0; request_id < reference.tokens.size(); ++request_id) {
        const auto& expected = reference.tokens[request_id];
        const auto& actual = result.tokens[request_id];
        if (expected == actual) {
            ++num_matching;
            continue;
        }
        const size_t common_len = std::min(expected.size(), actual.size());
        size_t position = 0;
        while (position < common_len && expected[position] == actual[position]) {
            ++position;
        }
        std::cout << "  [" << engine << "] request " << request_id << " differs from baseline at token " << position
                  << " (baseline length " << expected.size() << ", " << engine << " length " << actual.size() << ")"
                  << std::endl;
    }
    return num_matching;
}

std::vector<std::string> split(const std::string& value, char delimiter) {
    std::vector<std::string> parts;
    std::stringstream stream(value);
    std::string part;
    while (std::getline(stream, part, delimiter)) {
        if (!part.empty()) {
            parts.push_back(part);
        }
    }
    return parts;
}

}  // namespace

int main(int argc, char* argv[]) try {
    cxxopts::Options options("continuous_batching_staged_poc", "Compares the staged continuous batching engine with the regular one");

    options.add_options()
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>())
    ("device", "Target device", cxxopts::value<std::string>()->default_value("CPU"))
    ("n,num_requests", "Number of requests", cxxopts::value<size_t>()->default_value("8"))
    ("request_rate", "Requests per second with Poisson arrivals. 0 submits every request before the first step", cxxopts::value<double>()->default_value("0"))
    ("prompt_repeat", "Number of base prompts concatenated into each prompt", cxxopts::value<size_t>()->default_value("1"))
    ("varied_lengths", "Draw each prompt length from [1, prompt_repeat] base prompts and each output length from [max_new_tokens / 4, max_new_tokens]", cxxopts::value<bool>()->default_value("false"))
    ("max_new_tokens", "Number of tokens to generate per request", cxxopts::value<size_t>()->default_value("128"))
    ("ignore_eos", "Generate max_new_tokens for every request", cxxopts::value<bool>()->default_value("true"))
    ("temperature", "Sampling temperature. 0 keeps greedy decoding unless top_k is set", cxxopts::value<float>()->default_value("0"))
    ("top_k", "Sampling top_k. 0 keeps greedy decoding unless temperature is set", cxxopts::value<size_t>()->default_value("0"))
    ("engines", "Comma separated engines to run: baseline, serial, split", cxxopts::value<std::string>()->default_value("baseline,serial,split"))
    ("b,max_batch_size", "A maximum number of batched tokens", cxxopts::value<size_t>()->default_value("256"))
    ("cache_size", "Size of memory used for KV cache in GB", cxxopts::value<size_t>()->default_value("2"))
    ("warmup", "Run the workload once, with every request submitted at the start, before the measured run", cxxopts::value<bool>()->default_value("true"))
    ("h,help", "Print usage");

    cxxopts::ParseResult args;
    try {
        args = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
        std::cout << e.what() << "\n\n" << options.help() << std::endl;
        return EXIT_FAILURE;
    }
    if (args.count("help") || !args.count("model")) {
        std::cout << options.help() << std::endl;
        return args.count("help") ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    const std::string models_path = args["model"].as<std::string>();
    const std::string device = args["device"].as<std::string>();
    const size_t num_requests = args["num_requests"].as<size_t>();
    const std::vector<std::string> engines = split(args["engines"].as<std::string>(), ',');
    const bool warmup = args["warmup"].as<bool>();
    const double request_rate = args["request_rate"].as<double>();
    const bool is_dynamic = request_rate > 0.0;
    const size_t prompt_repeat = std::max<size_t>(args["prompt_repeat"].as<size_t>(), 1);
    const float temperature = args["temperature"].as<float>();
    const size_t top_k = args["top_k"].as<size_t>();

    ov::genai::GenerationConfig config;
    config.max_new_tokens = args["max_new_tokens"].as<size_t>();
    config.ignore_eos = args["ignore_eos"].as<bool>();
    if (temperature > 0.0f || top_k > 0) {
        config.do_sample = true;
        if (temperature > 0.0f) {
            config.temperature = temperature;
        }
        if (top_k > 0) {
            config.top_k = top_k;
        }
    }

    const std::vector<Request> requests =
        make_requests(num_requests, config, prompt_repeat, args["varied_lengths"].as<bool>(), request_rate);

    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = args["max_batch_size"].as<size_t>();
    scheduler_config.cache_size = args["cache_size"].as<size_t>();

    std::cout << "Workload: " << num_requests << " requests on " << device << ", ";
    if (is_dynamic) {
        std::cout << request_rate << " requests/s with the last arrival at " << requests.back().arrival_s << " s";
    } else {
        std::cout << "all submitted before the first step";
    }
    if (config.do_sample) {
        std::cout << ", sampling with temperature " << config.temperature << " and top_k ";
        if (top_k > 0) {
            std::cout << top_k;
        } else {
            std::cout << "off";
        }
    } else {
        std::cout << ", greedy decoding";
    }
    std::cout << std::endl;

    std::vector<RunResult> results;
    bool workload_sizes_printed = false;
    for (const std::string& engine : engines) {
        ov::AnyMap properties;
        if (engine != "baseline") {
            properties["staged_engine"] = engine;
        }
        ov::genai::ContinuousBatchingPipeline pipe(models_path, scheduler_config, device, properties);
        if (!workload_sizes_printed) {
            ov::genai::Tokenizer tokenizer = pipe.get_tokenizer();
            size_t total_prompt_tokens = 0, min_prompt_tokens = SIZE_MAX, max_prompt_tokens = 0;
            size_t min_new_tokens = SIZE_MAX, max_new_tokens = 0;
            for (const Request& request : requests) {
                const size_t prompt_tokens = tokenizer.encode(request.prompt).input_ids.get_size();
                total_prompt_tokens += prompt_tokens;
                min_prompt_tokens = std::min(min_prompt_tokens, prompt_tokens);
                max_prompt_tokens = std::max(max_prompt_tokens, prompt_tokens);
                min_new_tokens = std::min(min_new_tokens, request.config.max_new_tokens);
                max_new_tokens = std::max(max_new_tokens, request.config.max_new_tokens);
            }
            std::cout << "Prompt tokens per request: " << min_prompt_tokens << " to " << max_prompt_tokens << " ("
                      << total_prompt_tokens << " in total), new tokens per request: " << min_new_tokens << " to "
                      << max_new_tokens << std::endl;
            workload_sizes_printed = true;
        }
        std::cout << "Running engine '" << engine << "'" << std::endl;
        if (warmup) {
            run_workload(pipe, requests, false);
        }
        results.push_back(run_workload(pipe, requests, is_dynamic));
    }

    const auto baseline_it = std::find(engines.begin(), engines.end(), "baseline");
    // With arrivals the batches depend on timing, so token ids are not expected to match.
    const RunResult* baseline =
        baseline_it == engines.end() || is_dynamic ? nullptr : &results[baseline_it - engines.begin()];

    bool serial_matches = true;
    std::vector<std::string> match_column(engines.size(), "-");
    if (baseline != nullptr) {
        for (size_t i = 0; i < engines.size(); ++i) {
            if (&results[i] == baseline) {
                continue;
            }
            const size_t num_matching = count_matching_requests(*baseline, results[i], engines[i]);
            match_column[i] = std::to_string(num_matching) + "/" + std::to_string(num_requests);
            if (engines[i] == "serial" && num_matching != num_requests) {
                serial_matches = false;
            }
        }
    }

    std::cout << std::endl
              << std::left << std::setw(10) << "engine" << std::right << std::setw(9) << "wall s" << std::setw(10)
              << "active s" << std::setw(9) << "tokens" << std::setw(10) << "tokens/s" << std::setw(8) << "steps"
              << std::setw(11) << "forward %" << std::setw(10) << "TTFT ms" << std::setw(10) << "TTFT p95"
              << std::setw(9) << "ITL ms" << std::setw(9) << "ITL p95" << std::setw(9) << "ITL p99" << std::setw(13)
              << "exact match" << std::endl;
    for (size_t i = 0; i < engines.size(); ++i) {
        const RunResult& result = results[i];
        std::cout << std::left << std::setw(10) << engines[i] << std::right << std::fixed << std::setprecision(2)
                  << std::setw(9) << result.wall_s << std::setw(10) << result.active_s() << std::setw(9)
                  << result.num_tokens() << std::setw(10) << result.num_tokens() / result.wall_s << std::setw(8)
                  << result.num_steps << std::setw(11) << 100.0 * result.forward_s / result.active_s()
                  << std::setw(10) << mean(result.ttft_ms) << std::setw(10) << percentile(result.ttft_ms, 0.95)
                  << std::setw(9) << mean(result.inter_token_ms) << std::setw(9)
                  << percentile(result.inter_token_ms, 0.95) << std::setw(9)
                  << percentile(result.inter_token_ms, 0.99) << std::setw(13) << match_column[i] << std::endl;
    }
    std::cout << std::endl
              << "active s: wall time minus the time the engine had no requests and waited for the next arrival."
              << std::endl
              << "tokens/s: generated tokens over wall time. Below capacity it follows the arrival rate." << std::endl
              << "forward %: share of active time spent in ModelRunner::forward (input packing + inference)."
              << std::endl
              << "TTFT: from add_request() to the first token read after a step. ITL: time between tokens of a request."
              << std::endl
              << "exact match: requests whose generated token ids are identical to the baseline engine. Not compared"
              << " with --request_rate, where batches depend on timing." << std::endl;

    if (!serial_matches) {
        std::cout << "FAILED: the serial staged engine must reproduce the baseline token ids" << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
