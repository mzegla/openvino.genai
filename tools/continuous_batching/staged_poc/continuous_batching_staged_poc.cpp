// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// POC tool: runs the same closed loop greedy workload on the regular continuous batching engine and on the
// staged engine, then compares generated token ids and timings.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cxxopts.hpp>

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_handle.hpp"

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

struct RunResult {
    std::vector<std::vector<int64_t>> tokens;
    double wall_s = 0.0;
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

RunResult run_workload(ov::genai::ContinuousBatchingPipeline& pipe,
                       const std::vector<std::string>& prompts,
                       const ov::genai::GenerationConfig& config) {
    RunResult result;
    result.tokens.resize(prompts.size());
    std::vector<ov::genai::GenerationHandle> handles;
    handles.reserve(prompts.size());

    const auto start = Clock::now();
    for (size_t request_id = 0; request_id < prompts.size(); ++request_id) {
        handles.push_back(pipe.add_request(request_id, prompts[request_id], config));
    }

    std::vector<Clock::time_point> last_token_time(prompts.size(), start);
    const auto drain = [&] {
        for (size_t request_id = 0; request_id < handles.size(); ++request_id) {
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

    while (pipe.has_non_finished_requests()) {
        pipe.step();
        ++result.num_steps;
        result.forward_s += pipe.get_metrics().inference_duration / 1e6;
        drain();
    }
    drain();
    result.wall_s = to_ms(Clock::now() - start) / 1e3;
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
    ("n,num_requests", "Number of requests, all submitted before the first step", cxxopts::value<size_t>()->default_value("8"))
    ("max_new_tokens", "Number of tokens to generate per request", cxxopts::value<size_t>()->default_value("128"))
    ("ignore_eos", "Generate max_new_tokens for every request", cxxopts::value<bool>()->default_value("true"))
    ("engines", "Comma separated engines to run: baseline, serial, split", cxxopts::value<std::string>()->default_value("baseline,serial,split"))
    ("b,max_batch_size", "A maximum number of batched tokens", cxxopts::value<size_t>()->default_value("256"))
    ("cache_size", "Size of memory used for KV cache in GB", cxxopts::value<size_t>()->default_value("2"))
    ("warmup", "Run the workload once before the measured run", cxxopts::value<bool>()->default_value("true"))
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

    std::vector<std::string> prompts;
    for (size_t request_id = 0; request_id < num_requests; ++request_id) {
        // the prefix makes every prompt unique, so mixed up requests show as a mismatch
        prompts.push_back("Request " + std::to_string(request_id) + ". " +
                          BASE_PROMPTS[request_id % BASE_PROMPTS.size()]);
    }

    ov::genai::GenerationConfig config;
    config.max_new_tokens = args["max_new_tokens"].as<size_t>();
    config.ignore_eos = args["ignore_eos"].as<bool>();

    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = args["max_batch_size"].as<size_t>();
    scheduler_config.cache_size = args["cache_size"].as<size_t>();

    std::vector<RunResult> results;
    for (const std::string& engine : engines) {
        ov::AnyMap properties;
        if (engine != "baseline") {
            properties["staged_engine"] = engine;
        }
        std::cout << "Running engine '" << engine << "' on " << device << " with " << num_requests << " requests"
                  << std::endl;
        ov::genai::ContinuousBatchingPipeline pipe(models_path, scheduler_config, device, properties);
        if (warmup) {
            run_workload(pipe, prompts, config);
        }
        results.push_back(run_workload(pipe, prompts, config));
    }

    const auto baseline_it = std::find(engines.begin(), engines.end(), "baseline");
    const RunResult* baseline = baseline_it == engines.end() ? nullptr : &results[baseline_it - engines.begin()];

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
              << std::left << std::setw(10) << "engine" << std::right << std::setw(10) << "wall s" << std::setw(10)
              << "tokens" << std::setw(12) << "tokens/s" << std::setw(9) << "steps" << std::setw(12) << "forward %"
              << std::setw(11) << "TTFT ms" << std::setw(11) << "ITL ms" << std::setw(12) << "ITL p95 ms"
              << std::setw(13) << "exact match" << std::endl;
    for (size_t i = 0; i < engines.size(); ++i) {
        const RunResult& result = results[i];
        std::cout << std::left << std::setw(10) << engines[i] << std::right << std::fixed << std::setprecision(2)
                  << std::setw(10) << result.wall_s << std::setw(10) << result.num_tokens() << std::setw(12)
                  << result.num_tokens() / result.wall_s << std::setw(9) << result.num_steps << std::setw(12)
                  << 100.0 * result.forward_s / result.wall_s << std::setw(11) << mean(result.ttft_ms)
                  << std::setw(11) << mean(result.inter_token_ms) << std::setw(12)
                  << percentile(result.inter_token_ms, 0.95) << std::setw(13) << match_column[i] << std::endl;
    }
    std::cout << std::endl
              << "forward %: share of wall time spent in ModelRunner::forward (input packing + inference)." << std::endl
              << "exact match: requests whose generated token ids are identical to the baseline engine." << std::endl;

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
