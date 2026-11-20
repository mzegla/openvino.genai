// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "continuous_batching/pipeline_impl.hpp"

namespace ov::genai {

// POC: internal property selecting the staged engine, value is a launch policy name ("serial" or "split").
inline constexpr const char* STAGED_ENGINE_PROPERTY = "staged_engine";

// POC: step() schedules and commits groups of requests, a forward thread and a post thread run and sample them.
class ContinuousBatchingPipeline::StagedImpl : public ContinuousBatchingPipeline::ContinuousBatchingImpl {
public:
    enum class LaunchPolicy {
        // one group in flight holding every ready request, the batches match ContinuousBatchingImpl
        SERIAL,
        // two groups, each holding the requests that are ready when it is launched
        SPLIT
    };

    static LaunchPolicy parse_launch_policy(const std::string& name);

    StagedImpl(const std::shared_ptr<ov::Model>& model,
               const Tokenizer& tokenizer,
               const SchedulerConfig& scheduler_config,
               const std::string& device,
               const ov::AnyMap& properties,
               const ov::genai::GenerationConfig& generation_config,
               LaunchPolicy launch_policy);

    ~StagedImpl() override;

    void step() override;

protected:
    void drop_requests() override;

private:
    struct Slot;

    LaunchPolicy m_launch_policy;
    std::vector<std::unique_ptr<Slot>> m_slots;
    std::vector<std::string> m_cache_input_names;

    // owned by the thread calling step()
    std::unordered_set<const SequenceGroup*> m_busy_requests;
    size_t m_num_busy_slots = 0;

    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::deque<Slot*> m_forward_queue;
    std::deque<Slot*> m_post_queue;
    std::deque<Slot*> m_done_queue;
    bool m_stop = false;

    // Sampler keeps per-request state in a map that is not safe for concurrent access.
    std::mutex m_sampler_mutex;

    std::thread m_forward_thread;
    std::thread m_post_thread;

    void _forward_loop();
    void _post_loop();
    void _bind_shared_cache(Slot& slot);

    bool _try_launch();
    bool _commit_done_slots();
    void _commit(Slot& slot);
    void _free_non_running_ready_requests();
    void _wait_for_done_slot();
    void _abandon_slots();
};

}  // namespace ov::genai
