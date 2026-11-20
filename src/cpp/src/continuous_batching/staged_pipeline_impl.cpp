// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/staged_pipeline_impl.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

#include "continuous_batching/model_runner.hpp"
#include "continuous_batching/scheduler.hpp"
#include "sampling/sampler.hpp"

namespace ov::genai {

struct ContinuousBatchingPipeline::StagedImpl::Slot {
    std::shared_ptr<ModelRunner> runner;
    bool shares_primary_cache = false;
    ov::Shape bound_cache_shape;
    bool busy = false;

    std::vector<SequenceGroup::Ptr> groups;
    ContinuousBatchingScheduler::Output scheduler_output;
    ov::Tensor logits;
    std::vector<SequenceGroup::Ptr> sampled_groups;
    SamplerOutput sampler_output;
    float forward_us = 0.0f;
    float sampling_us = 0.0f;
    std::exception_ptr error;
};

ContinuousBatchingPipeline::StagedImpl::LaunchPolicy
ContinuousBatchingPipeline::StagedImpl::parse_launch_policy(const std::string& name) {
    if (name == "serial") {
        return LaunchPolicy::SERIAL;
    }
    if (name == "split") {
        return LaunchPolicy::SPLIT;
    }
    OPENVINO_THROW("Unknown staged engine launch policy '", name, "', expected 'serial' or 'split'");
}

ContinuousBatchingPipeline::StagedImpl::StagedImpl(const std::shared_ptr<ov::Model>& model,
                                                   const Tokenizer& tokenizer,
                                                   const SchedulerConfig& scheduler_config,
                                                   const std::string& device,
                                                   const ov::AnyMap& properties,
                                                   const ov::genai::GenerationConfig& generation_config,
                                                   LaunchPolicy launch_policy)
    : ContinuousBatchingImpl(model, tokenizer, scheduler_config, device, properties, generation_config),
      m_launch_policy(launch_policy) {
    // Growing the cache replaces its tensors, which must not happen under a group that is in flight.
    OPENVINO_ASSERT(scheduler_config.num_kv_blocks > 0 || scheduler_config.cache_size > 0,
                    "Staged engine requires a fixed size KV cache: set num_kv_blocks or cache_size");
    const SchedulerConfig& config = m_scheduler->get_config();
    OPENVINO_ASSERT(!config.use_cache_eviction && !config.use_sparse_attention && !config.enable_prefix_caching,
                    "Staged engine does not support cache eviction, sparse attention and prefix caching");
    OPENVINO_ASSERT(!m_scheduler->has_linear_attention_cache(),
                    "Staged engine does not support models with linear attention");
    OPENVINO_ASSERT(!m_adapter_controller, "Staged engine does not support LoRA adapters");

    ov::CompiledModel compiled_model = m_model_runner->get_infer_request().get_compiled_model();
    for (const auto& input : compiled_model.inputs()) {
        for (const auto& name : input.get_names()) {
            if (name.rfind("key_cache.", 0) == 0 || name.rfind("value_cache.", 0) == 0) {
                m_cache_input_names.push_back(name);
                break;
            }
        }
    }
    OPENVINO_ASSERT(!m_cache_input_names.empty(), "Staged engine found no KV cache inputs in the model");

    const size_t num_slots = m_launch_policy == LaunchPolicy::SERIAL ? 1 : 2;
    for (size_t slot_id = 0; slot_id < num_slots; ++slot_id) {
        auto slot = std::make_unique<Slot>();
        if (slot_id == 0) {
            slot->runner = m_model_runner;
        } else {
            slot->runner = std::make_shared<ModelRunner>(compiled_model.create_infer_request(),
                                                         m_scheduler->get_block_size(CacheType::KV_CACHE),
                                                         m_num_decoder_layers);
            slot->shares_primary_cache = true;
        }
        m_slots.push_back(std::move(slot));
    }

    m_forward_thread = std::thread(&StagedImpl::_forward_loop, this);
    m_post_thread = std::thread(&StagedImpl::_post_loop, this);
}

ContinuousBatchingPipeline::StagedImpl::~StagedImpl() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop = true;
    }
    m_cv.notify_all();
    m_forward_thread.join();
    m_post_thread.join();
}

void ContinuousBatchingPipeline::StagedImpl::_bind_shared_cache(Slot& slot) {
    ov::InferRequest primary = m_model_runner->get_infer_request();
    const ov::Shape cache_shape = primary.get_tensor(m_cache_input_names.front()).get_shape();
    if (cache_shape == slot.bound_cache_shape) {
        return;
    }
    OPENVINO_ASSERT(slot.bound_cache_shape.empty(), "KV cache was reallocated while the staged engine was running");
    ov::InferRequest request = slot.runner->get_infer_request();
    for (const std::string& name : m_cache_input_names) {
        request.set_tensor(name, primary.get_tensor(name));
    }
    slot.bound_cache_shape = cache_shape;
}

void ContinuousBatchingPipeline::StagedImpl::_forward_loop() {
    while (true) {
        Slot* slot = nullptr;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this] {
                return m_stop || !m_forward_queue.empty();
            });
            if (m_stop) {
                return;
            }
            slot = m_forward_queue.front();
            m_forward_queue.pop_front();
        }
        try {
            // This thread is the only one running inference, so the primary request is idle here.
            if (slot->shares_primary_cache) {
                _bind_shared_cache(*slot);
            }
            const auto start = std::chrono::steady_clock::now();
            slot->logits = slot->runner->forward(slot->groups, slot->scheduler_output);
            slot->forward_us = PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start);
        } catch (...) {
            slot->error = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            (slot->error ? m_done_queue : m_post_queue).push_back(slot);
        }
        m_cv.notify_all();
    }
}

void ContinuousBatchingPipeline::StagedImpl::_post_loop() {
    while (true) {
        Slot* slot = nullptr;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this] {
                return m_stop || !m_post_queue.empty();
            });
            if (m_stop) {
                return;
            }
            slot = m_post_queue.front();
            m_post_queue.pop_front();
        }
        try {
            // requires_sampling() changes once the sampler finishes the iteration
            for (const SequenceGroup::Ptr& group : slot->groups) {
                if (group->requires_sampling()) {
                    slot->sampled_groups.push_back(group);
                }
            }
            std::lock_guard<std::mutex> sampler_lock(m_sampler_mutex);
            const auto start = std::chrono::steady_clock::now();
            // handles are notified in _commit(), as ContinuousBatchingImpl does after sampling
            slot->sampler_output = m_sampler->sample(slot->groups, slot->logits, false, false);
            slot->sampling_us = PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start);
        } catch (...) {
            slot->error = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_done_queue.push_back(slot);
        }
        m_cv.notify_all();
    }
}

bool ContinuousBatchingPipeline::StagedImpl::_try_launch() {
    Slot* slot = nullptr;
    for (const auto& candidate : m_slots) {
        if (!candidate->busy) {
            slot = candidate.get();
            break;
        }
    }
    if (slot == nullptr) {
        return false;
    }

    std::vector<SequenceGroup::Ptr> ready;
    for (const SequenceGroup::Ptr& request : m_requests) {
        if (m_busy_requests.count(request.get()) == 0) {
            ready.push_back(request);
        }
    }
    if (ready.empty()) {
        return false;
    }
    // Starts the second group: the rest becomes ready for the other slot.
    if (m_launch_policy == LaunchPolicy::SPLIT && m_num_busy_slots == 0 && ready.size() > 1) {
        ready.resize((ready.size() + 1) / 2);
    }

    ContinuousBatchingScheduler::Output scheduler_output = m_scheduler->schedule(ready);

    if (scheduler_output.m_total_num_scheduled_tokens == 0) {
        // With a group in flight memory may still be released, otherwise the requests do not fit into the cache.
        if (m_num_busy_slots == 0) {
            for (const SequenceGroup::Ptr& group : ready) {
                if (!group->is_waiting()) {
                    _commit_perf_metrics(group);
                    group->set_out_of_memory();
                    group->notify_handle_oom();
                }
            }
            _free_non_running_ready_requests();
        }
        return false;
    }

    // Keep only scheduled requests in the group: the others stay ready and can join another group.
    std::vector<uint64_t>& scheduled_ids = scheduler_output.m_scheduled_sequence_groups_ids;
    std::vector<uint64_t> sorted_ids = scheduled_ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());
    std::vector<SequenceGroup::Ptr> groups;
    groups.reserve(sorted_ids.size());
    for (uint64_t id : sorted_ids) {
        groups.push_back(ready[id]);
    }
    for (uint64_t& id : scheduled_ids) {
        id = static_cast<uint64_t>(std::lower_bound(sorted_ids.begin(), sorted_ids.end(), id) - sorted_ids.begin());
    }

    for (const SequenceGroup::Ptr& group : groups) {
        m_busy_requests.insert(group.get());
    }
    ++m_num_busy_slots;
    slot->busy = true;
    slot->groups = std::move(groups);
    slot->scheduler_output = std::move(scheduler_output);
    slot->sampled_groups.clear();
    slot->sampler_output = SamplerOutput{};
    slot->error = nullptr;

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_forward_queue.push_back(slot);
    }
    m_cv.notify_all();
    return true;
}

void ContinuousBatchingPipeline::StagedImpl::_commit(Slot& slot) {
    for (const SequenceGroup::Ptr& group : slot.groups) {
        m_busy_requests.erase(group.get());
    }
    --m_num_busy_slots;
    slot.busy = false;
    if (slot.error) {
        std::rethrow_exception(std::exchange(slot.error, nullptr));
    }

    const SamplerOutput& sampler_output = slot.sampler_output;
    for (const auto& [parent_id, child_ids] : sampler_output.m_forked_sequences) {
        for (uint64_t child_id : child_ids) {
            m_scheduler->fork_sequence(parent_id, child_id);
        }
    }
    for (uint64_t seq_id : sampler_output.m_dropped_sequences) {
        m_scheduler->free_sequence(seq_id);
    }

    const auto commit_time = std::chrono::steady_clock::now();
    for (const SequenceGroup::Ptr& group : slot.groups) {
        const auto generated_it = sampler_output.num_generated_tokens_per_request.find(group->get_request_id());
        const size_t num_generated_tokens =
            generated_it == sampler_output.num_generated_tokens_per_request.end() ? 0 : generated_it->second;
        group->update_perf_metrics(MicroSeconds(slot.forward_us),
                                   MicroSeconds(slot.sampling_us),
                                   num_generated_tokens,
                                   commit_time);
    }
    for (const SequenceGroup::Ptr& group : slot.sampled_groups) {
        auto notification = group->prepare_notification();
        if (notification.publish && notification.status != GenerationStatus::RUNNING) {
            _commit_perf_metrics(group);
        }
        group->deliver_notification(std::move(notification));
    }

    m_batch_size += sampler_output.num_generated_tokens;
    m_pipeline_metrics.inference_duration += slot.forward_us;
    m_pipeline_metrics.sampling_duration += slot.sampling_us;
    slot.groups.clear();
    slot.sampled_groups.clear();
}

bool ContinuousBatchingPipeline::StagedImpl::_commit_done_slots() {
    bool committed = false;
    while (true) {
        Slot* slot = nullptr;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_done_queue.empty()) {
                break;
            }
            slot = m_done_queue.front();
            m_done_queue.pop_front();
        }
        _commit(*slot);
        committed = true;
    }
    return committed;
}

void ContinuousBatchingPipeline::StagedImpl::_free_non_running_ready_requests() {
    auto request_it = m_requests.begin();
    while (request_it != m_requests.end()) {
        const SequenceGroup::Ptr& request = *request_it;
        // A request in flight is released only after its group is committed.
        if (m_busy_requests.count(request.get()) != 0) {
            ++request_it;
            continue;
        }
        const bool is_finished = request->has_finished();
        if (!is_finished && !request->handle_stopped() && !request->handle_cancelled()) {
            ++request_it;
            continue;
        }
        if (!request->notified_terminal()) {
            _commit_perf_metrics(request);
            if (is_finished) {
                request->notify_handle_final();
            } else {
                request->notify_handle_stopped_or_cancelled();
            }
        }
        for (const auto& sequence : request->get_sequences()) {
            if (m_scheduler->has_block_table(sequence->get_id())) {
                m_scheduler->free_sequence(sequence->get_id());
            }
        }
        {
            std::lock_guard<std::mutex> sampler_lock(m_sampler_mutex);
            m_sampler->clear_request_info(request->get_request_id());
        }
        request_it = m_requests.erase(request_it);
    }
}

void ContinuousBatchingPipeline::StagedImpl::_wait_for_done_slot() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv.wait(lock, [this] {
        return !m_done_queue.empty();
    });
}

void ContinuousBatchingPipeline::StagedImpl::_abandon_slots() {
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [this] {
            return m_done_queue.size() == m_num_busy_slots;
        });
        m_done_queue.clear();
    }
    for (const auto& slot : m_slots) {
        slot->busy = false;
        slot->groups.clear();
        slot->sampled_groups.clear();
    }
    m_busy_requests.clear();
    m_num_busy_slots = 0;
}

void ContinuousBatchingPipeline::StagedImpl::drop_requests() {
    _abandon_slots();
    ContinuousBatchingImpl::drop_requests();
}

void ContinuousBatchingPipeline::StagedImpl::step() {
    {
        std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
        _throw_if_failed();
    }

    try {
        m_batch_size = 0;
        m_pipeline_metrics.inference_duration = 0.0f;
        m_pipeline_metrics.sampling_duration = 0.0f;

        _pull_awaiting_requests();
        const bool committed = _commit_done_slots();
        _free_non_running_ready_requests();
        while (_try_launch()) {
        }

        // Nothing to report yet: block until a group finishes, and relaunch before returning to the caller.
        if (!committed && m_num_busy_slots > 0) {
            _wait_for_done_slot();
            _commit_done_slots();
            _pull_awaiting_requests();
            _free_non_running_ready_requests();
            while (_try_launch()) {
            }
        }
    } catch (...) {
        const std::exception_ptr error = std::current_exception();
        _abandon_slots();
        _fail_pipeline(error);
        std::rethrow_exception(error);
    }
}

}  // namespace ov::genai
