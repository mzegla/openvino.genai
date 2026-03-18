// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <immintrin.h>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"

// ---------------------------------------------------------------------------
// CrossKVCache — slot buffer + swap-to-end compaction for cross-attention K/V
//
// Layout:  [n_layers, 2, max_slots, n_heads, T_enc, head_dim]
//          ^^^^fixed^^^^^^ ^var^    ^^^fixed (model-determined)^^^^^^^^
//
// Slot management: a free-list of available slot indices.
// release() returns a slot to the free list (O(1), no data movement).
// admit() pops a free slot, then writes K/V data lock-free.
// gather_batch() looks up each request's slot via m_req_to_slot.
//
// Usage:
//   CrossKVCache cache(proj_infer_request, max_slots, n_layers, n_heads, T_enc, head_dim, dtype);
//   cache.admit(req_id, encoder_hidden_state);  // once per request
//   const auto& kv = cache.gather_batch(ordered_req_ids);  // every step
//   ... set_tensor("cross_kv_K_l", kv[l][0]) for each layer l ...
//   cache.release(req_id);  // on completion
// ---------------------------------------------------------------------------

class CrossKVCache {
public:
    // Construct with a compiled projector InferRequest.
    //   projector inputs:  "encoder_hidden_states" [1, T_enc, D]
    //   projector outputs: "cross_proj_K_l", "cross_proj_V_l" for l in [0, n_layers)
    //                      each shape [1, n_heads, T_enc, head_dim]
    CrossKVCache(ov::InferRequest proj_request,
                 size_t max_slots,
                 size_t n_layers,
                 size_t n_heads,
                 size_t T_enc,
                 size_t head_dim,
                 ov::element::Type elem_type)
        : m_proj_request(std::move(proj_request)),
          m_max_slots(max_slots),
          m_n_layers(n_layers),
          m_n_heads(n_heads),
          m_T_enc(T_enc),
          m_head_dim(head_dim),
          m_elem_type(elem_type),
          m_storage_type(_resolve_storage_type(elem_type))
    {
        // Initialise free-slot stack: all slots available.
        m_free_slots.reserve(max_slots);
        for (size_t i = max_slots; i-- > 0; )  // push in reverse so slot 0 is popped first
            m_free_slots.push_back(static_cast<uint32_t>(i));

        // Allocate contiguous buffer: [n_layers, 2, max_slots, n_heads, T_enc, head_dim]
        // Stored in m_storage_type (may be bf16 even if model runs in f32, to halve DRAM/L3 pressure).
        ov::Shape buf_shape{n_layers, 2, max_slots, n_heads, T_enc, head_dim};
        m_buffer = ov::Tensor(m_storage_type, buf_shape);
        std::memset(m_buffer.data(), 0, m_buffer.get_byte_size());

        const size_t total_mb = m_buffer.get_byte_size() / (1024 * 1024);
        std::cout << "[CrossKVCache] Allocated slot buffer: "
                  << total_mb << " MB  (" << n_layers << " layers × 2 × "
                  << max_slots << " slots × " << n_heads << " heads × "
                  << T_enc << " enc_len × " << head_dim << " head_dim, "
                  << m_storage_type.get_type_name() << " storage / "
                  << elem_type.get_type_name() << " compute)\n";

        // Pre-allocate gather workspace in m_elem_type (what the decoder parameters expect).
        m_batch_raw_K.resize(n_layers);
        m_batch_raw_V.resize(n_layers);
        for (size_t l = 0; l < n_layers; ++l) {
            m_batch_raw_K[l] = ov::Tensor(elem_type, {max_slots, n_heads, T_enc, head_dim});
            m_batch_raw_V[l] = ov::Tensor(elem_type, {max_slots, n_heads, T_enc, head_dim});
        }
        m_alias_kv.resize(n_layers);

        // N=1 output: if storage==compute type use zero-copy aliases into slot buffer;
        // otherwise use per-layer single-slot conversion buffers (bf16→f32 on gather).
        m_alias1_kv.resize(n_layers);
        if (m_storage_type != m_elem_type) {
            m_conv1_K.resize(n_layers);
            m_conv1_V.resize(n_layers);
            for (size_t l = 0; l < n_layers; ++l) {
                m_conv1_K[l] = ov::Tensor(elem_type, {1, n_heads, T_enc, head_dim});
                m_conv1_V[l] = ov::Tensor(elem_type, {1, n_heads, T_enc, head_dim});
            }
        }

        const size_t workspace_mb = 2 * n_layers * m_batch_raw_K[0].get_byte_size() / (1024 * 1024);
        std::cout << "[CrossKVCache] Pre-allocated gather workspace: "
                  << workspace_mb << " MB (" << n_layers << " × 2 buffers × max_slots="
                  << max_slots << ")\n";
    }

    // -------------------------------------------------------------------
    // admit(): run projector once for req_id, store K/V into a new slot.
    //
    // Thread safety: called from the request-generator thread.
    // If all slots are occupied, BLOCKS until release() frees one.
    // The projector infer() is serialized via m_proj_mutex so concurrent
    // calls don't corrupt the InferRequest's output buffers.
    // -------------------------------------------------------------------
    void admit(uint64_t req_id, const ov::Tensor& encoder_hidden_state) {
        // Step 1: run projector (slow, ~50 ms) while holding only proj_mutex.
        // This does NOT block the main pipeline thread from calling release().
        auto [proj_K, proj_V] = _run_projector(encoder_hidden_state);

        // Step 2: wait for a free slot (fast bookkeeping only, lock released before memcpy).
        std::unique_lock<std::mutex> lk(m_mutex);
        m_slot_cv.wait(lk, [&] { return !m_free_slots.empty(); });
        _store_from_tensors(req_id, proj_K, proj_V, lk);  // releases lk before memcpy
    }

    // -------------------------------------------------------------------
    // admit_precomputed(): store K/V from a projector InferRequest that
    // has already been run (outputs are valid). Used by lazy-init path to
    // avoid running infer() twice for the very first request.
    // BLOCKS until a slot is free — do NOT call from the main pipeline
    // step() thread (would deadlock with release() on the same thread).
    // -------------------------------------------------------------------
    void admit_precomputed(uint64_t req_id, ov::InferRequest& proj_req) {
        // Copy outputs from the already-completed request.
        auto [proj_K, proj_V] = _copy_proj_outputs(proj_req);

        std::unique_lock<std::mutex> lk(m_mutex);
        m_slot_cv.wait(lk, [&] { return !m_free_slots.empty(); });
        _store_from_tensors(req_id, proj_K, proj_V, lk);  // releases lk before memcpy
    }

    // -------------------------------------------------------------------
    // try_admit_precomputed_nowait(): non-blocking variant of admit_precomputed().
    // Returns true and stores K/V if a free slot is available right now.
    // Returns false immediately (no K/V stored) if all slots are occupied.
    //
    // Safe to call from the main pipeline step() thread — never blocks.
    // -------------------------------------------------------------------
    bool try_admit_precomputed_nowait(uint64_t req_id, ov::InferRequest& proj_req) {
        auto [proj_K, proj_V] = _copy_proj_outputs(proj_req);
        std::unique_lock<std::mutex> lk(m_mutex);
        if (m_free_slots.empty()) return false;
        _store_from_tensors(req_id, proj_K, proj_V, lk);  // releases lk before memcpy
        return true;
    }

    // -------------------------------------------------------------------
    // release(): return the slot for req_id to the free list.
    // No data movement — O(1) bookkeeping only.
    // Called from main pipeline thread. Notifies any blocked admit() calls.
    // -------------------------------------------------------------------
    void release(uint64_t req_id) {
        {
            std::lock_guard<std::mutex> lk(m_mutex);
            auto it = m_req_to_slot.find(req_id);
            if (it == m_req_to_slot.end()) return;  // never admitted (CROSS_KV_CACHE=0 path)

            const uint32_t freed_slot = it->second;
            m_req_to_slot.erase(it);
            m_free_slots.push_back(freed_slot);

            std::cout << "[CrossKVCache] Released req " << req_id
                      << " slot=" << freed_slot
                      << " (active=" << m_req_to_slot.size() << "/" << m_max_slots << ")\n";
        }  // unlock before notify so a woken admit() can acquire immediately
        m_slot_cv.notify_one();
    }

    // -------------------------------------------------------------------
    // gather_batch(): assemble per-layer K/V tensors [N, n_heads, T_enc, head_dim]
    // for the N requests in req_ids (ordered to match the scheduler batch).
    //
    // Returns a reference to the internal alias tensor cache.
    //
    // out_need_set_tensor: true when N changed (shape change requires OV re-bind).
    //
    // N=1 SPECIAL CASE (zero-copy, non-caching):
    //   Wraps the contiguous slot-buffer slice directly — no memcpy, no workspace.
    //   Critically, m_last_req_ids and m_last_N are NOT updated for N=1.
    //   This preserves the early-exit for the N>1 GENERATE pass that follows in
    //   the same step (three-way split: PROMPT/TRANSITION solo → GENERATE batch).
    //   Without this, every GENERATE step after a solo phase would fully re-gather
    //   ~61-600MB of K/V data at ~10 GB/s, costing 6-60ms per step.
    // -------------------------------------------------------------------
    const std::vector<std::array<ov::Tensor, 2>>&
    gather_batch(const std::vector<uint64_t>& req_ids, bool& out_need_set_tensor) {
        const size_t N = req_ids.size();
        OPENVINO_ASSERT(N > 0, "CrossKVCache::gather_batch: empty req_ids");

        // ---------------------------------------------------------------
        // N=1 ZERO-COPY PATH:
        //   Wrap the contiguous slot-buffer slice directly — no memcpy.
        //   Does NOT update m_last_req_ids / m_last_N so the N>1 early-exit
        //   and set_tensor-avoidance logic for the GENERATE phase is undisturbed.
        //   Cost: 1 mutex lock + n_layers*2 Tensor header constructions (~ns).
        // ---------------------------------------------------------------
        if (N == 1) {
            uint32_t slot;
            {
                std::lock_guard<std::mutex> lk(m_mutex);
                slot = m_req_to_slot.at(req_ids[0]);
            }
            const ov::Shape shape1{1, m_n_heads, m_T_enc, m_head_dim};
            if (m_storage_type == m_elem_type) {
                // True zero-copy: wrap slot buffer pointer directly.
                for (size_t l = 0; l < m_n_layers; ++l) {
                    m_alias1_kv[l][0] = ov::Tensor(m_elem_type, shape1, _slot_ptr(l, 0, slot));
                    m_alias1_kv[l][1] = ov::Tensor(m_elem_type, shape1, _slot_ptr(l, 1, slot));
                }
            } else {
                // Storage (bf16) differs from compute (f32): convert into dedicated N=1 buffers.
                const size_t n_elems = _slot_elem_count();
                for (size_t l = 0; l < m_n_layers; ++l) {
                    _gather_slot_to(_slot_ptr(l, 0, slot),
                                    static_cast<uint8_t*>(m_conv1_K[l].data()), n_elems);
                    _gather_slot_to(_slot_ptr(l, 1, slot),
                                    static_cast<uint8_t*>(m_conv1_V[l].data()), n_elems);
                    m_alias1_kv[l][0] = ov::Tensor(m_elem_type, shape1, m_conv1_K[l].data());
                    m_alias1_kv[l][1] = ov::Tensor(m_elem_type, shape1, m_conv1_V[l].data());
                }
            }
            // Need set_tensor only when transitioning from N>1 to N=1
            out_need_set_tensor = (m_last_set_tensor_N != 1);
            if (out_need_set_tensor) m_last_set_tensor_N = 1;
            return m_alias1_kv;
        }

        // ---------------------------------------------------------------
        // N>1 fast path: identical batch → nothing to do.
        // ---------------------------------------------------------------
        if (req_ids == m_last_req_ids) {
            out_need_set_tensor = false;
            return m_alias_kv;
        }

        // ---------------------------------------------------------------
        // N>1 batch changed: snapshot slot assignments under lock.
        // ---------------------------------------------------------------
        std::vector<uint32_t> cur_slots(N);
        {
            std::lock_guard<std::mutex> lk(m_mutex);
            for (size_t i = 0; i < N; ++i)
                cur_slots[i] = m_req_to_slot.at(req_ids[i]);
        }

        m_last_req_ids = req_ids;

        // set_tensor needed only on shape change (N differs from last set_tensor call)
        const bool shape_changed = (N != m_last_set_tensor_N);
        if (shape_changed) m_last_set_tensor_N = N;
        out_need_set_tensor = shape_changed;

        // ---------------------------------------------------------------
        // Gather into pre-allocated workspace.
        // Parallelized across (layer × K/V) tasks to saturate DRAM bandwidth.
        // Each of the n_layers*2 tasks is independent: reads from slot buffer,
        // writes to pre-allocated workspace. For n_layers=4: 8 concurrent 14-73 MB
        // copies → fill memory bus instead of single-thread ~10 GB/s limit.
        // Sequential fallback for N==1 is handled by the zero-copy path above.
        // ---------------------------------------------------------------
        // Launch one async task per (layer, kv) pair
        // Each task gathers N rows from the slot buffer into the workspace,
        // converting storage type → compute type if needed (e.g. bf16 → f32).
        const size_t n_elems_per_slot = _slot_elem_count();
        const size_t dst_slot_bytes   = n_elems_per_slot * m_elem_type.size();
        const size_t n_tasks    = m_n_layers * 2;  // layer × {K,V}

        // Launch one async task per (layer, kv) pair
        std::vector<std::future<void>> futures;
        futures.reserve(n_tasks);
        for (size_t l = 0; l < m_n_layers; ++l) {
            for (size_t kv = 0; kv < 2; ++kv) {
                uint8_t* dst = static_cast<uint8_t*>(
                    kv == 0 ? m_batch_raw_K[l].data() : m_batch_raw_V[l].data());
                futures.push_back(std::async(std::launch::async,
                    [this, dst, l, kv, N, dst_slot_bytes, n_elems_per_slot, &cur_slots]() {
                        for (size_t i = 0; i < N; ++i)
                            _gather_slot_to(_slot_ptr(l, kv, cur_slots[i]),
                                            dst + i * dst_slot_bytes,
                                            n_elems_per_slot);
                    }));
            }
        }
        for (auto& f : futures) f.get();

        // Rebuild alias Tensor wrappers only when shape changes.
        // When N is stable the existing wrappers already point to the right
        // pre-allocated buffers; the parallel memcpy above updated them in-place.
        if (shape_changed) {
            const ov::Shape shapeN{N, m_n_heads, m_T_enc, m_head_dim};
            for (size_t l = 0; l < m_n_layers; ++l) {
                m_alias_kv[l][0] = ov::Tensor(m_elem_type, shapeN,
                                              m_batch_raw_K[l].data());
                m_alias_kv[l][1] = ov::Tensor(m_elem_type, shapeN,
                                              m_batch_raw_V[l].data());
            }
        }
        return m_alias_kv;
    }

    // -------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------
    bool has_slot(uint64_t req_id) const {
        std::lock_guard<std::mutex> lk(m_mutex);
        return m_req_to_slot.count(req_id) > 0;
    }

    uint32_t slot_of(uint64_t req_id) const {
        std::lock_guard<std::mutex> lk(m_mutex);
        return m_req_to_slot.at(req_id);
    }

    size_t n_active()  const { std::lock_guard<std::mutex> lk(m_mutex); return m_req_to_slot.size(); }
    size_t n_layers()  const { return m_n_layers; }
    size_t max_slots() const { return m_max_slots; }
    /// Cumulative wall time of all projector infer() calls, in microseconds.
    double get_proj_infer_us() const {
        std::lock_guard<std::mutex> lk(m_proj_mutex);
        return m_proj_infer_us;
    }

private:
    ov::InferRequest m_proj_request;

    // Mutex serializing projector infer() calls (admit() is potentially concurrent)
    mutable std::mutex m_proj_mutex;
    // Cumulative projector infer() wall time in microseconds (written only under m_proj_mutex)
    double m_proj_infer_us = 0.0;
    // Mutex protecting slot bookkeeping state; also used by m_slot_cv
    mutable std::mutex m_mutex;
    // Notified by release() to wake up admit() calls blocked on a full buffer
    std::condition_variable m_slot_cv;

    // Contiguous slab — layout [n_layers, 2, max_slots, n_heads, T_enc, head_dim]
    ov::Tensor m_buffer;

    const size_t m_max_slots;
    const size_t m_n_layers;
    const size_t m_n_heads;
    const size_t m_T_enc;
    const size_t m_head_dim;
    const ov::element::Type m_elem_type;     // compute precision (what OV decoder expects)
    const ov::element::Type m_storage_type;  // slot buffer storage precision (may be bf16)

    // Slot bookkeeping — free list; slots are not necessarily contiguous.
    std::vector<uint32_t>              m_free_slots;    // stack of available slot indices
    std::unordered_map<uint64_t, uint32_t> m_req_to_slot;  // req_id → slot (active entries only)

    // Gather workspace: pre-allocated once at construction, sized [max_slots, n_heads, T_enc, head_dim]
    // per (layer, K/V).  gather_batch() converts and copies into these when req_ids change.
    std::vector<ov::Tensor> m_batch_raw_K;   // [n_layers] each [max_slots, n_heads, T_enc, head_dim]
    std::vector<ov::Tensor> m_batch_raw_V;   // [n_layers] each [max_slots, n_heads, T_enc, head_dim]

    // N>1 alias tensors (returned by gather_batch for N>1); shape rebuilt only on N-change.
    std::vector<std::array<ov::Tensor, 2>> m_alias_kv;
    // N=1 aliases: zero-copy if storage==compute, else backed by m_conv1_K/V.
    std::vector<std::array<ov::Tensor, 2>> m_alias1_kv;
    // Per-layer single-slot f32 conversion buffers for N=1 when storage_type != elem_type.
    std::vector<ov::Tensor> m_conv1_K;
    std::vector<ov::Tensor> m_conv1_V;

    // N>1 batch identity cache: early-exit if req_ids == m_last_req_ids.
    std::vector<uint64_t> m_last_req_ids;
    // Last N for which set_tensor was actually called; set_tensor skipped when N is stable.
    size_t m_last_set_tensor_N = 0;

    // -------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------

    // Number of scalar elements in one (layer, kv, slot) slice
    size_t _slot_elem_count() const { return m_n_heads * m_T_enc * m_head_dim; }

    // Byte pointer to slot `slot` in layer `layer`, K/V index `kv` (0=K, 1=V)
    // Uses m_storage_type.size() since the buffer is in storage precision.
    uint8_t* _slot_ptr(size_t layer, size_t kv, uint32_t slot) const {
        const size_t slot_elems  = m_n_heads * m_T_enc * m_head_dim;
        const size_t kv_elems    = m_max_slots * slot_elems;
        const size_t layer_elems = 2          * kv_elems;
        const size_t offset      = layer * layer_elems + kv * kv_elems + slot * slot_elems;
        return static_cast<uint8_t*>(const_cast<void*>(m_buffer.data())) + offset * m_storage_type.size();
    }

    // Resolve storage type: check CROSS_KV_PRECISION env var; default = same as compute type.
    static ov::element::Type _resolve_storage_type(ov::element::Type compute_type) {
        const char* env = std::getenv("CROSS_KV_PRECISION");
        if (env) {
            std::string s(env);
            if (s == "bf16") return ov::element::bf16;
            if (s == "f16")  return ov::element::f16;
            if (s == "f32")  return ov::element::f32;
        }
        return compute_type;
    }

    // Convert n elements from src (f32) → dst (bf16) via bit-shift truncation.
    static void _f32_to_bf16(const float* src, uint16_t* dst, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            uint32_t u;
            std::memcpy(&u, src + i, 4);
            dst[i] = static_cast<uint16_t>(u >> 16);
        }
    }

    // Convert n elements from src (bf16) → dst (f32).
    static void _bf16_to_f32(const uint16_t* src, float* dst, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            uint32_t u = static_cast<uint32_t>(src[i]) << 16;
            std::memcpy(dst + i, &u, 4);
        }
    }

    // Generic copy with optional type conversion: slot_buffer (m_storage_type) → dst (m_elem_type).
    void _gather_slot_to(const uint8_t* src_slot, uint8_t* dst, size_t n_elems) const {
        if (m_storage_type == m_elem_type) {
            std::memcpy(dst, src_slot, n_elems * m_elem_type.size());
        } else if (m_storage_type == ov::element::bf16 && m_elem_type == ov::element::f32) {
            _bf16_to_f32(reinterpret_cast<const uint16_t*>(src_slot),
                         reinterpret_cast<float*>(dst), n_elems);
        } else {
            // Fallback: shouldn't happen with supported combinations.
            std::memcpy(dst, src_slot, n_elems * m_storage_type.size());
        }
    }

    // Generic copy with optional type conversion: src (m_elem_type) → slot_buffer (m_storage_type).
    void _store_to_slot(const uint8_t* src, uint8_t* dst_slot, size_t n_elems) const {
        if (m_storage_type == m_elem_type) {
            std::memcpy(dst_slot, src, n_elems * m_elem_type.size());
        } else if (m_elem_type == ov::element::f32 && m_storage_type == ov::element::bf16) {
            _f32_to_bf16(reinterpret_cast<const float*>(src),
                         reinterpret_cast<uint16_t*>(dst_slot), n_elems);
        } else {
            std::memcpy(dst_slot, src, n_elems * m_storage_type.size());
        }
    }

    // -------------------------------------------------------------------
    // Run projector and return deep-copies of K/V outputs.
    // Serialized via m_proj_mutex so concurrent admits don't corrupt outputs.
    // Does NOT hold m_mutex so the main thread can call release() freely.
    // -------------------------------------------------------------------
    using TensorVec = std::vector<ov::Tensor>;
    std::pair<TensorVec, TensorVec>
    _run_projector(const ov::Tensor& encoder_hidden_state) {
        std::lock_guard<std::mutex> plk(m_proj_mutex);
        m_proj_request.set_tensor("encoder_hidden_states", encoder_hidden_state);
        const auto t0 = std::chrono::steady_clock::now();
        m_proj_request.infer();
        m_proj_infer_us += std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t0).count();
        return _copy_proj_outputs(m_proj_request);
    }

    // Copy projector outputs (K/V per layer) into freshly-allocated tensors.
    // Stores in m_storage_type (converting from m_elem_type if they differ).
    std::pair<TensorVec, TensorVec>
    _copy_proj_outputs(ov::InferRequest& req) {
        const size_t n_elems = _slot_elem_count();
        const size_t storage_bytes = n_elems * m_storage_type.size();
        TensorVec proj_K(m_n_layers), proj_V(m_n_layers);
        for (size_t l = 0; l < m_n_layers; ++l) {
            ov::Tensor Ks = req.get_tensor("cross_proj_K_" + std::to_string(l));
            ov::Tensor Vs = req.get_tensor("cross_proj_V_" + std::to_string(l));
            proj_K[l] = ov::Tensor(m_storage_type, Ks.get_shape());
            proj_V[l] = ov::Tensor(m_storage_type, Vs.get_shape());
            _store_to_slot(static_cast<const uint8_t*>(Ks.data()),
                           static_cast<uint8_t*>(proj_K[l].data()), n_elems);
            _store_to_slot(static_cast<const uint8_t*>(Vs.data()),
                           static_cast<uint8_t*>(proj_V[l].data()), n_elems);
        }
        return {proj_K, proj_V};
    }

    // Assign a slot from the free list and copy K/V data into the buffer.
    // Phase 1 (under lock): pop free slot, update bookkeeping, release lock.
    // Phase 2 (lock-free):  memcpy data into slot — no other thread can use this
    //   slot until the sequence group is pushed to m_awaiting_requests (after this
    //   function returns), so there is no read-before-write race.
    void _store_from_tensors(uint64_t req_id,
                             const TensorVec& proj_K, const TensorVec& proj_V,
                             std::unique_lock<std::mutex>& lk) {
        OPENVINO_ASSERT(!m_free_slots.empty(), "_store_from_tensors: no free slot");
        const uint32_t slot = m_free_slots.back();
        m_free_slots.pop_back();
        m_req_to_slot[req_id] = slot;
        const size_t n_active = m_req_to_slot.size();
        lk.unlock();  // release mutex BEFORE the big memcpy

        const size_t slot_bytes = _slot_elem_count() * m_storage_type.size();
        for (size_t l = 0; l < m_n_layers; ++l) {
            std::memcpy(_slot_ptr(l, 0, slot), proj_K[l].data(), slot_bytes);
            std::memcpy(_slot_ptr(l, 1, slot), proj_V[l].data(), slot_bytes);
        }
        std::cout << "[CrossKVCache] Admitted req " << req_id << " → slot " << slot
                  << " (active=" << n_active << "/" << m_max_slots << ")\n";
    }
};
