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
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/tensor.hpp"

// GPU-side gather kernel: eliminates the per-batch-change PCIe blit by keeping
// the cross-KV slot buffer in VRAM and scattering with an OpenCL kernel.
// Compiled when cmake finds OpenCL (ENABLE_CROSS_KV_OCL is defined).
// Uses only raw C OpenCL API (CL/cl.h) + OV's RemoteContext property headers;
// no dependency on C++ wrapper headers (CL/opencl.hpp / CL/cl2.hpp).
#ifdef ENABLE_CROSS_KV_OCL
#  include "openvino/runtime/intel_gpu/remote_properties.hpp"
#  include <CL/cl.h>
#endif

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

        // Attempt to allocate the gather workspace in GPU VRAM.
        // On GPU, OV blits every host tensor on every infer() call, even when set_tensor
        // is skipped.  Keeping workspace tensors device-resident eliminates that blit
        // for stable-batch GENERATE steps (the dominant phase).
        // On failure (CPU device / no context) m_is_gpu stays false and the existing
        // CPU workspace path is used unchanged.
        try {
            m_remote_ctx = m_proj_request.get_compiled_model().get_context();
            if (!m_remote_ctx) throw std::runtime_error("null context");
            m_device_K.resize(n_layers);
            m_device_V.resize(n_layers);
            m_alias_device_kv.resize(n_layers);
            // Pre-allocate at N=1; reallocated (cheap clCreateBuffer) on first N>1 gather.
            const ov::Shape shape1{1, n_heads, T_enc, head_dim};
            for (size_t l = 0; l < n_layers; ++l) {
                m_device_K[l] = m_remote_ctx.create_tensor(elem_type, shape1);
                m_device_V[l] = m_remote_ctx.create_tensor(elem_type, shape1);
                m_alias_device_kv[l][0] = m_device_K[l];
                m_alias_device_kv[l][1] = m_device_V[l];
            }
            m_last_device_N = 1;
            m_is_gpu = true;
            std::cout << "[CrossKVCache] GPU workspace: "
                      << n_layers << " × 2 VRAM tensors allocated"
                      << " (blitter-free stable-batch decode)\n";
        } catch (const std::exception& ex) {
            std::cout << "[CrossKVCache] CPU workspace mode"
                      << " (no GPU context: " << ex.what() << ")\n";
        }

        // GPU-side gather: runs the slot-gather entirely on GPU memory (GDDR6 BW)
        // eliminating the large H2D blit that causes alternating compute/blitter on
        // batch-composition changes.  Falls back gracefully if OCL is unavailable.
#ifdef ENABLE_CROSS_KV_OCL
        if (m_is_gpu) {
            _init_cl_gather();
        }
#endif
    }

    ~CrossKVCache() {
#ifdef ENABLE_CROSS_KV_OCL
        _cleanup_cl_gather();
#endif
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
    // n_free_slots(): number of slots available right now.
    // Called from step() thread before _flush_pending_projections() to
    // decide how many requests can be batch-projected without blocking.
    // -------------------------------------------------------------------
    size_t n_free_slots() const {
        std::lock_guard<std::mutex> lk(m_mutex);
        return m_free_slots.size();
    }

    // -------------------------------------------------------------------
    // add_proj_infer_us(): accumulate external projector timing into the
    // internal counter so get_proj_infer_us() stays accurate when
    // projection is handled outside CrossKVCache (batched path).
    // -------------------------------------------------------------------
    void add_proj_infer_us(double us) {
        std::lock_guard<std::mutex> plk(m_proj_mutex);
        m_proj_infer_us += us;
    }

    // -------------------------------------------------------------------
    // admit_precomputed_batch(): store per-request K/V slices from a single
    // batched projector run into their allocated slots.
    //
    // Precondition: m_proj_infer_request.infer() was already called with a
    // [N, T_enc, D] input; outputs have shape [N, n_heads, T_enc, head_dim].
    // Precondition: at least N free slots are available (caller must have
    // checked n_free_slots() >= N before calling — no blocking occurs here).
    // Must be called from the step() thread (non-blocking by design).
    // -------------------------------------------------------------------
    void admit_precomputed_batch(const std::vector<uint64_t>& req_ids,
                                 ov::InferRequest& proj_req) {
        const size_t N = req_ids.size();
        if (N == 0) return;
        const size_t n_elems = _slot_elem_count();

        // Pre-extract all layer output data pointers — single-threaded, safe.
        // Avoids calling get_tensor() from multiple threads simultaneously.
        struct LayerPtrs { const uint8_t* K; const uint8_t* V; };
        std::vector<LayerPtrs> layer_ptrs(m_n_layers);
        for (size_t l = 0; l < m_n_layers; ++l) {
            layer_ptrs[l].K = static_cast<const uint8_t*>(
                proj_req.get_tensor("cross_proj_K_" + std::to_string(l)).data());
            layer_ptrs[l].V = static_cast<const uint8_t*>(
                proj_req.get_tensor("cross_proj_V_" + std::to_string(l)).data());
        }

        // Phase 1: assign ALL N slots under a single lock acquisition.
        std::vector<uint32_t> slots(N);
        {
            std::unique_lock<std::mutex> lk(m_mutex);
            OPENVINO_ASSERT(m_free_slots.size() >= N,
                "admit_precomputed_batch: not enough free slots — caller must check n_free_slots()");
            for (size_t i = 0; i < N; ++i) {
                slots[i] = m_free_slots.back();
                m_free_slots.pop_back();
                m_req_to_slot[req_ids[i]] = slots[i];
            }
            std::cout << "[CrossKVCache] Batch-admitted " << N << " reqs"
                      << " (active=" << m_req_to_slot.size() << "/" << m_max_slots << ")\n";
        }

        // Phase 2: parallel scatter — one async task per request.
        // Each task copies all layers for one request directly from projector
        // output to slot buffer.  No temporary tensors, one copy not two,
        // N tasks run in parallel saturating DRAM write bandwidth.
        const size_t src_stride = n_elems * m_elem_type.size();
        std::vector<std::future<void>> futs;
        futs.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            futs.push_back(std::async(std::launch::async,
                [this, i, &slots, &layer_ptrs, src_stride, n_elems]() {
                    for (size_t l = 0; l < m_n_layers; ++l) {
                        _store_to_slot(layer_ptrs[l].K + i * src_stride,
                                       _slot_ptr(l, 0, slots[i]), n_elems);
                        _store_to_slot(layer_ptrs[l].V + i * src_stride,
                                       _slot_ptr(l, 1, slots[i]), n_elems);
                    }
                }));
        }
        for (auto& f : futs) f.get();
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
            // Need set_tensor when N changes (N>1 → N=1) OR when the request changes at N=1.
            // The latter handles staggered injection: prompt_needs_solo fires two consecutive
            // N=1 forward passes in the same step (one per request). Without tracking req_id,
            // the second pass sees m_last_set_tensor_N==1 and skips set_tensor, leaving OV
            // with the first request's stale K/V tensors bound for the second request's decode.
            out_need_set_tensor = (m_last_set_tensor_N != 1) || (req_ids[0] != m_last1_req_id);
            if (out_need_set_tensor) {
                m_last_set_tensor_N = 1;
                m_last1_req_id = req_ids[0];
            }

            // GPU path: upload the CPU staging tensor to a VRAM N=1 tensor once per
            // request change; thereafter OV reads from VRAM without any DMA blit.
            if (m_is_gpu) {
                if (out_need_set_tensor) {
                    // Reallocate device tensor if we're transitioning from N>1.
                    if (m_last_device_N != 1) {
                        for (size_t l = 0; l < m_n_layers; ++l) {
                            m_device_K[l] = m_remote_ctx.create_tensor(m_elem_type, shape1);
                            m_device_V[l] = m_remote_ctx.create_tensor(m_elem_type, shape1);
                            m_alias_device_kv[l][0] = m_device_K[l];
                            m_alias_device_kv[l][1] = m_device_V[l];
                        }
                        m_last_device_N = 1;
                    }
                    for (size_t l = 0; l < m_n_layers; ++l) {
                        m_alias1_kv[l][0].copy_to(m_device_K[l]);
                        m_alias1_kv[l][1].copy_to(m_device_V[l]);
                    }
                }
                return m_alias_device_kv;
            }
            return m_alias1_kv;
        }

        // ---------------------------------------------------------------
        // N>1 fast path: identical batch AND shape already set → nothing to do.
        // Must also check m_last_set_tensor_N == N to guard against the case where
        // m_last_req_ids matches (leftover from a prior scenario or prior step) but
        // the OV tensor binding was downgraded to N=1 by intervening solo steps.
        // Without the shape guard, OV would use the stale N=1 tensor and every
        // sequence above index 0 would read the first request's K/V.
        // ---------------------------------------------------------------
        if (req_ids == m_last_req_ids && m_last_set_tensor_N == N) {
            out_need_set_tensor = false;
            // GPU: return VRAM tensors — OV reads from device, zero DMA blit.
            return m_is_gpu ? m_alias_device_kv : m_alias_kv;
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

        // GPU plugins upload host tensors to device memory inside set_tensor().
        // Skipping set_tensor() when only the *content* changed (same N, different
        // req_ids) leaves stale device-side cross-KV → wrong attention → repetitive
        // output ("in in in...").  Always call set_tensor() when data was actually
        // re-gathered (req_ids changed).  On CPU the extra call is a cheap pointer
        // store; on GPU it triggers the required device upload.
        const bool shape_changed = (N != m_last_set_tensor_N);
        if (shape_changed) m_last_set_tensor_N = N;
        out_need_set_tensor = true;  // always: data changed whenever we got past the early-exit above

        // ---------------------------------------------------------------
        // GPU gather via OpenCL kernel (preferred path on GPU):
        //   Uploads only the slot-index array (N × 4 B ≈ 64 B for N=16),
        //   then runs a kernel that gathers directly within VRAM at GPU
        //   memory bandwidth (~512 GB/s) instead of PCIe (~25 GB/s).
        //   The slot data was uploaded to the VRAM slot buffer once at
        //   admit() time, so the hot decode loop is PCIe-free.
        // ---------------------------------------------------------------
#ifdef ENABLE_CROSS_KV_OCL
        if (m_cl_gather) {
            _run_cl_gather_Ngt1(cur_slots, N, shape_changed);
            return m_alias_device_kv;
        }
#endif

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

        // GPU path: upload gathered CPU workspace to VRAM once per batch change.
        // Subsequent infer() calls on the stable batch read from device — no DMA.
        if (m_is_gpu) {
            const ov::Shape shapeN{N, m_n_heads, m_T_enc, m_head_dim};
            if (m_last_device_N != N) {
                for (size_t l = 0; l < m_n_layers; ++l) {
                    m_device_K[l] = m_remote_ctx.create_tensor(m_elem_type, shapeN);
                    m_device_V[l] = m_remote_ctx.create_tensor(m_elem_type, shapeN);
                    m_alias_device_kv[l][0] = m_device_K[l];
                    m_alias_device_kv[l][1] = m_device_V[l];
                }
                m_last_device_N = N;
            }
            for (size_t l = 0; l < m_n_layers; ++l) {
                // Temporary CPU aliases with exact {N, ...} shape for the copy.
                ov::Tensor stg_K(m_elem_type, shapeN, m_batch_raw_K[l].data());
                ov::Tensor stg_V(m_elem_type, shapeN, m_batch_raw_V[l].data());
                stg_K.copy_to(m_device_K[l]);
                stg_V.copy_to(m_device_V[l]);
            }
            return m_alias_device_kv;
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
    // Last N for which set_tensor was actually called; alias tensors are rebuilt on shape change.
    size_t m_last_set_tensor_N = 0;
    // Last request ID for which set_tensor was called in the N=1 path.
    // Used to detect request changes at N=1 (e.g. prompt_needs_solo: A-solo then B-solo
    // in the same step), where N stays 1 but the request — and therefore its K/V slot — changes.
    uint64_t m_last1_req_id = static_cast<uint64_t>(-1);

    // --- GPU device-resident workspace -----------------------------------
    // When the model runs on a GPU device, OV blits every CPU-backed tensor
    // into VRAM on every infer() call, even when set_tensor() is skipped.
    // Keeping the gather workspace in VRAM eliminates that per-step blit for
    // stable-batch GENERATE steps (the hot path).
    //
    // m_is_gpu: true when GPU context was found at construction time.
    // m_remote_ctx: the RemoteContext used to allocate device tensors.
    // m_device_K/V[l]: VRAM tensors, shape {N, n_heads, T_enc, head_dim};
    //   reallocated (cheap clCreateBuffer) only when N changes.
    // m_alias_device_kv[l]: {m_device_K[l], m_device_V[l]} — returned by
    //   gather_batch when m_is_gpu; these are device tensors, no CPU blit.
    // m_last_device_N: current N for which m_device_K/V are allocated.
    bool m_is_gpu = false;
    ov::RemoteContext m_remote_ctx;
    size_t m_last_device_N = 0;
    std::vector<ov::Tensor> m_device_K;
    std::vector<ov::Tensor> m_device_V;
    std::vector<std::array<ov::Tensor, 2>> m_alias_device_kv;

    // ---- GPU-gather members (ENABLE_CROSS_KV_OCL only) -----------------
    // When m_cl_gather is true:
    //   m_cl_slot_buf  — VRAM mirror of m_buffer (populated on every admit)
    //   m_cl_layer_K/V — per-layer output scratch [max_slots × elems]; wrapped
    //                    as OV RemoteTensors in m_device_K/V with shape {N,...}
    //   m_cl_indices   — device buffer holding current batch's slot indices [N]
    //   m_gather_kernels[l*2+kv] — per-(layer,kv) OCL kernel; fixed args set at
    //                    init; {output cl_mem, N} updated only on shape change
    // -------------------------------------------------------------------
#ifdef ENABLE_CROSS_KV_OCL
    bool             m_cl_gather  = false;
    cl_context       m_cl_ctx_raw = nullptr;   // AddRef'd by us; released in dtor
    cl_device_id     m_cl_device  = nullptr;
    cl_command_queue m_cl_queue   = nullptr;
    cl_program       m_cl_prog    = nullptr;
    cl_mem           m_cl_slot_buf = nullptr;  // VRAM slot buffer (same layout as m_buffer)
    std::vector<cl_mem> m_cl_layer_K;          // [n_layers] each max_slots × elems
    std::vector<cl_mem> m_cl_layer_V;
    cl_mem           m_cl_indices  = nullptr;  // [max_slots] int32 — indices for current batch
    std::vector<cl_kernel> m_gather_kernels;   // [n_layers * 2] one per (layer, kv)
#endif  // ENABLE_CROSS_KV_OCL

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

    // TensorVec must be declared before #ifdef ENABLE_CROSS_KV_OCL so the OCL
    // helper methods can reference it in their parameter types.
    using TensorVec = std::vector<ov::Tensor>;

#ifdef ENABLE_CROSS_KV_OCL
    // -------------------------------------------------------------------
    // OpenCL gather kernel source.
    // Instantiated once per (layer, kv) pair.  All fixed args set at init
    // time; only {output cl_mem, N} are updated when the batch size changes.
    //
    // Global work size per kernel = N × elems_per_slot
    //   gid / elems_per_slot → batch index i
    //   gid % elems_per_slot → element within slot
    //   indices[i]           → physical slot in slot_buf
    // -------------------------------------------------------------------
    static constexpr const char* kGatherKernelSrc =
        "__kernel void gather_kv_layer(\n"
        "    __global const ETYPE* slot_buf,\n"   // [n_layers, 2, max_slots, elems]
        "    __global const int*   indices,\n"     // [N]
        "    __global       ETYPE* out,\n"          // [N, elems] for one (layer, kv)
        "    int  max_slots,\n"
        "    int  elems_per_slot,\n"
        "    int  N,\n"
        "    long src_base\n"                      // element offset to start of this layer+kv
        ") {\n"
        "    size_t gid = get_global_id(0);\n"
        "    if (gid >= (size_t)N * elems_per_slot) return;\n"
        "    int i   = (int)(gid / elems_per_slot);\n"
        "    int off = (int)(gid % elems_per_slot);\n"
        "    int slot = indices[i];\n"
        "    out[gid] = slot_buf[src_base + (long)slot * elems_per_slot + off];\n"
        "}\n";

    // Compile kernel, allocate VRAM slot buffer and per-layer output buffers.
    // Called once from the constructor when m_is_gpu == true.
    void _init_cl_gather() {
        OPENVINO_ASSERT(m_storage_type == m_elem_type,
            "[CrossKVCache] GPU gather requires storage_type == elem_type "
            "(set CROSS_KV_PRECISION to match the model's compute precision).");
        try {
            // Extract cl_context via OV RemoteContext property API.
            // Avoids including ocl.hpp (which requires C++ OpenCL wrapper headers).
            cl_context raw_ctx = static_cast<cl_context>(
                m_remote_ctx.get_params()
                    .at(ov::intel_gpu::ocl_context.name())
                    .template as<ov::intel_gpu::gpu_handle_param>());
            clRetainContext(raw_ctx);
            m_cl_ctx_raw = raw_ctx;

            // Enumerate devices in this context.
            cl_uint n_devs = 0;
            cl_int err = clGetContextInfo(raw_ctx, CL_CONTEXT_NUM_DEVICES, sizeof(n_devs), &n_devs, nullptr);
            OPENVINO_ASSERT(err == CL_SUCCESS && n_devs > 0, "[CrossKVCache] clGetContextInfo DEVICES failed");
            clGetContextInfo(raw_ctx, CL_CONTEXT_DEVICES, sizeof(m_cl_device), &m_cl_device, nullptr);

            // Create in-order command queue.
            cl_int qerr;
            m_cl_queue = clCreateCommandQueue(raw_ctx, m_cl_device, 0, &qerr);
            OPENVINO_ASSERT(qerr == CL_SUCCESS, "[CrossKVCache] clCreateCommandQueue failed: " + std::to_string(qerr));

            // Build kernel with element-type define.
            const char* etype_str = (m_elem_type == ov::element::bf16) ? "ushort"
                                  : (m_elem_type == ov::element::f16)  ? "half"
                                  :                                       "float";
            const std::string opts = std::string("-D ETYPE=") + etype_str;
            const char* src = kGatherKernelSrc;
            const size_t src_len = std::strlen(src);
            m_cl_prog = clCreateProgramWithSource(raw_ctx, 1, &src, &src_len, &err);
            OPENVINO_ASSERT(err == CL_SUCCESS, "[CrossKVCache] clCreateProgramWithSource failed");

            err = clBuildProgram(m_cl_prog, 1, &m_cl_device, opts.c_str(), nullptr, nullptr);
            if (err != CL_SUCCESS) {
                char log[4096] = {};
                clGetProgramBuildInfo(m_cl_prog, m_cl_device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
                throw std::runtime_error(std::string("[CrossKVCache] clBuildProgram: ") + log);
            }

            // VRAM slot buffer: mirrors m_buffer in storage/compute type.
            // Layout: [n_layers, 2, max_slots, n_heads, T_enc, head_dim]
            const size_t slot_buf_bytes = m_n_layers * 2 * m_max_slots * _slot_elem_count() * m_elem_type.size();
            m_cl_slot_buf = clCreateBuffer(raw_ctx, CL_MEM_READ_WRITE, slot_buf_bytes, nullptr, &err);
            OPENVINO_ASSERT(err == CL_SUCCESS, "[CrossKVCache] clCreateBuffer slot_buf failed");

            // Per-layer output buffers (K and V), each [max_slots × elems].
            // Wrapped as OV tensors with shape {N, heads, T, hd} per gather call.
            const size_t layer_out_bytes = m_max_slots * _slot_elem_count() * m_elem_type.size();
            m_cl_layer_K.resize(m_n_layers, nullptr);
            m_cl_layer_V.resize(m_n_layers, nullptr);
            for (size_t l = 0; l < m_n_layers; ++l) {
                m_cl_layer_K[l] = clCreateBuffer(raw_ctx, CL_MEM_READ_WRITE, layer_out_bytes, nullptr, &err);
                OPENVINO_ASSERT(err == CL_SUCCESS, "[CrossKVCache] clCreateBuffer layer_K failed");
                m_cl_layer_V[l] = clCreateBuffer(raw_ctx, CL_MEM_READ_WRITE, layer_out_bytes, nullptr, &err);
                OPENVINO_ASSERT(err == CL_SUCCESS, "[CrossKVCache] clCreateBuffer layer_V failed");
            }

            // Indices buffer: [max_slots] int32.
            m_cl_indices = clCreateBuffer(raw_ctx, CL_MEM_READ_WRITE,
                                          m_max_slots * sizeof(cl_int), nullptr, &err);
            OPENVINO_ASSERT(err == CL_SUCCESS, "[CrossKVCache] clCreateBuffer indices failed");

            // Create one kernel per (layer, kv) and set all fixed args once.
            m_gather_kernels.resize(m_n_layers * 2, nullptr);
            const int max_sl  = static_cast<int>(m_max_slots);
            const int els     = static_cast<int>(_slot_elem_count());
            for (size_t l = 0; l < m_n_layers; ++l) {
                for (size_t kv = 0; kv < 2; ++kv) {
                    cl_kernel k = clCreateKernel(m_cl_prog, "gather_kv_layer", &err);
                    OPENVINO_ASSERT(err == CL_SUCCESS, "[CrossKVCache] clCreateKernel failed");

                    // Fixed args: slot_buf(0), indices(1), max_slots(3), elems(4), src_base(6).
                    // Variable args: out(2) and N(5) — set per gather call.
                    cl_mem out_buf = (kv == 0) ? m_cl_layer_K[l] : m_cl_layer_V[l];
                    const cl_long src_base = (static_cast<cl_long>(l) * 2 * m_max_slots
                                             + static_cast<cl_long>(kv) * m_max_slots)
                                            * els;
                    clSetKernelArg(k, 0, sizeof(cl_mem),  &m_cl_slot_buf);
                    clSetKernelArg(k, 1, sizeof(cl_mem),  &m_cl_indices);
                    clSetKernelArg(k, 2, sizeof(cl_mem),  &out_buf);   // initial; updated on N-change
                    clSetKernelArg(k, 3, sizeof(cl_int),  &max_sl);
                    clSetKernelArg(k, 4, sizeof(cl_int),  &els);
                    // arg 5 (N) intentionally left unset until first gather call
                    clSetKernelArg(k, 6, sizeof(cl_long), &src_base);
                    m_gather_kernels[l * 2 + kv] = k;
                }
            }

            m_cl_gather = true;
            std::cout << "[CrossKVCache] GPU gather kernel compiled (ETYPE=" << etype_str
                      << " slot_buf=" << slot_buf_bytes / (1024*1024) << " MB, "
                      << m_n_layers * 2 << " per-layer kernels)\n";

        } catch (const std::exception& ex) {
            std::cout << "[CrossKVCache] GPU gather kernel init failed: " << ex.what()
                      << " — batch recomposition will use H2D copy\n";
            _cleanup_cl_gather();
        }
    }

    void _cleanup_cl_gather() {
        for (auto k : m_gather_kernels) if (k) clReleaseKernel(k);
        m_gather_kernels.clear();
        for (auto b : m_cl_layer_K) if (b) clReleaseMemObject(b);
        m_cl_layer_K.clear();
        for (auto b : m_cl_layer_V) if (b) clReleaseMemObject(b);
        m_cl_layer_V.clear();
        if (m_cl_indices)  { clReleaseMemObject(m_cl_indices);  m_cl_indices  = nullptr; }
        if (m_cl_slot_buf) { clReleaseMemObject(m_cl_slot_buf); m_cl_slot_buf = nullptr; }
        if (m_cl_prog)     { clReleaseProgram(m_cl_prog);       m_cl_prog     = nullptr; }
        if (m_cl_queue)    { clReleaseCommandQueue(m_cl_queue);  m_cl_queue    = nullptr; }
        if (m_cl_ctx_raw)  { clReleaseContext(m_cl_ctx_raw);    m_cl_ctx_raw  = nullptr; }
        m_cl_gather = false;
    }

    // Upload a newly admitted slot's K/V data to the VRAM slot buffer.
    // proj_K[l] / proj_V[l] are CPU tensors in m_storage_type (== m_elem_type for GPU).
    // This is called from _store_from_tensors() on the admit thread — one-time per request.
    void _upload_slot_to_vram(uint32_t slot, const TensorVec& proj_K, const TensorVec& proj_V) {
        const size_t els      = _slot_elem_count();
        const size_t elem_sz  = m_elem_type.size();
        const size_t slot_bytes = els * elem_sz;
        for (size_t l = 0; l < m_n_layers; ++l) {
            // K: byte offset = (l * 2 * max_slots + slot) * slot_bytes
            const size_t K_off = (l * 2 * m_max_slots + slot) * slot_bytes;
            clEnqueueWriteBuffer(m_cl_queue, m_cl_slot_buf, CL_FALSE,
                                 K_off, slot_bytes, proj_K[l].data(), 0, nullptr, nullptr);
            // V: byte offset = (l * 2 * max_slots + max_slots + slot) * slot_bytes
            const size_t V_off = (l * 2 * m_max_slots + m_max_slots + slot) * slot_bytes;
            clEnqueueWriteBuffer(m_cl_queue, m_cl_slot_buf, CL_FALSE,
                                 V_off, slot_bytes, proj_V[l].data(), 0, nullptr, nullptr);
        }
        // Blocking finish: slot data must be in VRAM before admit() returns and the
        // request becomes schedulable for the first GENERATE step.
        clFinish(m_cl_queue);
    }

    // Run gather kernel for N>1.  Uploads indices (tiny H2D), updates per-kernel
    // output/N args on shape change, then enqueues all (layer × kv) kernels.
    // Precondition: cur_slots.size() == N, all slots valid.
    void _run_cl_gather_Ngt1(const std::vector<uint32_t>& cur_slots,
                              size_t N, bool shape_changed) {
        // --- 1. Upload slot indices (N × 4 B ≈ 64 B for N=16) ---
        std::vector<cl_int> idx(N);
        for (size_t i = 0; i < N; ++i) idx[i] = static_cast<cl_int>(cur_slots[i]);
        // Non-blocking; the in-order queue ensures this completes before the kernels run.
        clEnqueueWriteBuffer(m_cl_queue, m_cl_indices, CL_FALSE, 0,
                             N * sizeof(cl_int), idx.data(), 0, nullptr, nullptr);

        // --- 2. On N-change: update per-kernel {output cl_mem, N} and rewrap OV tensors ---
        if (shape_changed) {
            const ov::Shape shapeN{N, m_n_heads, m_T_enc, m_head_dim};
            const cl_int n_cl = static_cast<cl_int>(N);
            // Wrap the pre-allocated per-layer VRAM buffers as OV RemoteTensors
            // using the property-map API so we don't need ocl.hpp C++ wrappers.
            const ov::AnyMap buf_params_base = {
                {ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::OCL_BUFFER}
            };
            for (size_t l = 0; l < m_n_layers; ++l) {
                cl_kernel kK = m_gather_kernels[l * 2 + 0];
                cl_kernel kV = m_gather_kernels[l * 2 + 1];
                clSetKernelArg(kK, 2, sizeof(cl_mem), &m_cl_layer_K[l]);
                clSetKernelArg(kK, 5, sizeof(cl_int), &n_cl);
                clSetKernelArg(kV, 2, sizeof(cl_mem), &m_cl_layer_V[l]);
                clSetKernelArg(kV, 5, sizeof(cl_int), &n_cl);
                ov::AnyMap kp = buf_params_base;
                kp[ov::intel_gpu::mem_handle.name()] =
                    static_cast<ov::intel_gpu::gpu_handle_param>(m_cl_layer_K[l]);
                m_device_K[l] = m_remote_ctx.create_tensor(m_elem_type, shapeN, kp);
                kp[ov::intel_gpu::mem_handle.name()] =
                    static_cast<ov::intel_gpu::gpu_handle_param>(m_cl_layer_V[l]);
                m_device_V[l] = m_remote_ctx.create_tensor(m_elem_type, shapeN, kp);
                m_alias_device_kv[l][0] = m_device_K[l];
                m_alias_device_kv[l][1] = m_device_V[l];
            }
            m_last_device_N = N;
        }

        // --- 3. Enqueue per-(layer, kv) gather kernels ---
        // Global size: N × elems_per_slot.  Rounded up to a multiple of 64 for
        // better occupancy; the kernel bounds-checks so padding items are no-ops.
        const size_t base_gs  = N * _slot_elem_count();
        const size_t local_sz = 64;
        const size_t gs       = (base_gs + local_sz - 1) / local_sz * local_sz;
        for (size_t lk = 0; lk < m_n_layers * 2; ++lk)
            clEnqueueNDRangeKernel(m_cl_queue, m_gather_kernels[lk], 1,
                                   nullptr, &gs, &local_sz, 0, nullptr, nullptr);

        // Blocking finish: gathered VRAM tensors must be ready before infer() reads them.
        clFinish(m_cl_queue);
    }
#endif  // ENABLE_CROSS_KV_OCL

    // -------------------------------------------------------------------
    // Run projector and return deep-copies of K/V outputs.
    // Serialized via m_proj_mutex so concurrent admits don't corrupt outputs.
    // Does NOT hold m_mutex so the main thread can call release() freely.
    // -------------------------------------------------------------------
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

        // Mirror admitted slot to VRAM slot buffer so the GPU gather kernel
        // can read from it directly without a per-gather H2D upload.
        // This is a one-time cost per request admit; the hot decode loop is free.
#ifdef ENABLE_CROSS_KV_OCL
        if (m_cl_gather)
            _upload_slot_to_vram(slot, proj_K, proj_V);
#endif

        std::cout << "[CrossKVCache] Admitted req " << req_id << " → slot " << slot
                  << " (active=" << n_active << "/" << m_max_slots << ")\n";
    }
};
