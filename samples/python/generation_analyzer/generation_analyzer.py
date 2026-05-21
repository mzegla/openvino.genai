#!/usr/bin/env python3
"""
generation_analyzer.py — Logit-space monitoring for LLM inference.

Supports OpenVINO GenAI (ContinuousBatchingPipeline), llama.cpp
(llama-cpp-python), and HuggingFace Transformers (full-precision reference)
as generation backends.

Modes:

  1. Generate + analyse — single runtime:
       python3 generation_analyzer.py <MODEL> <PROMPT_OR_FILE.txt>
       python3 generation_analyzer.py <MODEL.gguf> <PROMPT> --runtime llamacpp
       python3 generation_analyzer.py <MODEL_HF_DIR> <PROMPT> --runtime hf

  2. Dual-runtime comparison (apples-to-apples):
       python3 generation_analyzer.py <MODEL_A> <PROMPT> \\
           --compare <MODEL_B.gguf> \\
           [--runtime ovgenai] [--runtime-b llamacpp] \\
           --no-sample   # greedy = isolates quantization effects

  3. Dual-runtime comparison with full-precision reference:
       python3 generation_analyzer.py <MODEL_A> <PROMPT> \\
           --compare <MODEL_B> --runtime-b llamacpp \\
           --reference <HF_MODEL_DIR> \\
           --no-sample --shared-tokenizer llamacpp

  4. Single-model vs full-precision reference:
       python3 generation_analyzer.py <MODEL> <PROMPT> \\
           --reference <HF_MODEL_DIR> \\
           --no-sample --shared-tokenizer llamacpp

  5. Analyse an existing log file (no model required):
       python3 generation_analyzer.py --analyze <logits_stats.log>

Generation options:
    --runtime R          ovgenai | llamacpp | hf        (default: ovgenai)
    --device CPU|GPU     OpenVINO device                (default: CPU)
    --max-tokens N       Maximum new tokens             (default: 2000)
    --window N           Tokens averaged per entry      (default: 10)
    --temperature T      Sampling temperature           (default: 0.9)
    --top-k K            Top-k sampling                 (default: 30)
    --top-p P            Top-p nucleus sampling         (default: 0.9)
    --no-sample          Greedy decoding
    --n-gpu-layers N     GPU layers to offload (llamacpp, default: 0)
    --n-ctx N            Context window size   (llamacpp, default: 4096)
    --n-threads N        CPU threads           (llamacpp, default: auto)

Comparison options:
    --compare MODEL_B    Second model for side-by-side comparison
    --runtime-b R        Runtime for MODEL_B  (default: llamacpp)
    --reference MODEL    Full-precision reference model path
    --runtime-ref R      Runtime for reference model   (default: hf)
    --shared-tokenizer R Tokenize once with this runtime and share IDs;
                         eliminates tokenizer divergence (ovgenai | llamacpp)

Output options:
    --log FILE           Log path  (default: logits_stats.log)
    --plot FILE          PNG path  (default: <log>.png; pass "none" to skip)

NOTE — HF backend speed:
    The hf runtime runs one full forward pass per generated token and loads
    the model at full precision (bf16/fp16).  For large models (>7B) this is
    very slow on CPU and requires substantial VRAM on GPU.  Use --max-tokens
    to limit the run length when using it as a reference baseline.
"""

import re
import sys
import copy
import math
import time
import argparse
import statistics
from pathlib import Path

# ============================================================================
# Diagnostic zones
# ============================================================================
ZONES = [
    # (max_H,  label,          hex_colour)
    (0.15,  "LOCK",       "#d62728"),
    (0.50,  "COLLAPSING", "#ff7f0e"),
    (1.50,  "NORMAL",     "#2ca02c"),
    (3.00,  "HIGH-H",     "#1f77b4"),
    (1e9,   "CHAOS",      "#9467bd"),
]

def zone_for(h):
    for max_h, label, colour in ZONES:
        if h <= max_h:
            return label, colour
    return ZONES[-1][1], ZONES[-1][2]

# ============================================================================
# Metrics computation  (raw float logit list -> dict)
# ============================================================================

def _softmax(logits):
    max_l = max(logits)
    exp_l = [math.exp(l - max_l) for l in logits]
    s = sum(exp_l)
    probs = [e / s for e in exp_l]
    log_sum_exp = max_l + math.log(s)
    log_probs = [l - log_sum_exp for l in logits]
    return probs, log_probs

def compute_metrics(logits):
    """Compute per-token distribution metrics from raw (pre-sampling) logits.

    Accepts a list of floats or a numpy array.  Uses numpy for vectorised
    computation — critical for large vocabularies such as Qwen3.6's 248k-token
    vocab where pure-Python loops are ~100x slower.  Falls back to pure Python
    when numpy is unavailable.
    """
    try:
        import numpy as _np
        lf  = _np.asarray(logits, dtype=_np.float64)
        V   = len(lf)
        if V == 0:
            return {}
        lf_s  = lf - lf.max()                         # shift for numerical stability
        exp_l = _np.exp(lf_s)
        s     = float(exp_l.sum())
        probs = exp_l / s
        lp    = lf_s - math.log(s)                    # log-probs (nats)
        H     = float(-_np.dot(probs, lp))
        VH    = float(_np.dot(probs, (-lp - H) ** 2))
        t1    = int(_np.argmax(probs))
        if V > 1:
            masked    = probs.copy(); masked[t1] = -1.0
            t2        = int(_np.argmax(masked))
        else:
            t2 = t1
        lstd   = float(_np.std(_np.asarray(logits, dtype=_np.float64)))
        k10    = min(10, V)
        top10  = _np.argpartition(probs, -k10)[-k10:]
        return dict(
            entropy              = H,
            varentropy           = VH,
            top1_prob            = float(probs[t1]),
            top1_top2_log_margin = float(lp[t1] - lp[t2]),
            eff_vocab            = math.exp(H),
            logit_std            = lstd,
            top10_mass           = float(probs[top10].sum()),
        )
    except ImportError:
        pass
    # ── pure-Python fallback (no numpy) ─────────────────────────────────────
    V = len(logits)
    if V == 0:
        return {}
    probs, log_probs = _softmax(logits)
    H  = -sum(p * lp for p, lp in zip(probs, log_probs) if p > 0)
    VH =  sum(p * ((-lp) - H) ** 2 for p, lp in zip(probs, log_probs) if p > 0)
    order   = sorted(range(V), key=lambda i: probs[i], reverse=True)
    top1    = order[0]
    top2    = order[1] if V > 1 else top1
    lmean   = sum(logits) / V
    lstd    = math.sqrt(sum((l - lmean) ** 2 for l in logits) / V)
    top10_mass = sum(probs[order[i]] for i in range(min(10, V)))
    return dict(
        entropy              = H,
        varentropy           = VH,
        top1_prob            = probs[top1],
        top1_top2_log_margin = log_probs[top1] - log_probs[top2],
        eff_vocab            = math.exp(H),
        logit_std            = lstd,
        top10_mass           = top10_mass,
    )

def _top_k_indices(logits, k=20):
    try:
        import numpy as _np
        lf = _np.asarray(logits, dtype=_np.float32)
        V  = len(lf)
        k  = min(k, V)
        if V <= k:
            return list(_np.argsort(lf)[::-1])
        idx = _np.argpartition(lf, -k)[-k:]
        return idx[_np.argsort(lf[idx])[::-1]].tolist()
    except ImportError:
        pass
    V = len(logits)
    order = sorted(range(V), key=lambda i: logits[i], reverse=True)
    return order[:min(k, V)]

def compute_jaccard(logits, prev_top20):
    if not prev_top20:
        return 0.0
    cur  = set(_top_k_indices(logits))
    prev = set(prev_top20)
    inter = len(cur & prev)
    union = len(cur | prev)
    return inter / union if union else 1.0

# ============================================================================
# EMA
# ============================================================================

def ema(values, alpha):
    out, s = [], values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        out.append(s)
    return out

# ============================================================================
# Phase segmentation
# ============================================================================

def segments(records):
    if not records:
        return []
    segs = []
    lbl0, col0 = zone_for(records[0]["H"])
    start = records[0]["tok"]
    for r in records[1:]:
        lbl, col = zone_for(r["H"])
        if lbl != lbl0:
            segs.append((start, r["tok"], lbl0, col0))
            lbl0, col0, start = lbl, col, r["tok"]
    segs.append((start, records[-1]["tok"], lbl0, col0))
    return segs

def dominant_phase(records):
    counts = {}
    for r in records:
        lbl, _ = zone_for(r["H"])
        counts[lbl] = counts.get(lbl, 0) + 1
    return max(counts, key=counts.get)

# ============================================================================
# Log writer  (writes formatted text file AND accumulates in-memory records)
# ============================================================================

class LogWriter:
    def __init__(self, path, window=10):
        self.path    = path
        self.window  = window
        self._buf    = []
        self.tok_n   = 0
        self.records = []          # window-averaged records for the dashboard
        self._fh     = open(path, "w")
        self._last_top20       = []   # top-20 of the most recently pushed token
        self._prev_flush_top20 = []   # window-level aggregate top-20 (for win-win Jaccard)
        self._buf_top20s       = []   # per-token top-20 lists accumulated in current window
        self.token_top20s      = []   # per-token top-20s for cross-model divergence analysis
        self.token_top100s     = []   # per-token top-100s for rank-in-other-model lookup

    def push(self, metrics, jaccard, top20=None, top100=None):
        m = dict(metrics); m["jaccard"] = jaccard
        self._buf.append(m)
        self.tok_n += 1
        if top20 is not None:
            self._last_top20 = top20
            self._buf_top20s.append(top20)
            self.token_top20s.append(list(top20))
        if top100 is not None:
            self.token_top100s.append(list(top100))
        elif top20 is not None:  # fallback: top100 = top20 if not separately provided
            self.token_top100s.append(list(top20))
        if len(self._buf) >= self.window:
            self._flush()

    def _flush(self):
        buf, self._buf = self._buf, []
        def avg(k): return statistics.mean(r[k] for r in buf)

        H   = avg("entropy")
        eff = avg("eff_vocab")
        jk  = avg("jaccard")

        # Window-to-window Jaccard: compare the aggregate top-20 of this window
        # vs the previous window.  We accumulate token frequencies across all
        # top-20 lists in the window and take the 20 most frequent tokens.
        # This is robust to cycle length — a repeating phrase of any period will
        # produce a stable aggregate regardless of where the window boundary falls.
        buf_top20s, self._buf_top20s = self._buf_top20s, []
        if buf_top20s:
            freq: dict = {}
            for t20 in buf_top20s:
                for tok_id in t20:
                    freq[tok_id] = freq.get(tok_id, 0) + 1
            win_top20 = [tok_id for tok_id, _ in
                         sorted(freq.items(), key=lambda x: -x[1])[:20]]
        else:
            win_top20 = list(self._last_top20)

        if self._prev_flush_top20 and win_top20:
            prev = set(self._prev_flush_top20)
            cur  = set(win_top20)
            inter = len(cur & prev)
            union = len(cur | prev)
            win_jk = inter / union if union else 1.0
        else:
            win_jk = float('nan')
        self._prev_flush_top20 = win_top20

        lines = [
            f"[logits_stats @ token {self.tok_n}]",
            f"LogitsStats [{len(buf)} steps]",
            f"  entropy         : {H:.4f} nats  (eff vocab: {eff:.2f})",
            f"  varentropy      : {avg('varentropy'):.4f}",
            f"  top-1 prob      : {avg('top1_prob'):.4f}",
            f"  top1/top2 margin: {avg('top1_top2_log_margin'):.4f} (log)",
            f"  logit std       : {avg('logit_std'):.4f}",
            f"  top-10 mass     : {avg('top10_mass'):.4f}",
            f"  jaccard top-20  : {jk:.4f}",
        ]
        if not math.isnan(win_jk):
            lines.append(f"  win-jaccard     : {win_jk:.4f}")
        self._fh.write("\n".join(lines) + "\n")
        self._fh.flush()

        # accumulate dashboard record
        self.records.append(dict(
            tok       = self.tok_n,
            H         = H,
            VH        = avg("varentropy"),
            top1      = avg("top1_prob"),
            eff_v     = eff,
            margin    = avg("top1_top2_log_margin"),
            logit_std = avg("logit_std"),
            top10     = avg("top10_mass"),
            jaccard   = jk,
            win_jaccard = win_jk,
        ))

    def close(self):
        if self._buf:
            self._flush()
        self._fh.close()

# ============================================================================
# Parse an existing log file  (for --analyze mode)
# ============================================================================

def parse_log(path):
    text = Path(path).read_text(errors="replace")
    blocks = re.split(r'\[logits_stats @ token (\d+)\]', text)
    it = iter(blocks)
    next(it)
    records = []
    for tok_str, block in zip(it, it):
        tok = int(tok_str)
        def g(pat, blk=block):
            m = re.search(pat, blk)
            return float(m.group(1)) if m else None
        H = g(r'entropy\s+:\s+([\d.]+)')
        if H is None:
            continue
        jk = g(r'jaccard top-20\s+:\s+([\d.]+)')
        wjk = g(r'win-jaccard\s+:\s+([\d.]+)')
        records.append(dict(
            tok       = tok,
            H         = H,
            VH        = g(r'varentropy\s+:\s+([\d.]+)') or 0,
            top1      = g(r'top-1 prob\s+:\s+([\d.]+)') or 0,
            eff_v     = g(r'eff vocab:\s+([\d.]+)') or 0,
            margin    = g(r'top1/top2 margin:\s+([\d.]+)') or 0,
            logit_std = g(r'logit std\s+:\s+([\d.]+)') or 0,
            top10     = g(r'top-10 mass\s+:\s+([\d.]+)') or 0,
            jaccard     = jk  if jk  is not None else float('nan'),
            win_jaccard = wjk if wjk is not None else float('nan'),
        ))
    return records

# ============================================================================
# Text summary
# ============================================================================

SHORT_PHASE  = {"LOCK": "LOC", "COLLAPSING": "COL", "NORMAL": "NOR",
                "HIGH-H": "HIG", "CHAOS": "CHA"}
PHASE_ORDER  = {"CHAOS": 4, "HIGH-H": 3, "NORMAL": 2, "COLLAPSING": 1, "LOCK": 0}

def temporal_dynamics(records):
    """Print the temporal dynamics section: phase runs, entropy trend, autocorrelation."""
    hs   = [r['H']   for r in records]
    toks = [r['tok'] for r in records]
    n    = len(records)

    # Phase run-length encoding
    phases = [zone_for(h)[0] for h in hs]
    runs = []
    for p in phases:
        if runs and runs[-1][0] == p:
            runs[-1][1] += 1
        else:
            runs.append([p, 1])
    seq_str = "→".join(
        f"{SHORT_PHASE.get(p, p[:3])}×{c}" if c > 1 else SHORT_PHASE.get(p, p[:3])
        for p, c in runs)

    # Longest streak per phase
    streak: dict = {}
    for p, c in runs:
        streak[p] = max(streak.get(p, 0), c)
    streak_str = "  ".join(
        f"{SHORT_PHASE.get(p,p[:3])}={c}"
        for p, c in sorted(streak.items(), key=lambda x: -x[1]))

    # Entropy slope (OLS)
    mean_t = sum(toks) / n
    mean_h = sum(hs)   / n
    denom  = sum((t - mean_t) ** 2 for t in toks)
    slope  = (sum((t - mean_t) * (h - mean_h) for t, h in zip(toks, hs)) / denom
              if denom > 0 else 0.0)
    if   slope < -0.003: direction = "↓ collapsing"
    elif slope >  0.003: direction = "↑ diverging"
    else:                direction = "→ flat"

    # H autocorrelation at lag 1  (measures phase persistence)
    var_h = (sum((h - mean_h) ** 2 for h in hs) / n) if n > 1 else 0
    if var_h > 1e-9 and n >= 3:
        autocorr = (sum((hs[i] - mean_h) * (hs[i+1] - mean_h) for i in range(n-1))
                    / ((n - 1) * var_h))
        if   autocorr > 0.7: ac_label = "stable attractor"
        elif autocorr > 0.2: ac_label = "moderate switching"
        else:                ac_label = "oscillating / erratic"
        ac_str = f"{autocorr:.3f}  ({ac_label})"
    else:
        ac_str = "n/a"

    # Top-1 slope
    t1s    = [r['top1'] for r in records]
    mean_t1 = sum(t1s) / n
    slope_t1 = (sum((t - mean_t) * (p - mean_t1) for t, p in zip(toks, t1s)) / denom
                if denom > 0 else 0.0)
    if   slope_t1 < -0.001: t1_dir = "↓ becoming uncertain"
    elif slope_t1 >  0.001: t1_dir = "↑ becoming overconfident"
    else:                   t1_dir = "→ stable confidence"

    # Window-to-window Jaccard stats
    wjks = [r.get('win_jaccard', float('nan')) for r in records]
    wjks_valid = [w for w in wjks if not math.isnan(w)]

    print("  ── Temporal dynamics ─────────────────────────────────────────")
    print(f"  Phase sequence  : {seq_str}")
    print(f"  Longest streak  : {streak_str}")
    print(f"  H slope         : {slope:+.4f}/tok  ({direction})")
    print(f"  top-1 slope     : {slope_t1:+.4f}/tok  ({t1_dir})")
    print(f"  H autocorr(lag1): {ac_str}")
    if wjks_valid:
        mean_wjk = statistics.mean(wjks_valid)
        wjk_note = ""
        if mean_wjk > 0.55:
            wjk_note = "  ← HIGH: same token-set looping window-to-window"
        print(f"  win-win Jaccard : mean={mean_wjk:.3f}  "
              f"min={min(wjks_valid):.3f}  max={max(wjks_valid):.3f}{wjk_note}")
        # Show per-window trajectory when elevated (mean > 0.20) so loop
        # onset and any subsequent drop are immediately visible.
        if mean_wjk > 0.30:
            wjk_all = [r.get('win_jaccard', float('nan')) for r in records]
            cells   = []
            for w in wjk_all:
                if math.isnan(w):
                    cells.append("  -- ")
                elif w >= 0.55:
                    cells.append(f"\033[91m{w:.2f}\033[0m")   # red = high
                elif w >= 0.40:
                    cells.append(f"\033[93m{w:.2f}\033[0m")   # yellow = elevated
                else:
                    cells.append(f"{w:.2f}")
            # Wrap into rows of 10 for readability
            row_size = 10
            for row_start in range(0, len(cells), row_size):
                row  = cells[row_start:row_start + row_size]
                toks = records[row_start + row_size - 1]['tok'] if row_start + row_size <= len(records) else records[-1]['tok']
                print(f"    tok ~{records[min(row_start, len(records)-1)]['tok']:>3}–{toks:<3}  " + "  ".join(row))
    print()


def early_warning_signals(records):
    """Scan records chronologically for notable events and emit forward predictions."""
    if len(records) < 2:
        return

    window = records[1]['tok'] - records[0]['tok'] if len(records) > 1 else 10
    signals = []   # (tok_start, tok_end, level, message)

    # Pre-compute isolated CHAOS windows so transition handler can emit
    # the richer hallucination-risk message directly (avoids duplicate signals).
    _phases_all = [zone_for(r['H'])[0] for r in records]
    _isolated_chaos_idx = {
        i for i in range(1, len(records) - 1)
        if (_phases_all[i] == "CHAOS"
            and _phases_all[i-1] != "CHAOS"
            and _phases_all[i+1] != "CHAOS")}

    for i, r in enumerate(records):
        tok_end   = r['tok']
        tok_start = records[i - 1]['tok'] + 1 if i > 0 else tok_end - window + 1
        label, _  = zone_for(r['H'])

        if i == 0:
            if r['H'] > 3.0:
                signals.append((tok_start, tok_end, "INFO",
                    f"Starts in CHAOS (H={r['H']:.2f}) — model searching for topic"))
            elif r['H'] < 0.3:
                signals.append((tok_start, tok_end, "WARN",
                    f"Starts COLLAPSING (H={r['H']:.2f}) — prompt may be trivially constrained"))
            continue

        prev        = records[i - 1]
        prev_label, _ = zone_for(prev['H'])
        delta_h   = r['H']    - prev['H']
        delta_t1  = r['top1'] - prev['top1']
        delta_vh  = r['VH']   - prev['VH']

        # Look-ahead for trend confirmation (avoids single-window false alarms)
        has_next   = i + 1 < len(records)
        next_label = zone_for(records[i + 1]['H'])[0] if has_next else label
        next_rank  = PHASE_ORDER.get(next_label, 2)
        # 2-window cumulative H change when all 3 windows are in the same phase
        has_prev2  = i >= 2 and zone_for(records[i - 2]['H'])[0] == label
        delta_h_2w = r['H'] - records[i - 2]['H'] if has_prev2 else None

        # ── Phase transitions ────────────────────────────────────────────
        if prev_label != label:
            rank_p = PHASE_ORDER.get(prev_label, 2)
            rank_c = PHASE_ORDER.get(label,      2)
            if rank_c < rank_p:                              # descending
                if label == "LOCK":
                    signals.append((tok_start, tok_end, "WARN",
                        f"{prev_label}→LOCK (H={r['H']:.3f}) — "
                        "single-token repetition attractor reached"))
                elif label == "COLLAPSING":
                    # Only signal if next window also stays in COLLAPSING/LOCK
                    # — a single-window dip and bounce is one high-conf token, not a trend
                    if not has_next or next_rank <= PHASE_ORDER["COLLAPSING"]:
                        signals.append((tok_start, tok_end, "WARN",
                            f"{prev_label}→COLLAPSING (H={prev['H']:.2f}→{r['H']:.2f}) — "
                            "distribution tightening, loop may form ahead"))
            elif rank_c > rank_p:                            # ascending
                if label == "CHAOS":
                    if i in _isolated_chaos_idx:
                        # Single isolated spike — emit hallucination-risk directly
                        signals.append((tok_start, tok_end, "WARN",
                            f"Isolated CHAOS spike (H={r['H']:.2f}) — "
                            "high hallucination / invented-fact risk; "
                            f"{prev_label}→CHAOS→{_phases_all[i+1]}"))
                    else:
                        signals.append((tok_start, tok_end, "WARN",
                            f"{prev_label}→CHAOS (H={r['H']:.2f}) — "
                            "sustained uncertainty spike; possible hallucination or topic jump"))
                elif prev_label == "LOCK" and label == "COLLAPSING":
                    # Exiting single-token lock but still very confident
                    signals.append((tok_start, tok_end, "NOTE",
                        f"LOCK→COLLAPSING (H={prev['H']:.3f}→{r['H']:.2f}) — "
                        "single-token lock eased; still narrow distribution"))
                elif prev_label in ("LOCK", "COLLAPSING") and label in ("NORMAL", "HIGH-H"):
                    # Recovery: confirm it persists (not a one-window bounce)
                    if not has_next or next_rank >= rank_c:
                        signals.append((tok_start, tok_end, "NOTE",
                            f"{prev_label}→{label} (H={prev['H']:.2f}→{r['H']:.2f}) — "
                            "entropy recovering from over-confident state"))
                else:
                    # General ascending: confirm the new phase persists
                    if not has_next or next_rank >= rank_c:
                        signals.append((tok_start, tok_end, "NOTE",
                            f"{prev_label}→{label} (H={prev['H']:.2f}→{r['H']:.2f}) — "
                            "entropy rising, model losing confidence"))
        else:
            # Same-phase sharp movement.
            # Prefer 2-window cumulative H change (has_prev2) over single ΔH:
            # one outlier window (punctuation, markdown header, special token) can
            # look like a drop/spike; two consecutive windows in the same phase
            # with a consistent trend are much more reliable.
            h_ref      = records[i - 2]['H'] if has_prev2 else prev['H']
            cum_delta  = r['H'] - h_ref
            span       = "2w" if has_prev2 else "1w"
            sharp_drop = (has_prev2 and cum_delta < -0.55) or (not has_prev2 and delta_h < -0.6)
            sharp_rise = (has_prev2 and cum_delta >  1.00) or (not has_prev2 and delta_h > 1.2)
            if sharp_drop:
                if label == "CHAOS":
                    signals.append((tok_start, tok_end, "NOTE",
                        f"CHAOS easing (H={h_ref:.2f}→{r['H']:.2f} /{span}) — model settling"))
                elif label == "COLLAPSING":
                    signals.append((tok_start, tok_end, "WARN",
                        f"Fast collapse (H={h_ref:.2f}→{r['H']:.2f} /{span}) — loop forming"))
                else:
                    signals.append((tok_start, tok_end, "NOTE",
                        f"H dropping (H={h_ref:.2f}→{r['H']:.2f} /{span}) — confidence building"))
            elif sharp_rise:
                signals.append((tok_start, tok_end, "NOTE",
                    f"H rising (H={h_ref:.2f}→{r['H']:.2f} /{span}) — uncertainty burst"))

        wjk      = r.get('win_jaccard',    float('nan'))
        prev_wjk = prev.get('win_jaccard', float('nan'))
        next_wjk = records[i + 1].get('win_jaccard', float('nan')) if has_next else float('nan')
        # Require the crossing to persist into the next window (not a one-off spike)
        if (not math.isnan(wjk) and not math.isnan(prev_wjk) and prev_wjk < 0.45 <= wjk
                and (math.isnan(next_wjk) or next_wjk >= 0.38)):
            signals.append((tok_start, tok_end, "WARN",
                f"win-Jaccard crossed 0.45 ({prev_wjk:.2f}→{wjk:.2f}) — "
                "same candidate-set repeating window-to-window"))
        # Downward crossing: loop may be easing (require real drop, not just boundary noise)
        elif (not math.isnan(wjk) and not math.isnan(prev_wjk) and prev_wjk >= 0.45 > wjk
                and wjk < 0.38
                and (math.isnan(next_wjk) or next_wjk < 0.45)):
            signals.append((tok_start, tok_end, "NOTE",
                f"win-Jaccard dropping ({prev_wjk:.2f}→{wjk:.2f}) — "
                "cycling candidate-set diversifying; loop may be easing"))

        if delta_vh > 3.5 and r['VH'] > 5.5:
            signals.append((tok_start, tok_end, "NOTE",
                f"VH burst ΔVH={delta_vh:+.2f} (now {r['VH']:.2f}) — "
                "heavy tail noise spike; check quantisation"))

    # ── Sustained top-1 confidence runs ─────────────────────────────────────
    # Single-window threshold crossings are too noisy (one markdown header,
    # a proper noun, or a punctuation token can spike top-1 briefly).
    # Signal only when the model holds high confidence for multiple windows.
    top1s = [r['top1'] for r in records]
    for thresh, min_run, level, msg in (
            (0.95, 2, "WARN", "near-deterministic — loop likely"),
            (0.88, 3, "NOTE", "sustained overconfidence — repetition risk increasing"),
    ):
        idx = 0
        while idx < len(records):
            if top1s[idx] >= thresh:
                j = idx
                while j < len(records) and top1s[j] >= thresh:
                    j += 1
                run_len = j - idx
                if run_len >= min_run:
                    tok_s = records[idx - 1]['tok'] + 1 if idx > 0 else 1
                    tok_e = records[j - 1]['tok']
                    peak  = max(top1s[idx:j])
                    signals.append((tok_s, tok_e, level,
                        f"top-1 ≥ {thresh:.0%} for {run_len} consecutive windows "
                        f"(peak={peak:.2f}) — {msg}"))
                idx = j
            else:
                idx += 1

    # ── Forward-looking prediction (tail-slope OLS) ─────────────────────────
    n_tail   = min(5, len(records))
    tail     = records[-n_tail:]
    h_tail   = [r['H']    for r in tail]
    t1_tail  = [r['top1'] for r in tail]
    tok_tail = [r['tok']  for r in tail]
    mean_tt  = sum(tok_tail) / n_tail
    denom_p  = sum((t - mean_tt) ** 2 for t in tok_tail) or 1
    sh = sum((t - mean_tt) * (h - sum(h_tail) / n_tail)
             for t, h in zip(tok_tail, h_tail)) / denom_p
    st = sum((t - mean_tt) * (p - sum(t1_tail) / n_tail)
             for t, p in zip(tok_tail, t1_tail)) / denom_p

    predictions = []
    cur_h = tail[-1]['H']
    if sh < -0.020 and st > 0.003:
        if cur_h > 0.20:
            eta = max(10, int((0.20 - cur_h) / sh))
            predictions.append(
                f"[PRED] Tail trend: H slope={sh:+.4f}/tok, top-1 slope={st:+.4f}/tok — "
                f"LOCK threshold (H=0.20) reached in ~{eta} more tokens; "
                "add repetition_penalty now")
        else:
            predictions.append(
                f"[PRED] H already ≤0.20 and still falling (slope={sh:+.4f}/tok) — "
                "model fully locked into repetition")
    elif sh > 0.030:
        predictions.append(
            f"[PRED] Tail entropy rising (slope={sh:+.4f}/tok) — "
            "model becoming increasingly uncertain; hallucination risk ahead")
    elif abs(sh) < 0.005 and cur_h < 0.40:
        predictions.append(
            f"[PRED] H stable and low (H={cur_h:.2f}, slope={sh:+.4f}/tok) — "
            "model in sustained over-confident state; loop may already be active")

    # ── Print ─────────────────────────────────────────────────────────────
    print("  ── Early warning signals ─────────────────────────────────────")
    if signals:
        MARKER = {"WARN": "!", "NOTE": "~", "INFO": " "}
        for tok_s, tok_e, level, msg in signals:
            print(f"    [{MARKER[level]}] tok {tok_s:>4}–{tok_e:<4}  {msg}")
    else:
        print("    (no notable phase transitions or threshold crossings)")
    if predictions:
        print()
        for p in predictions:
            print(f"    {p}")
    print()


# ── Segment label helper ─────────────────────────────────────────────────────
# Used by the regional timeline inside print_summary.  Classifies a short
# window of records (typically 5) using only LOCAL statistics so late-onset
# collapse is visible without global-average dilution.

_SEG_BAD  = {"LOCK", "COLLAPSING", "CYCLING"}
_SEG_WARN = {"CHAOS", "HIGH-H", "MIXED"}
_SEG_GOOD = {"NORMAL", "HIGH-CONF"}

def _seg_label(chunk):
    """Return a short string label for a segment of records."""
    n      = len(chunk)
    phases = [zone_for(r['H'])[0] for r in chunk]
    f_lock = phases.count("LOCK")        / n
    f_col  = phases.count("COLLAPSING")  / n
    f_cha  = phases.count("CHAOS")       / n
    f_hih  = phases.count("HIGH-H")      / n
    f_nor  = phases.count("NORMAL")      / n
    wjks   = [r.get('win_jaccard', float('nan')) for r in chunk]
    wjk_v  = [w for w in wjks if not math.isnan(w)]
    wjk    = statistics.mean(wjk_v) if wjk_v else float('nan')
    if f_lock >= 0.4:
        return "LOCK"
    if f_lock + f_col >= 0.6 and not math.isnan(wjk) and wjk > 0.35:
        return "CYCLING"
    if f_lock + f_col >= 0.6 and (math.isnan(wjk) or wjk < 0.28):
        # Tight distribution but low win-win Jaccard: model is confident but the
        # candidate token sets change window-to-window — typical of factual content,
        # not a cycling phrase attractor.  Label as HIGH-CONF, not a loop.
        return "HIGH-CONF"
    if f_lock + f_col >= 0.6:
        return "COLLAPSING"   # wjk in 0.28-0.35: ambiguous, treat conservatively
    if f_cha >= 0.4:
        return "CHAOS"
    if f_hih >= 0.5:
        return "HIGH-H"
    if f_nor >= 0.5:
        return "NORMAL"
    return "MIXED"


def print_summary(records, title=""):
    if not records:
        print("No records — nothing to summarise.")
        return

    # Compute per-window deltas (in-place augmentation; idempotent)
    for i, r in enumerate(records):
        prev = records[i - 1] if i > 0 else None
        r['delta_H']   = r['H']    - prev['H']    if prev else 0.0
        r['delta_top1']= r['top1'] - prev['top1'] if prev else 0.0
        r['delta_VH']  = r['VH']   - prev['VH']   if prev else 0.0

    toks = [r['tok']    for r in records]
    hs   = [r['H']      for r in records]
    vhs  = [r['VH']     for r in records]
    t1s  = [r['top1']   for r in records]
    evs  = [r['eff_v']  for r in records]
    mgs  = [r['margin'] for r in records]

    def row(name, vals, fmt=".3f"):
        mn, med, mx = min(vals), statistics.median(vals), max(vals)
        mean = statistics.mean(vals)
        print(f"  {name:<22} mean={mean:{fmt}}  median={med:{fmt}}  "
              f"min={mn:{fmt}}  max={mx:{fmt}}")

    print("=" * 70)
    if title:
        print(f"  LOGITS MONITORING SUMMARY  —  {title}")
        print("=" * 70)
    print(f"  Windows logged : {len(records)}  (tokens {toks[0]} – {toks[-1]})")
    print()

    print("  ── Distribution metrics ──────────────────────────────────────")
    row("entropy  (H)",     hs)
    row("varentropy (VH)",  vhs)
    row("top-1 prob",       t1s)
    row("eff vocab",        evs, fmt=".1f")
    row("top1/top2 margin", mgs)
    jks = [r['jaccard'] for r in records if not math.isnan(r['jaccard'])]
    if jks:
        row("jaccard top-20",  jks)
    print()

    print("  ── Phase breakdown ───────────────────────────────────────────")
    phase_recs = {}
    for r in records:
        lbl, _ = zone_for(r['H'])
        phase_recs.setdefault(lbl, []).append(r)
    for _, lbl, _ in ZONES:
        if lbl not in phase_recs:
            continue
        pr   = phase_recs[lbl]
        frac = len(pr) / len(records)
        mh   = statistics.mean(r['H'] for r in pr)
        print(f"    {lbl:<12} {len(pr):4d} windows ({frac*100:5.1f}%)  "
              f"H={mh:.3f}  first@tok={pr[0]['tok']}  last@tok={pr[-1]['tok']}")
    print()

    temporal_dynamics(records)
    early_warning_signals(records)

    frac_chaos       = sum(1 for h in hs if h > 3.0)        / len(hs)
    frac_lock        = sum(1 for h in hs if h < 0.15)       / len(hs)
    frac_collapsing  = sum(1 for h in hs if 0.15 <= h < 0.5) / len(hs)
    frac_norm        = sum(1 for h in hs if 0.5 <= h <= 1.5) / len(hs)
    mean_h, med_h    = statistics.mean(hs), statistics.median(hs)
    mean_top1        = statistics.mean(t1s)

    # Entropy and top-1 slopes (OLS)
    toks_list = [r['tok'] for r in records]
    n = len(records)
    mean_tok = sum(toks_list) / n
    denom_t  = sum((t - mean_tok) ** 2 for t in toks_list) or 1
    h_slope  = sum((t - mean_tok) * (h - mean_h) for t, h in zip(toks_list, hs)) / denom_t
    t1_slope = sum((t - mean_tok) * (p - mean_top1)
                   for t, p in zip(toks_list, t1s)) / denom_t

    # H autocorrelation at lag 1 (also displayed in temporal_dynamics, used in scorecard)
    h_var = (sum((h - mean_h) ** 2 for h in hs) / n) if n > 1 else 0
    if h_var > 1e-9 and n >= 3:
        h_autocorr = (sum((hs[i] - mean_h) * (hs[i+1] - mean_h) for i in range(n-1))
                      / ((n - 1) * h_var))
    else:
        h_autocorr = float('nan')

    # Detect monotonically descending phase sequence (each run lower entropy than previous)
    phases_seq = [zone_for(h)[0] for h in hs]
    runs_seq: list = []
    for p in phases_seq:
        if runs_seq and runs_seq[-1][0] == p:
            runs_seq[-1][1] += 1
        else:
            runs_seq.append([p, 1])
    PHASE_ORDER_LOCAL = PHASE_ORDER  # already module-level
    run_ranks = [PHASE_ORDER_LOCAL.get(p, 2) for p, _ in runs_seq]
    _monotone_descent = (len(run_ranks) >= 3
                         and all(run_ranks[i] >= run_ranks[i+1]
                                 for i in range(len(run_ranks)-1))
                         and run_ranks[0] > run_ranks[-1])

    # Pre-compute isolated CHAOS count (used in both scorecard and pattern interpretation)
    _isolated_chaos_count = sum(
        1 for i in range(1, len(records) - 1)
        if (zone_for(records[i]['H'])[0] == "CHAOS"
            and zone_for(records[i-1]['H'])[0] != "CHAOS"
            and zone_for(records[i+1]['H'])[0] != "CHAOS"))

    checks = []

    if med_h < 0.2:
        checks.append(("WARN", f"Median entropy {med_h:.3f} very low — likely persistent lock/loop"))
    elif med_h > 3.0:
        checks.append(("WARN", f"Median entropy {med_h:.3f} extremely high — model may be broken "
                               "(wrong precision, hardware error, wrong model format)"))
    else:
        checks.append(("OK",   f"Median entropy {med_h:.3f} (mean={mean_h:.3f}) in expected range"))

    # Detect longest consecutive CHAOS streak and whether it's front-loaded
    phases_list_ck = [zone_for(h)[0] for h in hs]
    chaos_streak = max_streak = cur_streak = 0
    chaos_streak_start_frac = 0.0
    cur_start = 0
    for i, p in enumerate(phases_list_ck):
        if p == "CHAOS":
            cur_streak += 1
            if cur_streak > max_streak:
                max_streak = cur_streak
                chaos_streak_start_frac = cur_start / len(phases_list_ck)
        else:
            cur_streak = 0
            cur_start = i + 1
    chaos_streak = max_streak
    _sustained_chaos_start = (chaos_streak >= 5 and chaos_streak_start_frac < 0.2)

    if frac_chaos > 0.5:
        checks.append(("WARN", f"{frac_chaos*100:.0f}% CHAOS — model cannot generate coherently; "
                               "check format, precision, hardware"))
    elif _sustained_chaos_start:
        checks.append(("WARN",
            f"Sustained CHAOS at start ({chaos_streak} consecutive windows) — model could not "
            "find coherent footing on this prompt; conflicting training signal or "
            "sensitive/ambiguous topic"))
    elif frac_chaos > 0.1:
        checks.append(("NOTE", f"{frac_chaos*100:.0f}% CHAOS — extended uncertainty phases"))
    else:
        checks.append(("OK",   f"Chaos fraction {frac_chaos*100:.1f}% — no runaway chaos"))

    if not math.isnan(h_autocorr) and h_autocorr < 0.15 and _isolated_chaos_count >= 3:
        checks.append(("WARN",
            f"H autocorr={h_autocorr:.3f} (near zero) + {_isolated_chaos_count} isolated "
            "CHAOS spikes — model is erratic/fragmented; each window nearly independent. "
            "Prompt may be ambiguous, conflicting, or beyond model's knowledge"))
    elif not math.isnan(h_autocorr) and h_autocorr < 0.15 and frac_norm < 0.55:
        # Only flag when there is genuine instability — if NORMAL dominates (≥55%)
        # the low autocorr is just structured document rhythm (headers/bullets), not erratic.
        checks.append(("NOTE",
            f"H autocorr={h_autocorr:.3f} — phase switching with low persistence; "
            "generation lacks a stable state (conflicting signal or ambiguous prompt)"))

    if frac_lock > 0.35:
        checks.append(("WARN", f"{frac_lock*100:.0f}% LOCK — single-token repetition loop; "
                               "try repetition penalty or higher temperature"))
    elif frac_lock > 0.1:
        checks.append(("NOTE", f"{frac_lock*100:.0f}% LOCK — some confident/repetitive phases"))
    else:
        checks.append(("OK",   f"Lock fraction {frac_lock*100:.1f}% — no persistent repetition"))

    # Cycling repetition loop: model cycles a short phrase confidently.
    # Signature: COLLAPSING dominates + high mean top-1 + low token-token Jaccard.
    # Win-win Jaccard is the strongest signal: if the window-to-window Jaccard
    # is HIGH (same top-20 recurring across entire windows), the model is looping.
    wjks = [r.get('win_jaccard', float('nan')) for r in records]
    wjks_valid = [w for w in wjks if not math.isnan(w)]
    mean_wjk = statistics.mean(wjks_valid) if wjks_valid else float('nan')
    _win_cycling = (not math.isnan(mean_wjk) and mean_wjk > 0.50)
    # _tok_cycling: COLLAPSING + high confidence + same-slot diversity (low jk) + sustained
    # win-win Jaccard AND falling entropy.  Low jk alone isn't enough — a well-trained model
    # generating correct factual text also shows low jk but with flat entropy slope and low wjk.
    _h_slope_rel_quick = h_slope / max(mean_h, 0.20)   # pre-compute for gate below
    _tok_cycling = (jks and frac_collapsing >= 0.35
                    and mean_top1 > 0.72
                    and statistics.mean(jks) < 0.35
                    and not math.isnan(mean_wjk) and mean_wjk > 0.30
                    and _h_slope_rel_quick < -0.008)    # require entropy actually falling
    _cycling = _win_cycling or _tok_cycling

    # Converging (early warning): entropy falling, confidence rising,
    # phases trending toward LOCK — loop is forming, not yet fully established.
    # Accept high LOCK fraction as evidence (model may skip COLLAPSING and hit LOCK directly).
    #
    # Slope thresholds are NORMALIZED by the run's own baseline so the same
    # criterion applies regardless of model size or temperature:
    #   h_slope_rel  = h_slope  / mean_h          → fraction of mean H lost per token
    #   t1_slope_rel = t1_slope / (1 - mean_top1) → fraction of remaining headroom gained per token
    # Fixed absolute slopes (e.g. -0.010) would fire too easily for high-H runs
    # and miss convergence for already-tight (low-H) runs.
    _h_baseline  = max(mean_h,         0.20)   # guard against /0 for near-lock runs
    _t1_headroom = max(1.0 - mean_top1, 0.05)  # guard against /0 for near-deterministic runs
    _h_slope_rel  = h_slope  / _h_baseline      # < 0 means entropy shrinking
    _t1_slope_rel = t1_slope / _t1_headroom     # > 0 means headroom closing
    _converging = (not _cycling
                   and _h_slope_rel  < -0.015   # entropy falling > 1.5% of mean per token
                   and _t1_slope_rel >  0.008   # confidence rising > 0.8% of headroom per token
                   and (_monotone_descent or frac_collapsing >= 0.25 or frac_lock >= 0.25))

    # Oscillating loop: second half alternates between high-confidence
    # (COLLAPSING/LOCK) and lower-confidence (NORMAL/HIGH-H) windows.
    n_half = len(records) // 2
    second_phases = [zone_for(records[i]['H'])[0] for i in range(n_half, len(records))]
    _high_conf = {"COLLAPSING", "LOCK"}
    _col_to_norm = sum(1 for i in range(len(second_phases) - 1)
                       if second_phases[i]     in _high_conf
                       and second_phases[i+1] not in _high_conf)
    _norm_to_col = sum(1 for i in range(len(second_phases) - 1)
                       if second_phases[i]    not in _high_conf
                       and second_phases[i+1]  in _high_conf)
    _oscillating = (not _cycling
                    and _col_to_norm >= 2 and _norm_to_col >= 2
                    and frac_lock + frac_collapsing >= 0.20
                    and (not math.isnan(mean_wjk) and mean_wjk > 0.25 or h_autocorr < 0.0))
    # Low win-win Jaccard means diverse content each window (structured document, not looping).
    # Negative H autocorr means the model is bouncing to the same attractor, which is also
    # a valid oscillating-loop signal even when win-win Jaccard is moderate.

    # Late-onset collapse: model starts healthy then falls into LOCK/COLLAPSING in second half.
    # Global averages look fine because the healthy first half inflates mean H and deflates
    # frac_lock — this pattern is invisible to run-wide statistics.
    mean_h_first  = statistics.mean(r['H'] for r in records[:n_half])  if n_half > 0            else mean_h
    mean_h_second = statistics.mean(r['H'] for r in records[n_half:])  if n_half < len(records) else mean_h
    frac_tight_2nd = (sum(1 for r in records[n_half:] if r['H'] < 0.5)
                      / max(len(records) - n_half, 1))
    _late_onset_collapse = (not _cycling and not _converging
                             and 0.5 < mean_h_first < 3.5   # started in NORMAL/HIGH-H (not CHAOS)
                             and mean_h_second < 0.5 * mean_h_first  # second half < 50% of first
                             and frac_tight_2nd > 0.40)    # 40%+ COLLAPSING/LOCK in second half

    if _cycling:
        detail_parts = [f"COLLAPSING {frac_collapsing*100:.0f}%",
                        f"top-1={mean_top1:.2f}"]
        if jks:
            detail_parts.append(f"tok-Jk={statistics.mean(jks):.3f}")
        if not math.isnan(mean_wjk):
            detail_parts.append(f"win-Jk={mean_wjk:.3f}")
        checks.append(("WARN",
            "  +  ".join(detail_parts) + " — cycling repetition attractor; "
            "add repetition_penalty or higher temperature"))
    elif _oscillating:
        checks.append(("WARN",
            f"Oscillating LOCK↔NORMAL in second half — "
            f"{_col_to_norm} high→low and {_norm_to_col} low→high transitions; "
            "model commits then reconsiders, never escaping the same attractor"))
    elif _converging:
        checks.append(("WARN",
            f"H slope={h_slope:+.4f}/tok  top-1 slope={t1_slope:+.4f}/tok  "
            f"LOCK={frac_lock*100:.0f}%  COLLAPSING={frac_collapsing*100:.0f}% "
            "\u2014 converging toward repetition loop; "
            "add repetition_penalty now or expect loop to form"))
    elif _late_onset_collapse:
        checks.append(("WARN",
            f"First-half H={mean_h_first:.2f}  second-half H={mean_h_second:.2f} "
            f"({mean_h_second / mean_h_first * 100:.0f}% of start)  "
            f"tight-2nd={frac_tight_2nd*100:.0f}% \u2014 "
            "model started healthy then collapsed into a repetition loop in second half"))
    elif frac_collapsing > 0.5:
        checks.append(("NOTE",
            f"COLLAPSING dominates ({frac_collapsing*100:.0f}%) — model is over-confident; "
            "output may be repetitive or formulaic"))

    mean_vh_h = statistics.mean(
        r['VH'] / r['H'] if r['H'] > 0.1 else 0 for r in records)
    if mean_vh_h > 6.0:
        checks.append(("WARN", f"VH/H ratio {mean_vh_h:.1f} — very high tail noise; "
                               "likely INT4/INT8 quantisation artefact or hardware bug"))
    elif mean_vh_h > 3.0:
        checks.append(("NOTE", f"VH/H ratio {mean_vh_h:.1f} — elevated tail noise"))
    else:
        checks.append(("OK",   f"VH/H ratio {mean_vh_h:.1f} — tail noise within expected range"))

    max_std = max(r['logit_std'] for r in records)
    if max_std > 30:
        checks.append(("WARN", f"Max logit_std={max_std:.1f} — possible NaN/Inf in logits "
                               "(hardware or wrong dtype)"))
    else:
        checks.append(("OK",   f"Max logit_std={max_std:.1f} — no overflow signs"))

    if jks:
        mean_jk = statistics.mean(jks)
        pre_lock_high = sum(
            1 for r in records
            if not math.isnan(r['jaccard']) and r['jaccard'] > 0.7 and r['H'] > 0.15)
        if mean_jk > 0.75:
            checks.append(("WARN", f"Mean Jaccard {mean_jk:.3f} — distribution barely moves; "
                                   "persistent repetition or trivial completion"))
        elif pre_lock_high >= 3:
            checks.append(("NOTE", f"{pre_lock_high} pre-lock windows with Jaccard>0.7 — "
                                   "attractor formation detectable before entropy collapse"))
        elif _cycling:
            # Suppress misleading "varies normally" when cycling is already flagged above.
            checks.append(("NOTE", f"Jaccard top-20 mean={mean_jk:.3f} — low because each cycle "
                                   "position has a different top-20 (not genuine diversity)"))
        else:
            checks.append(("OK",   f"Jaccard top-20 mean={mean_jk:.3f} — top-K set varies normally"))

    print("  ── Health scorecard ──────────────────────────────────────────")
    for tag, msg in checks:
        marker = {"WARN": "x", "NOTE": "o", "OK": "v"}[tag]
        print(f"    [{tag:<4}] {marker} {msg}")
    print()

    # EXPLORING: model genuinely uncertain throughout, no loop or lock.
    # Characterised by HIGH-H dominant, low Jaccard (diverse vocab), and
    # scattered (not sustained) CHAOS spikes.
    frac_high_h = sum(1 for h in hs if 1.5 <= h <= 3.5) / len(hs)
    _exploring = (not _cycling and not _converging
                  and frac_high_h > 0.40
                  and frac_collapsing < 0.30
                  and frac_lock == 0)

    chaos_recs = [r for r in records if r['H'] > 3.0]
    lock_recs  = [r for r in records if r['H'] < 0.15]
    chaos_then_lock = (chaos_recs and lock_recs and
                       chaos_recs[-1]['tok'] < lock_recs[0]['tok'])
    # Cycling: chaos at start (model searching) then collapses into a short phrase loop
    _chaos_then_cycling = (chaos_recs and not lock_recs and frac_collapsing >= 0.35
                           and chaos_recs[-1]['tok'] < records[len(records)//2]['tok'])
    print("  ── Regional timeline ─────────────────────────────────────────")
    win = records[1]['tok'] - records[0]['tok'] if len(records) > 1 else 10
    seg_size = max(3, min(5, len(records) // 4))
    segs = []
    for seg_start in range(0, len(records), seg_size):
        chunk = records[seg_start:min(seg_start + seg_size, len(records))]
        if not chunk:
            continue
        tok_s    = (records[seg_start - 1]['tok'] + 1) if seg_start > 0 else 1
        tok_e    = chunk[-1]['tok']
        seg_h    = statistics.mean(r['H'] for r in chunk)
        phases   = [zone_for(r['H'])[0] for r in chunk]
        # Compact phase string: "NOR×4  COL×1"
        seen = {}
        for p in phases:
            seen[p] = seen.get(p, 0) + 1
        phase_str = "  ".join(
            (f"{SHORT_PHASE.get(p,p[:3])}×{c}" if c > 1 else SHORT_PHASE.get(p, p[:3]))
            for p, c in sorted(seen.items(), key=lambda kv: -kv[1])
        )
        wjk_v  = [r.get('win_jaccard', float('nan')) for r in chunk
                  if not math.isnan(r.get('win_jaccard', float('nan')))]
        wjk_s  = f"wjk={statistics.mean(wjk_v):.2f}" if wjk_v else "wjk=--  "
        label  = _seg_label(chunk)
        icon   = "[!]" if label in _SEG_BAD else ("[~]" if label in _SEG_WARN else "[ ]")
        segs.append({'tok_s': tok_s, 'tok_e': tok_e, 'label': label, 'seg_h': seg_h})
        print(f"  {icon} tok {tok_s:>3}–{tok_e:<3}  H={seg_h:.2f}  "
              f"{phase_str:<17}  {wjk_s}  → {label}")

    # Overall trend derived from segment sequence (no global-average dilution)
    labels      = [s['label'] for s in segs]
    n_s         = len(labels)
    n_bad_first = sum(1 for l in labels[:n_s // 2] if l in _SEG_BAD)
    n_bad_last  = sum(1 for l in labels[n_s // 2:] if l in _SEG_BAD)
    n_last      = max(n_s - n_s // 2, 1)
    print()
    if n_s == 0:
        pass
    elif all(l in _SEG_BAD for l in labels):
        print("  Overall: SUSTAINED LOOP — model was in repetition state throughout")
    elif n_bad_first == 0 and n_bad_last >= (n_last + 1) // 2:
        first_bad = next((s['tok_s'] for s in segs if s['label'] in _SEG_BAD), '?')
        print(f"  Overall: LATE-ONSET COLLAPSE — healthy start, fell into "
              f"{labels[-1]} from ~tok {first_bad}")
    elif n_bad_last > n_bad_first:
        print(f"  Overall: DETERIORATING — more structural issues in second half  "
              f"[{'  '.join(labels)}]")
    elif n_bad_first > 0 and n_bad_last == 0:
        print(f"  Overall: RECOVERED — issues in first half resolved  "
              f"[{'  '.join(labels)}]")
    elif all(l in {"NORMAL", "HIGH-H", "MIXED", "HIGH-CONF"} for l in labels):
        if any(l == "HIGH-CONF" for l in labels):
            print("  Overall: CLEAN — high-confidence generation, no loop evidence")
        else:
            print("  Overall: CLEAN — no structural anomalies detected")
    else:
        print(f"  Overall: MIXED  [{'  '.join(labels)}]")
    print("=" * 70)

# ============================================================================
# Plot  (6-panel matplotlib dashboard)
# ============================================================================

def make_plot(records, title="", out_path=None):
    try:
        import matplotlib
        if out_path and out_path != "none":
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec
        import numpy as np
    except ImportError:
        print("(matplotlib not available — skipping plot; pip install matplotlib)",
              file=sys.stderr)
        return

    toks     = np.array([r['tok']      for r in records])
    hs       = np.array([r['H']        for r in records])
    vhs      = np.array([r['VH']       for r in records])
    t1s      = np.array([r['top1']     for r in records])
    evs      = np.array([r['eff_v']    for r in records])
    mgs      = np.array([r['margin']   for r in records])
    jks_raw  = np.array([r['jaccard']              for r in records])
    wjks_raw = np.array([r.get('win_jaccard', float('nan')) for r in records])
    has_jk   = not np.all(np.isnan(jks_raw))
    has_wjk  = not np.all(np.isnan(wjks_raw))

    ema_fast = np.array(ema(hs.tolist(), 0.30))
    ema_slow = np.array(ema(hs.tolist(), 0.10))
    segs     = segments(records)

    fig = plt.figure(figsize=(16, 14), constrained_layout=True)
    fig.suptitle(f"Logits Dashboard — {title}", fontsize=13, fontweight="bold")
    gs = GridSpec(4, 2, figure=fig, height_ratios=[2.2, 1.4, 1.4, 1.4])

    ax_main  = fig.add_subplot(gs[0, :])
    ax_phase = fig.add_subplot(gs[1, 0])
    ax_ev    = fig.add_subplot(gs[1, 1])
    ax_t1    = fig.add_subplot(gs[2, 0])
    ax_vhdtl = fig.add_subplot(gs[2, 1])
    ax_jk    = fig.add_subplot(gs[3, :])

    # ── Entropy timeline ────────────────────────────────────────────────
    hi_top = max(float(hs.max()) * 1.1, 3.5)
    zone_bounds = [(0, 0.15), (0.15, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, hi_top)]
    for (lo, hi), (_, _, col) in zip(zone_bounds, ZONES):
        ax_main.axhspan(lo, hi, alpha=0.08, color=col, linewidth=0)
    for start, end, _, col in segs:
        mask = (toks >= start) & (toks <= end)
        ax_main.fill_between(toks[mask], hs[mask], alpha=0.25, color=col, linewidth=0)
    ax_main.scatter(toks, hs, s=18, c=[zone_for(h)[1] for h in hs], zorder=3, linewidths=0)
    ax_main.plot(toks, ema_fast, lw=1.5, color="#333", label="EMA a=0.30 (fast)", zorder=4)
    ax_main.plot(toks, ema_slow, lw=1.5, color="#888", ls="--", label="EMA a=0.10 (slow)", zorder=4)
    zone_labels = [l for _, l, _ in ZONES]
    zone_mids   = [0.075, 0.325, 1.0, 2.25, 3.5]
    ylim_top = max(float(hs.max()) * 1.1, 3.5)
    ax_r = ax_main.twinx()
    ax_r.set_ylim(0, ylim_top)
    ax_main.set_ylim(0, ylim_top)
    visible_mids   = [m for m in zone_mids if m < ylim_top]
    visible_labels = [zone_labels[i] for i, m in enumerate(zone_mids) if m < ylim_top]
    ax_r.set_yticks(visible_mids)
    ax_r.set_yticklabels(visible_labels, fontsize=8, color="#555")
    ax_r.tick_params(length=0)
    ax_main.set_title("Entropy H(t) — per-window with EMA tracks", fontsize=10)
    ax_main.set_xlabel("Token")
    ax_main.set_ylabel("Entropy (nats)")
    ax_main.legend(fontsize=8, loc="upper right")
    ax_main.set_xlim(toks[0], toks[-1])

    # ── Phase space H vs VH ─────────────────────────────────────────────
    colours_by_time = plt.cm.plasma(np.linspace(0.1, 0.9, len(toks)))
    ax_phase.scatter(hs, vhs, c=colours_by_time, s=22, alpha=0.7, linewidths=0)
    mh, mvh = float(hs.max()) * 1.05 / 2, float(vhs.max()) * 1.05 / 2
    for x, y, text, col in [(mh*0.1, mvh*1.7, "noisy-tail", "#888"),
                              (mh*1.5, mvh*1.7, "CHAOS",      "#9467bd"),
                              (mh*0.1, mvh*0.3, "LOCK",       "#d62728"),
                              (mh*1.5, mvh*0.3, "exploration","#2ca02c")]:
        ax_phase.text(x, y, text, fontsize=7, color=col, ha="center", style="italic")
    ax_phase.axvline(mh, color="#ccc", lw=0.8, ls=":")
    ax_phase.axhline(mvh, color="#ccc", lw=0.8, ls=":")
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(toks[0], toks[-1]))
    sm.set_array([])
    fig.colorbar(sm, ax=ax_phase, fraction=0.04, pad=0.02).set_label("token", fontsize=7)
    ax_phase.set_xlabel("Entropy H")
    ax_phase.set_ylabel("Varentropy VH")
    ax_phase.set_title("Phase space H vs VH (colour = time)", fontsize=9)

    # ── Effective vocab + margin ─────────────────────────────────────────
    ax_ev.semilogy(toks, evs, color="#1f77b4", lw=1.2, label="eff vocab")
    ax_ev.axhline(1.0, color="#aaa", lw=0.8, ls=":")
    ax_ev.axhline(10,  color="#aaa", lw=0.8, ls=":")
    ax_ev2 = ax_ev.twinx()
    ax_ev2.plot(toks, mgs, color="#ff7f0e", lw=1.0, alpha=0.7)
    ax_ev2.set_ylabel("margin (log)", color="#ff7f0e", fontsize=8)
    ax_ev2.tick_params(colors="#ff7f0e", labelsize=7)
    ax_ev.set_ylabel("eff vocab (log)", fontsize=8)
    ax_ev.set_title("Effective vocab size + margin", fontsize=9)
    ax_ev.set_xlabel("Token")

    # ── Top-1 probability ───────────────────────────────────────────────
    ax_t1.plot(toks, t1s, color="#2ca02c", lw=1.2)
    ax_t1.axhline(0.9, color="#d62728", lw=0.8, ls="--", label="p=0.9")
    ax_t1.axhline(0.5, color="#ff7f0e", lw=0.8, ls="--", label="p=0.5")
    ax_t1.fill_between(toks, t1s, alpha=0.15, color="#2ca02c")
    ax_t1.set_ylim(0, 1.05)
    ax_t1.set_ylabel("top-1 probability")
    ax_t1.set_title("Top-1 probability", fontsize=9)
    ax_t1.set_xlabel("Token")
    ax_t1.legend(fontsize=7)

    # ── Varentropy detail ───────────────────────────────────────────────
    ax_vhdtl.plot(toks, vhs, color="#9467bd", lw=1.0, label="VH")
    ax_vhdtl.plot(toks, ema(vhs.tolist(), 0.2), color="#333", lw=1.5, ls="--",
                  label="VH EMA a=0.2")
    ax_vhdtl.set_ylabel("Varentropy VH")
    ax_vhdtl.set_title("Varentropy detail", fontsize=9)
    ax_vhdtl.set_xlabel("Token")
    ax_vhdtl.legend(fontsize=7)

    # ── Jaccard top-20 (tok-tok and win-win) ────────────────────────────
    if has_jk or has_wjk:
        if has_jk:
            jk_v = np.where(np.isnan(jks_raw), np.nan, jks_raw)
            jk_e = np.array(ema(np.where(np.isnan(jk_v), 0.0, jk_v).tolist(), 0.25))
            ax_jk.scatter(toks, jk_v, s=14, c=[zone_for(h)[1] for h in hs],
                          zorder=3, alpha=0.6, linewidths=0, label="tok-tok (per window avg)")
            ax_jk.plot(toks, jk_e, color="#1f77b4", lw=1.6,
                       label="tok-tok EMA a=0.25", zorder=4)
        if has_wjk:
            wjk_v = np.where(np.isnan(wjks_raw), np.nan, wjks_raw)
            wjk_e = np.array(ema(np.where(np.isnan(wjk_v), 0.0, wjk_v).tolist(), 0.25))
            ax_jk.scatter(toks, wjk_v, s=28, marker="D", color="#d62728",
                          zorder=5, alpha=0.55, linewidths=0, label="win-win (cycling signal)")
            ax_jk.plot(toks, wjk_e, color="#d62728", lw=1.6, ls="--",
                       label="win-win EMA a=0.25", zorder=6)
        ax_jk.axhline(0.55, color="#d62728", lw=0.9, ls=":",
                      label="win-win cycle threshold 0.55")
        ax_jk.axhline(0.75, color="#ff7f0e", lw=0.9, ls="--", label="tok-tok stability 0.75")
        ax_jk.set_ylim(0, 1.05)
        ax_jk.set_ylabel("Jaccard top-20")
        ax_jk.set_title(
            "Jaccard top-20: tok-tok (avg within window) vs win-win (window boundary)\n"
            "High win-win = same tokens looping across windows (cycling attractor)",
            fontsize=9)
        ax_jk.legend(fontsize=7, loc="upper left")
    else:
        ax_jk.text(0.5, 0.5, "Jaccard top-20 not available in this log",
                   ha="center", va="center", transform=ax_jk.transAxes,
                   fontsize=10, color="#888")
        ax_jk.set_title("Top-20 Jaccard (not available)", fontsize=9)
    ax_jk.set_xlabel("Token")
    ax_jk.set_xlim(toks[0], toks[-1])

    # ── Zone legend ─────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(
            color=col, alpha=0.6,
            label=f"{lbl}  ({'H<'+str(ZONES[i][0]) if i < len(ZONES)-1 else 'H>'+str(ZONES[-2][0])})")
        for i, (_, lbl, col) in enumerate(ZONES)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8,
               title="Entropy zones", title_fontsize=8, framealpha=0.9)

    if out_path and out_path != "none":
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved -> {out_path}", file=sys.stderr)
    else:
        plt.show()

# ============================================================================
# Generation loop
# ============================================================================

# ANSI color codes for annotated output replay.
# NORMAL / HIGH-CONF: no color (default terminal color).
# Using foreground colors so they work on both light and dark terminals.
_ANNO_CLR = {
    "LOCK":       "\033[1;31m",   # bold red   — single-token repetition attractor
    "CYCLING":    "\033[1;31m",   # bold red   — phrase cycling loop
    "COLLAPSING": "\033[33m",     # yellow     — distribution tightening, loop risk
    "CHAOS":      "\033[35m",     # magenta    — runaway entropy spike
    "HIGH-H":     "\033[36m",     # cyan       — unusually high uncertainty
}
_ANNO_RESET = "\033[0m"


def _print_annotated(token_texts, records, window_size):
    """Re-print the generated output with per-window ANSI color annotations."""
    if not records or not token_texts:
        return
    print("\n── Annotated output ──────────────────────────────────────────────────")
    print(f"   {_ANNO_CLR['LOCK']}■ red{_ANNO_RESET}=LOCK/CYCLING  "
          f"{_ANNO_CLR['COLLAPSING']}■ yellow{_ANNO_RESET}=COLLAPSING (loop risk)  "
          f"{_ANNO_CLR['CHAOS']}■ magenta{_ANNO_RESET}=CHAOS  "
          f"{_ANNO_CLR['HIGH-H']}■ cyan{_ANNO_RESET}=HIGH-H  "
          f"□=NORMAL\n")
    for i, r in enumerate(records):
        zone  = zone_for(r['H'])[0]
        start = i * window_size
        end   = min(start + window_size, len(token_texts))
        chunk = "".join(token_texts[start:end])
        color = _ANNO_CLR.get(zone, "")
        if color:
            print(f"{color}{chunk}{_ANNO_RESET}", end="")
        else:
            print(chunk, end="")
    # Any tokens in a partial final window not yet flushed as a record
    tail_start = len(records) * window_size
    if tail_start < len(token_texts):
        print("".join(token_texts[tail_start:]), end="")
    print("\n")


# ============================================================================
# Generation backends
# ============================================================================

def run_generation_ovgenai(args):
    try:
        import openvino_genai as ov_genai
    except ImportError:
        sys.exit("openvino_genai not found — pip install openvino-genai")

    p = Path(args.prompt)
    if args.prompt.endswith(".txt") and p.exists():
        prompt = p.read_text()
        print(f"[prompt file: {args.prompt}, {len(prompt)} chars]", file=sys.stderr)
    else:
        prompt = args.prompt

    scheduler_cfg = ov_genai.SchedulerConfig()
    scheduler_cfg.max_num_batched_tokens = 256
    pipe      = ov_genai.ContinuousBatchingPipeline(args.model_dir, scheduler_cfg, args.device)
    tokenizer = pipe.get_tokenizer()

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.max_tokens
    config.return_logits  = True
    if args.no_sample or args.temperature == 0.0:
        config.do_sample = False
    else:
        config.do_sample   = True
        config.temperature = args.temperature
        config.top_k       = args.top_k
        config.top_p       = args.top_p

    # Tokenize prompt — or use pre-computed shared token IDs from --shared-tokenizer.
    shared_ids = getattr(args, 'input_ids', None)
    if shared_ids is not None:
        import numpy as _np
        import openvino as _ov
        prompt_ids = list(shared_ids)
        arr = _np.ascontiguousarray([prompt_ids], dtype=_np.int64)  # shape [1, N]
        try:
            # Preferred: wrap the numpy buffer directly (OV >= 2024.1)
            ids_tensor = _ov.Tensor(arr)
            _ = ids_tensor.data  # probe — raises if _impl is null
        except Exception:
            # Fallback: allocate with explicit Shape then fill
            ids_tensor = _ov.Tensor(_ov.Type.i64, _ov.Shape([1, len(prompt_ids)]))
            ids_tensor.data[:] = arr.flatten()
        print(f"[ovgenai  device={args.device}  max_tokens={args.max_tokens}  "
              f"return_logits=True  window={args.window}]", file=sys.stderr)
        print(f"[ovgenai] prompt = {len(prompt_ids)} tokens  (shared tokens)", file=sys.stderr)
        try:
            handle = pipe.add_request(0, ids_tensor, config)
        except RuntimeError as _te:
            print(f"[ovgenai] tensor path failed ({_te}), retrying with string prompt",
                  file=sys.stderr)
            handle = pipe.add_request(0, prompt, config)
    else:
        try:
            encoded_prompt = tokenizer.encode(prompt)
            ids = encoded_prompt.input_ids
            try:
                # OV Tensor → numpy via .data property (most reliable path)
                prompt_ids = ids.data.flatten().astype(int).tolist()
            except AttributeError:
                import numpy as _np
                prompt_ids = _np.array(ids).flatten().astype(int).tolist()
        except Exception:
            prompt_ids = []
        print(f"[ovgenai  device={args.device}  max_tokens={args.max_tokens}  "
              f"return_logits=True  window={args.window}]", file=sys.stderr)
        print(f"[ovgenai] prompt = {len(prompt_ids)} tokens", file=sys.stderr)
        handle = pipe.add_request(0, prompt, config)
    if getattr(args, 'debug_tokens', False) and prompt_ids:
        head = prompt_ids[:30]
        tail = prompt_ids[-10:] if len(prompt_ids) > 30 else []
        decoded_head = [tokenizer.decode([t]) for t in head]
        print(f"[ovgenai] first 30 token ids : {head}", file=sys.stderr)
        print(f"[ovgenai] first 30 decoded   : {decoded_head}", file=sys.stderr)
        if tail:
            print(f"[ovgenai] last  10 token ids : {tail}", file=sys.stderr)
    log    = LogWriter(args.log, window=args.window)
    prev_top20, token_count, t0 = [], 0, time.time()
    token_texts = []
    _debug_gen = getattr(args, 'debug_tokens', False)
    _gen_ids_log = []  # first 5 generated token IDs for --debug-tokens

    while handle.get_status() == ov_genai.GenerationStatus.RUNNING or handle.can_read():
        pipe.step()
        if not handle.can_read():
            continue
        for _seq_id, out in handle.read().items():
            for token_id in out.generated_ids:
                token_count += 1
                if _debug_gen and len(_gen_ids_log) < 5:
                    _dec = tokenizer.decode([token_id], skip_special_tokens=False)
                    _gen_ids_log.append((token_id, repr(_dec)))
                    if len(_gen_ids_log) == 5:
                        print(f"[ovgenai] first 5 generated ids: "
                              f"{[x[0] for x in _gen_ids_log]}", file=sys.stderr)
                        print(f"[ovgenai] first 5 generated tok: "
                              f"{[x[1] for x in _gen_ids_log]}", file=sys.stderr)
                decoded = tokenizer.decode([token_id], skip_special_tokens=False)
                token_texts.append(decoded)
                print(decoded, end="", flush=True)
                raw = list(out.generated_logits) if out.generated_logits else []
                if raw:
                    m     = compute_metrics(raw)
                    top_k = _top_k_indices(raw, k=100)
                    top20 = top_k[:20]
                    jk    = compute_jaccard(raw, prev_top20)
                    prev_top20 = top20
                    log.push(m, jk, top20=top20, top100=top_k)

    print()
    elapsed = time.time() - t0
    print(f"\n[{token_count} tokens in {elapsed:.1f}s = {token_count/elapsed:.1f} tok/s]",
          file=sys.stderr)
    log.close()
    print(f"[log -> {args.log}]", file=sys.stderr)
    _print_annotated(token_texts, log.records, args.window)
    return log.records, args.log, (log.token_top20s, log.token_top100s)


def run_generation_llamacpp(args):
    """llama.cpp backend via llama-cpp-python.

    Uses LogitsProcessorList (the official callback API) to capture raw logits
    at every generated token.  This is version-stable and avoids all ctypes
    pointer arithmetic that was causing the flat-distribution bug.
    """
    try:
        import llama_cpp
        from llama_cpp import LogitsProcessorList
        import numpy as np
    except ImportError:
        sys.exit("llama-cpp-python not found — pip install llama-cpp-python")

    model_path = Path(args.model_dir)
    if model_path.is_dir():
        gguf_files = sorted(model_path.glob("*.gguf"))
        if not gguf_files:
            sys.exit(f"No .gguf file found in {model_path}; pass the .gguf path directly")
        gguf_path = str(gguf_files[0])
        if len(gguf_files) > 1:
            print(f"[llamacpp] multiple .gguf found, using: {gguf_path}", file=sys.stderr)
    else:
        gguf_path = str(model_path)

    p = Path(args.prompt)
    if args.prompt.endswith(".txt") and p.exists():
        prompt = p.read_text()
        print(f"[prompt file: {args.prompt}, {len(prompt)} chars]", file=sys.stderr)
    else:
        prompt = args.prompt

    n_ctx        = getattr(args, 'n_ctx', 4096) or 4096
    n_gpu_layers = getattr(args, 'n_gpu_layers', 0) or 0
    n_threads    = getattr(args, 'n_threads', 0) or 0
    if n_threads == 0:
        import os
        n_threads = max(1, (os.cpu_count() or 4) // 2)  # physical-core estimate

    print(f"[llamacpp  model={gguf_path}  n_ctx={n_ctx}  n_gpu_layers={n_gpu_layers}"
          f"  n_threads={n_threads}  max_tokens={args.max_tokens}  window={args.window}]",
          file=sys.stderr)

    llm = llama_cpp.Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        logits_all=False,
        verbose=False,
    )

    try:
        eos_id = llm.token_eos()
    except AttributeError:
        import llama_cpp.llama_cpp as _lib
        eos_id = _lib.llama_token_eos(llm.model)

    # add_bos=False: Qwen3 / modern models set add_bos_token=false in tokenizer_config.
    # Pass --add-bos on the command line only for legacy LLaMA-1/2 models.
    shared_ids = getattr(args, 'input_ids', None)
    if shared_ids is not None:
        tokens = list(shared_ids)
        print(f"[llamacpp] prompt = {len(tokens)} tokens  (shared tokens)", file=sys.stderr)
    else:
        add_bos = getattr(args, 'add_bos', False)
        tokens = list(llm.tokenize(prompt.encode('utf-8'), add_bos=add_bos, special=True))
        print(f"[llamacpp] prompt = {len(tokens)} tokens  (add_bos={add_bos})", file=sys.stderr)
    if getattr(args, 'debug_tokens', False):
        head = tokens[:30]
        tail = tokens[-10:] if len(tokens) > 30 else []
        decoded_head = [llm.detokenize([t]).decode('utf-8', errors='replace') for t in head]
        print(f"[llamacpp] first 30 token ids : {head}", file=sys.stderr)
        print(f"[llamacpp] first 30 decoded   : {decoded_head}", file=sys.stderr)
        if tail:
            print(f"[llamacpp] last  10 token ids : {tail}", file=sys.stderr)

    # Capture raw pre-sampling logits via the official logits_processor callback.
    # The callback is called once per token BEFORE sampling, receiving the full
    # vocab logit vector as a numpy array — no ctypes, no pointer arithmetic.
    captured = [None]

    def _capture(input_ids, logits):
        captured[0] = np.array(logits, dtype=np.float32)
        return logits

    greedy = args.no_sample or args.temperature == 0.0
    gen_kwargs = dict(
        top_k  = 1 if greedy else args.top_k,
        top_p  = 1.0,
        temp   = 1e-8 if greedy else args.temperature,
        reset  = True,
        logits_processor = LogitsProcessorList([_capture]),
    )

    log = LogWriter(args.log, window=args.window)
    prev_top20, token_count, t0 = [], 0, time.time()
    token_texts = []
    _debug_gen = getattr(args, 'debug_tokens', False)
    _gen_ids_log = []  # first 5 generated token IDs for --debug-tokens

    for token_id in llm.generate(tokens, **gen_kwargs):
        logits_np = captured[0]
        if token_id == eos_id or token_count >= args.max_tokens:
            break
        if logits_np is None:
            continue

        if _debug_gen and len(_gen_ids_log) < 5:
            _dec = llm.detokenize([token_id]).decode('utf-8', errors='replace')
            _gen_ids_log.append((token_id, repr(_dec)))
            if len(_gen_ids_log) == 5:
                print(f"[llamacpp] first 5 generated ids: "
                      f"{[x[0] for x in _gen_ids_log]}", file=sys.stderr)
                print(f"[llamacpp] first 5 generated tok: "
                      f"{[x[1] for x in _gen_ids_log]}", file=sys.stderr)

        decoded = llm.detokenize([token_id]).decode('utf-8', errors='replace')
        token_texts.append(decoded)
        print(decoded, end="", flush=True)

        # Pass numpy array directly — avoids 248k-element .tolist() allocation
        m     = compute_metrics(logits_np)
        top_k = _top_k_indices(logits_np, k=100)
        top20 = top_k[:20]
        jk    = compute_jaccard(logits_np, prev_top20)
        prev_top20 = top20
        log.push(m, jk, top20=top20, top100=top_k)
        token_count += 1

    print()
    elapsed = time.time() - t0
    print(f"\n[{token_count} tokens in {elapsed:.1f}s = {token_count/elapsed:.1f} tok/s]",
          file=sys.stderr)
    log.close()
    print(f"[log -> {args.log}]", file=sys.stderr)
    _print_annotated(token_texts, log.records, args.window)
    return log.records, args.log, (log.token_top20s, log.token_top100s)


def run_generation_hf(args):
    """HuggingFace transformers backend — original model precision (bf16/fp16/fp32).

    Loads the model with torch_dtype='auto' so it uses whatever dtype is declared
    in config.json (bf16 for most modern models).  Intended as a reference/baseline
    to compare quantized runs against.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import LogitsProcessorList, LogitsProcessor
        import torch
        import numpy as np
    except ImportError:
        sys.exit("transformers / torch not found — pip install transformers torch")

    model_path = args.model_dir
    p = Path(args.prompt)
    if args.prompt.endswith(".txt") and p.exists():
        prompt = p.read_text()
        print(f"[prompt file: {args.prompt}, {len(prompt)} chars]", file=sys.stderr)
    else:
        prompt = args.prompt

    print(f"[hf  model={model_path}  max_tokens={args.max_tokens}  window={args.window}]",
          file=sys.stderr)
    print("[hf] WARNING: single forward pass per token — very slow for large models; "
          "set --max-tokens to a small value if testing.", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    shared_ids = getattr(args, 'input_ids', None)
    if shared_ids is not None:
        tokens = list(shared_ids)
        print(f"[hf] prompt = {len(tokens)} tokens  (shared tokens)", file=sys.stderr)
    else:
        add_bos = getattr(args, 'add_bos', False)
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_bos)
        tokens = enc["input_ids"][0].tolist()
        print(f"[hf] prompt = {len(tokens)} tokens", file=sys.stderr)

    if getattr(args, 'debug_tokens', False):
        head = tokens[:30]
        decoded_head = [tokenizer.decode([t]) for t in head]
        print(f"[hf] first 30 token ids : {head}", file=sys.stderr)
        print(f"[hf] first 30 decoded   : {decoded_head}", file=sys.stderr)

    # Logits processor captures raw logits before any sampling
    class _Capture(LogitsProcessor):
        def __init__(self):
            self.last = None
        def __call__(self, input_ids, scores):
            self.last = scores[0].float().cpu().numpy()
            return scores

    capture = _Capture()
    greedy  = args.no_sample or args.temperature == 0.0

    gen_kwargs = dict(
        max_new_tokens     = args.max_tokens,
        do_sample          = False if greedy else True,
        logits_processor   = LogitsProcessorList([capture]),
        return_dict_in_generate = False,
        output_scores      = False,
    )
    if not greedy:
        gen_kwargs['temperature'] = args.temperature
        gen_kwargs['top_k']       = args.top_k
        gen_kwargs['top_p']       = args.top_p

    input_ids_t = torch.tensor([tokens])
    device = next(model.parameters()).device
    input_ids_t = input_ids_t.to(device)

    log = LogWriter(args.log, window=args.window)
    prev_top20, token_count, t0 = [], 0, time.time()
    token_texts = []
    _debug_gen = getattr(args, 'debug_tokens', False)
    _gen_ids_log = []

    # Generate one token at a time to capture per-token logits
    with torch.no_grad():
        context = input_ids_t.clone()
        for _ in range(args.max_tokens):
            outputs = model(input_ids=context)
            logits_np = outputs.logits[0, -1].float().cpu().numpy()

            if greedy:
                next_id = int(np.argmax(logits_np))
            else:
                probs = np.exp(logits_np - logits_np.max()).astype(np.float64)
                probs /= probs.sum()
                next_id = int(np.random.choice(len(probs), p=probs))

            if next_id == tokenizer.eos_token_id:
                break

            if _debug_gen and len(_gen_ids_log) < 5:
                _dec = tokenizer.decode([next_id])
                _gen_ids_log.append((next_id, repr(_dec)))
                if len(_gen_ids_log) == 5:
                    print(f"[hf] first 5 generated ids: "
                          f"{[x[0] for x in _gen_ids_log]}", file=sys.stderr)
                    print(f"[hf] first 5 generated tok: "
                          f"{[x[1] for x in _gen_ids_log]}", file=sys.stderr)

            decoded = tokenizer.decode([next_id], skip_special_tokens=False)
            token_texts.append(decoded)
            print(decoded, end="", flush=True)

            top_k = _top_k_indices(logits_np, k=100)
            top20 = top_k[:20]
            jk    = compute_jaccard(logits_np, prev_top20)
            m     = compute_metrics(logits_np)
            prev_top20 = top20
            log.push(m, jk, top20=top20, top100=top_k)
            token_count += 1

            context = torch.cat(
                [context, torch.tensor([[next_id]], device=device)], dim=1
            )

    print()
    elapsed = time.time() - t0
    print(f"\n[{token_count} tokens in {elapsed:.1f}s = {token_count/elapsed:.1f} tok/s]",
          file=sys.stderr)
    log.close()
    print(f"[log -> {args.log}]", file=sys.stderr)
    _print_annotated(token_texts, log.records, args.window)
    return log.records, args.log, (log.token_top20s, log.token_top100s)


def run_generation(args):
    """Dispatch to the appropriate backend based on args.runtime."""
    runtime = getattr(args, 'runtime', 'ovgenai') or 'ovgenai'
    if runtime == 'llamacpp':
        return run_generation_llamacpp(args)
    if runtime == 'hf':
        return run_generation_hf(args)
    return run_generation_ovgenai(args)


# ============================================================================
# Comparative analysis
# ============================================================================

def print_comparison(records_a, records_b, label_a, label_b):
    """Side-by-side metric comparison of two generation runs."""
    n = min(len(records_a), len(records_b))
    if n == 0:
        print("Cannot compare — one or both runs produced no records.")
        return

    ra, rb = records_a[:n], records_b[:n]

    def zone_pct(recs, zone_lbl):
        return 100.0 * sum(1 for r in recs if zone_for(r['H'])[0] == zone_lbl) / len(recs)

    def mean_finite(vals):
        finite = [v for v in vals if not math.isnan(v)]
        return statistics.mean(finite) if finite else float('nan')

    def drow(name, va, vb, fmt=".3f"):
        d    = vb - va
        sign = "+" if d >= 0 else ""
        print(f"    {name:<22}  A: {va:{fmt}}    B: {vb:{fmt}}    Δ: {sign}{d:{fmt}}")

    ma_H   = statistics.mean(r['H']    for r in ra)
    mb_H   = statistics.mean(r['H']    for r in rb)
    ma_t1  = statistics.mean(r['top1'] for r in ra)
    mb_t1  = statistics.mean(r['top1'] for r in rb)
    ma_ev  = statistics.mean(r['eff_v'] for r in ra)
    mb_ev  = statistics.mean(r['eff_v'] for r in rb)
    ma_wjk = mean_finite([r.get('win_jaccard', float('nan')) for r in ra])
    mb_wjk = mean_finite([r.get('win_jaccard', float('nan')) for r in rb])

    print("=" * 70)
    print("  COMPARATIVE ANALYSIS")
    print("=" * 70)
    print(f"  Runtime A  [{label_a}]  {len(records_a)} windows total")
    print(f"  Runtime B  [{label_b}]  {len(records_b)} windows total")
    print(f"  Comparing first {n} windows (min of both)\n")

    print("  ── Mean metrics ──────────────────────────────────────────────")
    drow("entropy H (nats)",  ma_H,  mb_H)
    drow("top-1 prob",        ma_t1, mb_t1)
    drow("eff_vocab",         ma_ev, mb_ev, fmt=".2f")
    if not (math.isnan(ma_wjk) and math.isnan(mb_wjk)):
        a_wjk = 0.0 if math.isnan(ma_wjk) else ma_wjk
        b_wjk = 0.0 if math.isnan(mb_wjk) else mb_wjk
        drow("win-Jaccard", a_wjk, b_wjk)
    print()

    print("  ── Zone distribution ─────────────────────────────────────────")
    print(f"    {'Zone':<12}  {'A%':>7}  {'B%':>7}  {'Δ%':>8}")
    for _, lbl, _ in ZONES:
        pa   = zone_pct(ra, lbl)
        pb   = zone_pct(rb, lbl)
        d    = pb - pa
        sign = "+" if d >= 0 else ""
        flag = "  ⚠" if abs(d) > 10 and lbl in ("CHAOS", "LOCK", "COLLAPSING") else ""
        print(f"    {lbl:<12}  {pa:>6.1f}%  {pb:>6.1f}%  {sign}{d:>6.1f}%{flag}")
    print()

    # Per-window drift (only meaningful when both ran the same prompt + settings)
    h_drifts  = [abs(rb[i]['H']    - ra[i]['H'])    for i in range(n)]
    t1_drifts = [abs(rb[i]['top1'] - ra[i]['top1']) for i in range(n)]
    mean_hd   = statistics.mean(h_drifts)
    max_hd    = max(h_drifts)
    mean_td   = statistics.mean(t1_drifts)
    max_td    = max(t1_drifts)

    print("  ── Per-window drift (matched by window index) ─────────────────")
    print(f"    mean |ΔH|     : {mean_hd:.3f}  max: {max_hd:.3f}")
    print(f"    mean |Δtop-1| : {mean_td:.3f}  max: {max_td:.3f}")

    if mean_hd < 0.10:
        drift_label = "LOW — distributions nearly identical"
    elif mean_hd < 0.30:
        drift_label = "MILD — small distributional differences"
    elif mean_hd < 0.60:
        drift_label = "MODERATE — notable divergence (quantization or runtime)"
    else:
        drift_label = "STRONG — significant distributional divergence"
    print(f"    Drift: {drift_label}\n")
    print("  (TIP: use --temperature 0.0 on both runs to isolate quantization effects)")
    print("  (     with greedy decoding the per-window drift reflects only weight precision)")
    print("=" * 70)


def print_cross_model_divergence(top20s_a, top20s_b, top100s_a, top100s_b, label_a, label_b):
    """Cross-model top-20 Jaccard analysis up to the first token divergence.

    top20s_*/top100s_* are per-token lists from LogWriter.token_top20s/top100s.
    At the divergence point we report the rank of each model's chosen token in
    the other model's top-100 list, showing whether the choice was at least
    considered by the other quantization.
    """
    n = min(len(top20s_a), len(top20s_b))
    if n == 0:
        return

    rows = []          # (token_idx, jaccard, tok_a, tok_b, diverged)
    diverge_at = None

    for i in range(n):
        top_a = top20s_a[i]
        top_b = top20s_b[i]
        if not top_a or not top_b:
            continue

        sa, sb = set(top_a), set(top_b)
        inter = len(sa & sb)
        union = len(sa | sb)
        j = inter / union if union else 0.0

        tok_a = top_a[0]
        tok_b = top_b[0]
        diverged = tok_a != tok_b
        rows.append((i, j, tok_a, tok_b, diverged))

        if diverged and diverge_at is None:
            diverge_at = i
            break   # stop: subsequent tokens are on different context paths

    if not rows:
        return

    print("\n" + "=" * 70)
    print("  CROSS-MODEL TOP-20 JACCARD  (up to first token divergence)")
    print("=" * 70)
    print(f"  A = {label_a}")
    print(f"  B = {label_b}\n")

    # ── Per-token table ───────────────────────────────────────────────────
    MAX_ROWS = 30
    show = rows if len(rows) <= MAX_ROWS else rows[:15] + [None] + rows[-5:]
    print(f"  {'Tok':>4}  {'Jaccard':>7}  {'A-argmax':>10}  {'B-argmax':>10}  Status")
    print(f"  {'-'*4}  {'-'*7}  {'-'*10}  {'-'*10}  ------")
    for r in show:
        if r is None:
            print(f"  {'...':>4}  {'...':>7}  {'...':>10}  {'...':>10}")
            continue
        idx, j, ta, tb, div = r
        status = "DIVERGE ←" if div else "agree"
        print(f"  {idx:>4}  {j:>7.3f}  {ta:>10}  {tb:>10}  {status}")

    # ── Summary ───────────────────────────────────────────────────────────
    jaccards = [r[1] for r in rows]
    mean_j   = statistics.mean(jaccards)
    min_j    = min(jaccards)

    print()
    if diverge_at is None:
        print(f"  No divergence detected in {n} tokens.  "
              f"Mean cross-model Jaccard: {mean_j:.3f}")
    elif diverge_at == 0:
        j0 = rows[0][1]
        print(f"  Divergence at token 0 (first generated token).")
        print(f"  Cross-model Jaccard at divergence point: {j0:.3f}")
        if j0 >= 0.70:
            label = "HIGH overlap — both models considered mostly the same candidates"
        elif j0 >= 0.40:
            label = "MODERATE overlap — partial agreement on candidate set"
        else:
            label = "LOW overlap — models were looking at very different candidates"
        print(f"  Overlap: {label}")
    else:
        print(f"  Agreed on {diverge_at} token(s) before diverging at token {diverge_at}.")
        print(f"  Mean cross-model Jaccard (agreement window): {mean_j:.3f}  "
              f"min: {min_j:.3f}")
        j_div = rows[-1][1]   # Jaccard at the divergence token itself
        print(f"  Cross-model Jaccard at divergence point    : {j_div:.3f}")

    # ── Rank-in-other-model at divergence ─────────────────────────────────
    if diverge_at is not None:
        div_row = rows[-1]   # last row = divergence token
        _, _, tok_a, tok_b, _ = div_row
        t100_a = top100s_a[diverge_at] if diverge_at < len(top100s_a) else []
        t100_b = top100s_b[diverge_at] if diverge_at < len(top100s_b) else []

        def rank_in(tok, ranked_list):
            try:
                return ranked_list.index(tok) + 1  # 1-indexed
            except ValueError:
                return None

        rank_a_in_b = rank_in(tok_a, t100_b)
        rank_b_in_a = rank_in(tok_b, t100_a)
        k = len(t100_a) or len(t100_b) or 100

        def rank_str(r, k):
            if r is None:
                return f"not in top-{k}"
            return f"rank #{r} in top-{k}"

        print()
        print(f"  ── Token cross-rank at divergence point (token {diverge_at}) ───")
        print(f"  A chose {tok_a:>8}  →  rank #1 in A   /  {rank_str(rank_a_in_b, k)} of B")
        print(f"  B chose {tok_b:>8}  →  rank #1 in B   /  {rank_str(rank_b_in_a, k)} of A")
        if rank_a_in_b and rank_b_in_a:
            print(f"  Both tokens were in each other's top-{k}: toss-up between quant schemes.")
        elif rank_a_in_b is None and rank_b_in_a is None:
            print(f"  Neither token appeared in the other's top-{k}: deep distributional split.")
        elif rank_a_in_b is None:
            print(f"  A's choice not in B's top-{k}: B had no probability mass there.")
        else:
            print(f"  B's choice not in A's top-{k}: A had no probability mass there.")

    print()
    print("  Interpretation:")
    print("   - High Jaccard before divergence → models 'agree' on which tokens")
    print("     are plausible; the chosen token is a toss-up between quant schemes.")
    print("   - Low Jaccard → the two quantizations are exploring genuinely")
    print("     different probability mass; divergence is structurally deeper.")
    print("=" * 70)


def print_reference_comparison(records_a, records_b, records_ref,
                               top20s_a, top20s_b, top20s_ref,
                               top100s_a, top100s_b, top100s_ref,
                               label_a, label_b, label_ref):
    """Three-way quality ladder: score A and B against a reference run.

    Metrics reported per model vs reference:
      - Mean |ΔH|         : entropy drift (lower = more faithful)
      - Mean |Δtop-1|     : top-1 prob drift
      - Token agreement % : fraction of tokens where argmax matches reference
      - Mean cross-Jaccard: avg top-20 set overlap with reference per token
    """
    n_a   = min(len(records_a),   len(records_ref))
    n_b   = min(len(records_b),   len(records_ref))
    n_tok = min(len(top20s_a), len(top20s_b), len(top20s_ref))
    if n_a == 0 and n_b == 0:
        return

    def _window_stats(recs_x, recs_ref, n):
        """Mean |ΔH| and |Δtop-1| between x and ref over first n windows."""
        dh  = statistics.mean(abs(recs_x[i]['H']    - recs_ref[i]['H'])    for i in range(n))
        dt1 = statistics.mean(abs(recs_x[i]['top1'] - recs_ref[i]['top1']) for i in range(n))
        return dh, dt1

    def _token_stats(top20s_x, top100s_x, top20s_ref, top100s_ref, n):
        """Token-level: argmax agreement %, mean cross-Jaccard, mean rank of ref argmax in x."""
        agree   = 0
        jacs    = []
        ranks   = []
        for i in range(n):
            ta = top20s_x[i];   tr = top20s_ref[i]
            if not ta or not tr:
                continue
            agree += int(ta[0] == tr[0])
            inter = len(set(ta) & set(tr))
            union = len(set(ta) | set(tr))
            jacs.append(inter / union if union else 0.0)
            # rank of reference argmax in x's top-100
            ref_tok = tr[0]
            try:
                rank = top100s_x[i].index(ref_tok) + 1
            except (ValueError, IndexError):
                rank = None
            ranks.append(rank)
        n_valid   = max(sum(1 for t in top20s_x[:n] if t) , 1)
        agree_pct = 100.0 * agree / n_valid
        mean_jac  = statistics.mean(jacs)  if jacs  else float('nan')
        valid_rnk = [r for r in ranks if r is not None]
        mean_rank = statistics.mean(valid_rnk) if valid_rnk else float('nan')
        missed    = ranks.count(None)
        return agree_pct, mean_jac, mean_rank, missed, n_valid

    dh_a, dt1_a = _window_stats(records_a, records_ref, n_a) if n_a else (float('nan'),)*2
    dh_b, dt1_b = _window_stats(records_b, records_ref, n_b) if n_b else (float('nan'),)*2

    agr_a, jac_a, rnk_a, miss_a, nv_a = _token_stats(
        top20s_a, top100s_a, top20s_ref, top100s_ref, n_tok)
    agr_b, jac_b, rnk_b, miss_b, nv_b = _token_stats(
        top20s_b, top100s_b, top20s_ref, top100s_ref, n_tok)

    def fmt(v, f=".3f"):
        return "n/a" if math.isnan(v) else f"{v:{f}}"

    print("\n" + "=" * 70)
    print("  REFERENCE QUALITY COMPARISON")
    print("=" * 70)
    print(f"  Reference : {label_ref}")
    print(f"  Model A   : {label_a}")
    print(f"  Model B   : {label_b}")
    print(f"  Windows   : A={n_a}  B={n_b}  (matched to reference)")
    print(f"  Tokens    : {n_tok} (per-token metrics)\n")

    print(f"  {'Metric':<30}  {'Model A':>10}  {'Model B':>10}  {'Better':>8}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*8}")

    def winner(va, vb, lower_is_better=True):
        if math.isnan(va) or math.isnan(vb):
            return "  —"
        if lower_is_better:
            return "  A" if va < vb else ("  B" if vb < va else "tie")
        return "  A" if va > vb else ("  B" if vb > va else "tie")

    rows = [
        ("mean |ΔH| vs ref",          dh_a,  dh_b,  True,  ".4f"),
        ("mean |Δtop-1| vs ref",      dt1_a, dt1_b, True,  ".4f"),
        ("token argmax agreement %",   agr_a, agr_b, False, ".1f"),
        ("mean cross-Jaccard vs ref",  jac_a, jac_b, False, ".3f"),
        ("mean rank of ref argmax",    rnk_a, rnk_b, True,  ".1f"),
    ]
    for name, va, vb, lib, fmt_s in rows:
        w = winner(va, vb, lib)
        print(f"  {name:<30}  {fmt(va, fmt_s):>10}  {fmt(vb, fmt_s):>10}  {w:>8}")

    print()
    if not math.isnan(miss_a) and nv_a:
        print(f"  A: ref argmax not in A top-100 for {miss_a}/{nv_a} tokens "
              f"({100*miss_a/nv_a:.1f}%)")
    if not math.isnan(miss_b) and nv_b:
        print(f"  B: ref argmax not in B top-100 for {miss_b}/{nv_b} tokens "
              f"({100*miss_b/nv_b:.1f}%)")

    # Tally wins
    wins_a = sum(1 for _, va, vb, lib, _ in rows
                 if not math.isnan(va) and not math.isnan(vb)
                 and ((lib and va < vb) or (not lib and va > vb)))
    wins_b = sum(1 for _, va, vb, lib, _ in rows
                 if not math.isnan(va) and not math.isnan(vb)
                 and ((lib and vb < va) or (not lib and vb > va)))
    print()
    if wins_a > wins_b:
        print(f"  VERDICT: Model A ({label_a}) is closer to reference "
              f"({wins_a}/{len(rows)} metrics better)")
    elif wins_b > wins_a:
        print(f"  VERDICT: Model B ({label_b}) is closer to reference "
              f"({wins_b}/{len(rows)} metrics better)")
    else:
        print(f"  VERDICT: Models A and B are equally close to reference ({wins_a} each)")
    print("=" * 70)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model_dir", nargs="?", help="Model path (OV dir or .gguf file/dir)")
    parser.add_argument("prompt",    nargs="?", help="Prompt text or .txt file path")
    parser.add_argument("--runtime", default="ovgenai",
                        choices=["ovgenai", "llamacpp", "hf"],
                        help="Inference runtime for model_dir (default: ovgenai)")
    parser.add_argument("--device",      default="CPU")
    parser.add_argument("--max-tokens",  type=int,   default=2000, dest="max_tokens")
    parser.add_argument("--window",      type=int,   default=10)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k",       type=int,   default=30,  dest="top_k")
    parser.add_argument("--top-p",       type=float, default=0.9, dest="top_p")
    parser.add_argument("--no-sample",   action="store_true", dest="no_sample")
    parser.add_argument("--log",         default="logits_stats.log")
    parser.add_argument("--plot",        default=None,
                        help='PNG output path, or "none" to skip (default: <log>.png)')
    parser.add_argument("--analyze",     metavar="LOG_FILE",
                        help="Skip generation; analyse an existing log file instead")
    # llama.cpp-specific options
    parser.add_argument("--n-gpu-layers", type=int, default=0, dest="n_gpu_layers",
                        help="Layers to offload to GPU (llamacpp backend, default: 0=CPU only)")
    parser.add_argument("--n-ctx",        type=int, default=4096, dest="n_ctx",
                        help="Context window size (llamacpp backend, default: 4096)")
    parser.add_argument("--n-threads",    type=int, default=0,    dest="n_threads",
                        help="CPU threads (llamacpp backend, default: auto = cpu_count/2)")
    parser.add_argument("--add-bos",      action="store_true",    dest="add_bos",
                        help="Prepend BOS token (llamacpp only; off by default — correct for "
                             "Qwen3/Mistral/modern models; enable for LLaMA-1/2)")
    parser.add_argument("--debug-tokens", action="store_true",    dest="debug_tokens",
                        help="Print first 30 / last 10 prompt token IDs from each backend "
                             "to stderr — useful for verifying both sides see the same input")
    parser.add_argument("--shared-tokenizer", default=None, dest="shared_tokenizer",
                        choices=["ovgenai", "llamacpp"],
                        help="Tokenize the prompt once using this backend's tokenizer and feed "
                             "identical token IDs to both runtimes.  Eliminates tokenizer "
                             "divergence in cross-runtime comparisons.")
    # Comparison mode
    parser.add_argument("--compare",    metavar="MODEL_B",
                        help="Run a second model and show comparative analysis")
    parser.add_argument("--runtime-b",  default=None, dest="runtime_b",
                        choices=["ovgenai", "llamacpp", "hf"],
                        help="Runtime for MODEL_B (default: llamacpp if --compare given, "
                             "else same as --runtime)")
    # Reference / baseline mode
    parser.add_argument("--reference",   default=None, metavar="MODEL_REF",
                        help="HuggingFace model dir to run as full-precision reference. "
                             "Scores MODEL_A and MODEL_B against it to measure quantization "
                             "fidelity.  Requires --compare.")
    parser.add_argument("--runtime-ref", default="hf", dest="runtime_ref",
                        choices=["hf", "ovgenai", "llamacpp"],
                        help="Runtime for the reference model (default: hf)")
    args = parser.parse_args()
    args.input_ids = None  # populated below if --shared-tokenizer is used

    # ── Shared tokenization (--shared-tokenizer) ───────────────────────────────
    # Tokenize the prompt once and pass identical IDs to both backends so that
    # any difference in generated text is purely due to weights / quantization,
    # not tokenizer divergence.
    if args.shared_tokenizer and getattr(args, 'prompt', None):
        p_path = Path(args.prompt)
        raw = p_path.read_text() if args.prompt.endswith('.txt') and p_path.exists() else args.prompt

        # Find the model path for the chosen tokenizer runtime.
        # In comparison mode model A is args.model_dir / runtime args.runtime,
        # model B is args.compare / runtime_b.  Pick whichever matches.
        runtime_b = getattr(args, 'runtime_b', None) or (
            'llamacpp' if args.runtime == 'ovgenai' else 'ovgenai'
        ) if args.compare else args.runtime
        if args.shared_tokenizer == args.runtime:
            tok_path = args.model_dir
        elif args.compare and args.shared_tokenizer == runtime_b:
            tok_path = args.compare
        else:
            tok_path = args.model_dir  # best guess

        try:
            if args.shared_tokenizer == 'ovgenai':
                import openvino_genai as _ovg
                import numpy as _np
                _tok = _ovg.Tokenizer(tok_path)
                _enc = _tok.encode(raw)
                args.input_ids = _np.array(_enc.input_ids).flatten().astype(int).tolist()
            else:  # llamacpp
                import llama_cpp as _lc
                _tok = _lc.Llama(model_path=tok_path, vocab_only=True, verbose=False)
                args.input_ids = list(_tok.tokenize(raw.encode('utf-8'), add_bos=False, special=True))
            print(f"[shared-tokenizer={args.shared_tokenizer}] {len(args.input_ids)} tokens "
                  f"from {Path(tok_path).name}", file=sys.stderr)
        except Exception as _e:
            print(f"[shared-tokenizer] failed ({_e}); each backend will tokenize independently",
                  file=sys.stderr)
            args.input_ids = None

    if args.analyze:
        # ── Analysis-only mode ──────────────────────────────────────────
        records = parse_log(args.analyze)
        if not records:
            sys.exit(f"No logits_stats blocks found in {args.analyze}")
        title = Path(args.analyze).name
        png   = args.plot if args.plot else str(Path(args.analyze).with_suffix(".png"))
        print_summary(records, title)
        make_plot(records, title, png)

    elif args.compare:
        # ── Dual-runtime comparison mode ────────────────────────────────
        if not args.model_dir or not args.prompt:
            parser.error("model_dir and prompt are required for comparison mode")

        # Resolve runtime for B
        runtime_b = args.runtime_b or ("llamacpp" if args.runtime == "ovgenai" else "ovgenai")

        # ── Run reference first (if requested) ──────────────────────────
        records_ref = log_ref = top20s_ref = top100s_ref = label_ref = None
        if args.reference:
            runtime_ref = args.runtime_ref or 'hf'
            args_ref = copy.copy(args)
            args_ref.model_dir = args.reference
            args_ref.runtime   = runtime_ref
            args_ref.log       = Path(args.log).stem + "_REF" + Path(args.log).suffix
            label_ref = f"{runtime_ref}:{Path(args.reference).name}"
            print(f"\n{'='*70}", file=sys.stderr)
            print(f"[comparison] Running Reference  ({label_ref})", file=sys.stderr)
            print(f"{'='*70}", file=sys.stderr)
            records_ref, log_ref, (top20s_ref, top100s_ref) = run_generation(args_ref)

        # Run model A
        args_a = copy.copy(args)
        args_a.log = Path(args.log).stem + "_A" + Path(args.log).suffix
        label_a = f"{args.runtime}:{Path(args.model_dir).name}"
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"[comparison] Running Model A  ({label_a})", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)
        records_a, log_a, (top20s_a, top100s_a) = run_generation(args_a)

        # Run model B
        args_b = copy.copy(args)
        args_b.model_dir = args.compare
        args_b.runtime   = runtime_b
        args_b.log       = Path(args.log).stem + "_B" + Path(args.log).suffix
        label_b = f"{runtime_b}:{Path(args.compare).name}"
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"[comparison] Running Model B  ({label_b})", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)
        records_b, log_b, (top20s_b, top100s_b) = run_generation(args_b)

        # Individual summaries — suppressed when --reference is given to reduce noise
        show_summary = args.reference is None
        if records_ref is not None:
            if show_summary:
                print_summary(records_ref, f"Reference — {label_ref}")
            png_ref = str(Path(log_ref).with_suffix(".png"))
            make_plot(records_ref, f"Reference — {label_ref}", png_ref)

        if show_summary:
            print_summary(records_a, f"Model A — {label_a}")
        png_a = args.plot if args.plot else str(Path(log_a).with_suffix(".png"))
        make_plot(records_a, f"Model A — {label_a}", png_a)

        if show_summary:
            print_summary(records_b, f"Model B — {label_b}")
        png_b = str(Path(log_b).with_suffix(".png"))
        make_plot(records_b, f"Model B — {label_b}", png_b)

        # Comparison
        print_comparison(records_a, records_b, label_a, label_b)
        print_cross_model_divergence(top20s_a, top20s_b, top100s_a, top100s_b, label_a, label_b)

        if records_ref is not None:
            print_cross_model_divergence(
                top20s_a, top20s_ref, top100s_a, top100s_ref, label_a, label_ref)
            print_cross_model_divergence(
                top20s_b, top20s_ref, top100s_b, top100s_ref, label_b, label_ref)
            print_reference_comparison(
                records_a, records_b, records_ref,
                top20s_a,  top20s_b,  top20s_ref,
                top100s_a, top100s_b, top100s_ref,
                label_a, label_b, label_ref,
            )
    elif args.reference:
        # ── Single-model vs reference mode ──────────────────────────────
        if not args.model_dir or not args.prompt:
            parser.error("model_dir and prompt are required for reference comparison mode")

        runtime_ref = args.runtime_ref or 'hf'
        args_ref = copy.copy(args)
        args_ref.model_dir = args.reference
        args_ref.runtime   = runtime_ref
        args_ref.log       = Path(args.log).stem + "_REF" + Path(args.log).suffix
        label_ref = f"{runtime_ref}:{Path(args.reference).name}"
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"[reference] Running Reference  ({label_ref})", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)
        records_ref, log_ref, (top20s_ref, top100s_ref) = run_generation(args_ref)

        label_a = f"{args.runtime}:{Path(args.model_dir).name}"
        args_a = copy.copy(args)
        args_a.log = Path(args.log).stem + "_A" + Path(args.log).suffix
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"[reference] Running Model  ({label_a})", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)
        records_a, log_a, (top20s_a, top100s_a) = run_generation(args_a)

        # Summaries suppressed when --reference is given; plots still saved to disk
        png_ref = str(Path(log_ref).with_suffix(".png"))
        make_plot(records_ref, f"Reference — {label_ref}", png_ref)

        png_a = args.plot if args.plot else str(Path(log_a).with_suffix(".png"))
        make_plot(records_a, f"Model — {label_a}", png_a)

        # Single-model report: reuse print_reference_comparison with B == ref
        # so the table shows how A compares to ground truth directly
        print_cross_model_divergence(
            top20s_a, top20s_ref, top100s_a, top100s_ref, label_a, label_ref)
        print_reference_comparison(
            records_a, records_ref, records_ref,
            top20s_a,  top20s_ref, top20s_ref,
            top100s_a, top100s_ref, top100s_ref,
            label_a, label_ref, label_ref,
        )

    else:
        # ── Generation + analysis mode ──────────────────────────────────
        if not args.model_dir or not args.prompt:
            parser.error("model_dir and prompt are required for generation mode "
                         "(or use --analyze LOG_FILE for analysis-only)")
        records, log_path, _ = run_generation(args)
        if not records:
            print("No logit windows recorded (no return_logits data).", file=sys.stderr)
            return
        title = Path(log_path).name
        png   = args.plot if args.plot else str(Path(log_path).with_suffix(".png"))
        print_summary(records, title)
        make_plot(records, title, png)

if __name__ == "__main__":
    main()
