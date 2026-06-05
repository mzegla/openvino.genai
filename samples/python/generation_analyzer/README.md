# Generation Analyzer

A runtime-agnostic logit-space diagnostic tool for LLM inference.  Supports
**OpenVINO GenAI** (`ContinuousBatchingPipeline`), **llama.cpp**
(`llama-cpp-python`), and **HuggingFace Transformers** (full-precision reference)
as interchangeable generation backends.  Captures raw logits at every token,
aggregates them into windows, and produces a health report: entropy zones,
phase timelines, repetition-loop detection, a color-annotated replay of the
generated text, and (optionally) a side-by-side comparative analysis of two
quantized backends scored against a full-precision reference.

---

## Quick start

### Prerequisites

Install the backend(s) you intend to use plus the shared dependencies:

```bash
# OpenVINO GenAI backend (default) - for now it requires custom genai build with logits exposed
# Intel GPU is natively supported — no extra steps beyond the standard wheel
pip install openvino-genai matplotlib

# llama.cpp backend — CPU-only build (baseline / cross-check)
pip install llama-cpp-python matplotlib

# HuggingFace Transformers backend (full-precision reference)
pip install transformers torch accelerate

# llama.cpp backend — Intel GPU via SYCL (Intel oneAPI)
# Requires the Intel oneAPI Base Toolkit to be installed and sourced:
#   https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html
# Then build llama-cpp-python from source with the SYCL backend:
source /opt/intel/oneapi/setvars.sh
CMAKE_ARGS="-DGGML_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx" \
    pip install llama-cpp-python --no-binary llama-cpp-python
```

> All backends can be installed side-by-side; the script lazy-imports whichever
> is needed at runtime.

### Usage

```bash
# OpenVINO GenAI on Intel GPU
python generation_analyzer.py /path/to/ov_model "Your prompt" --device GPU

# OpenVINO GenAI on CPU
python generation_analyzer.py /path/to/ov_model "Your prompt"

# llama.cpp — pass a .gguf file or a directory containing one (CPU)
python generation_analyzer.py model.gguf "Your prompt" --runtime llamacpp

# llama.cpp on Intel GPU (SYCL build required — see Prerequisites above)
python generation_analyzer.py model.gguf "Your prompt" --runtime llamacpp \
    --n-gpu-layers -1

# HuggingFace Transformers — original model precision (bf16/fp16)
# Recommended only for small models or short --max-tokens runs; see NOTE below
python generation_analyzer.py /path/to/hf_model "Your prompt" --runtime hf

# Analyse an existing log without re-running the model
python generation_analyzer.py --analyze logits_stats.log
```

> **NOTE — HF backend speed:** the `hf` runtime runs one full forward pass per
> generated token at full precision.  For models larger than ~7B this is very
> slow on CPU and requires substantial VRAM on GPU.  Use `--max-tokens` to limit
> the run length when using it as a reference baseline.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--runtime R` | `ovgenai` | Backend: `ovgenai`, `llamacpp`, or `hf` |
| `--device S` | `CPU` | OpenVINO device string (ovgenai only) |
| `--n-gpu-layers N` | `0` | Layers to offload to GPU (llamacpp only; `-1` = all) |
| `--n-ctx N` | `4096` | Context window size (llamacpp only) |
| `--n-threads N` | auto | CPU threads (llamacpp only; default = cpu_count / 2) |
| `--add-bos` | off | Prepend BOS token (llamacpp only; for LLaMA-1/2 legacy models) |
| `--max-tokens N` | 2000 | Maximum new tokens to generate |
| `--window N` | 10 | Tokens per analysis window |
| `--temperature F` | 0.9 | Sampling temperature (0.0 = greedy) |
| `--top-k N` | 30 | Top-K sampling |
| `--top-p F` | 0.9 | Nucleus (top-P) sampling |
| `--no-sample` | — | Force greedy decoding |
| `--log PATH` | logits_stats.log | Log file path |
| `--plot PATH` | `<log>.png` | Save 6-panel dashboard PNG (`none` to skip) |
| `--analyze LOG` | — | Analyse an existing log; skip generation entirely |
| `--compare MODEL_B` | — | Run a second model and show comparative analysis |
| `--runtime-b R` | `llamacpp` | Runtime for `--compare` model |
| `--reference MODEL` | — | Run a full-precision reference model and score A (and B) against it |
| `--runtime-ref R` | `hf` | Runtime for the reference model |
| `--shared-tokenizer R` | — | Tokenize once with this backend and feed identical token IDs to all runs; eliminates tokenizer divergence (`ovgenai` or `llamacpp`) |
| `--debug-tokens` | — | Print prompt token IDs and first 5 generated IDs to stderr (verify identical inputs) |

### Comparative analysis

Run the same prompt through two models or runtimes and get a side-by-side report.
Always use `--no-sample` (greedy decoding) and `--shared-tokenizer` for apples-to-apples
comparison — this eliminates both sampling noise and tokenizer divergence, so any
difference in output is purely due to quantization or runtime.

```bash
# OV INT4 vs llama.cpp Q4_K_M — greedy, shared tokenizer
python generation_analyzer.py /models/Qwen3-int4-ov "Your prompt" \
    --compare /models/Qwen3-Q4_K_M.gguf --runtime-b llamacpp \
    --no-sample --shared-tokenizer llamacpp \
    --log run.log --plot comparison.png

# Two OpenVINO quantizations
python generation_analyzer.py /models/Qwen3-fp16-ov "Your prompt" \
    --compare /models/Qwen3-int8-ov --runtime-b ovgenai \
    --no-sample --shared-tokenizer ovgenai
```

The comparison report shows:
- **Mean metrics** — entropy H, top-1 probability, effective vocabulary, win-win
  Jaccard — for both runs side-by-side with deltas.
- **Zone distribution** — percentage of windows in each entropy zone per run.
- **Per-window drift** — mean and max `|ΔH|` and `|Δtop-1|` matched by window index.
- **Cross-model Jaccard table** — per-token top-20 set overlap up to the first
  divergence point, showing exactly where the two quantizations start disagreeing.
- **Cross-rank at divergence** — the rank of each model's chosen token in the other
  model's top-100 list, revealing whether the divergence was a toss-up or a deep
  distributional split.

> **GPU device selection:**
> - **OpenVINO GenAI** — use `--device GPU` (or `GPU.0`, `GPU.1` for multi-GPU).
>   Intel GPU is natively supported by the standard OpenVINO wheel.
> - **llama.cpp** — no `--device` string; GPU usage is `--n-gpu-layers -1` (all layers)
>   or a partial count.  Intel GPU requires a SYCL build (see Prerequisites);
>   the GPU type is baked into the wheel at compile time, not selected at runtime.

### Reference quality comparison

Add `--reference` to score one or two quantized models against a full-precision
baseline, answering the question *"which quantization/runtime is more numerically
faithful to the original model?"*

When `--reference` is given, the reference model runs **first**; quantized runs follow.
All three runs share the same token IDs when `--shared-tokenizer` is set.

```bash
# Single quantized model vs HF reference
python generation_analyzer.py /models/Qwen3-int4-ov "Your prompt" \
    --reference /models/Qwen3-hf \
    --no-sample --shared-tokenizer ovgenai --max-tokens 50

# Two quantizations vs HF reference (reference runs first)
python generation_analyzer.py /models/Qwen3-int4-ov "Your prompt" \
    --compare /models/Qwen3-Q4_K_M.gguf --runtime-b llamacpp \
    --reference /models/Qwen3-hf \
    --no-sample --shared-tokenizer llamacpp --max-tokens 50
```

The reference comparison table reports five metrics per model vs reference:

| Metric | Lower/Higher = better |
|--------|----------------------|
| mean `\|ΔH\|` vs ref | lower |
| mean `\|Δtop-1 prob\|` vs ref | lower |
| token argmax agreement % | higher |
| mean cross-Jaccard (top-20) vs ref | higher |
| mean rank of ref's argmax in model's top-100 | lower |

A **VERDICT** line at the bottom names the model that wins more metrics.

---

## Understanding the output

### Phase zones

The tool classifies every analysis window into one of five entropy zones based on the
Shannon entropy **H** of the model's output distribution (measured in bits, log base 2).

#### LOCK — H < 0.15 bits · red in annotated output

The distribution has collapsed almost entirely onto a single token.  Top-1 probability
is typically above 98%.  Effective vocabulary ≈ 1.0–1.1 tokens.

**Intuition:** the model has no meaningful choice; it is deterministically emitting one
token.  In isolation (e.g., a newline at the end of a sentence) this is normal.  When it
persists for multiple windows the model is stuck in a single-token repetition attractor —
a true loop.

#### COLLAPSING — 0.15 ≤ H < 0.5 bits · yellow in annotated output

Distribution is tight: top-1 holds 60–97% of the probability mass.  The model is highly
confident, choosing from roughly 1–3 meaningful candidates.

**Intuition:** this is the dual-use zone.  A well-trained model generating factual
content (a mathematical formula, a proper name, a standard phrase) will spend a lot of
time here — that is correct and healthy.  But COLLAPSING is also the *precursor* zone to
a repetition loop.  Whether it is benign depends on context: if entropy is *falling
over time* and win-win Jaccard is *rising*, the model is converging toward a loop.  If
entropy is *flat* and win-win Jaccard stays low, the model is simply confident.

#### NORMAL — 0.5 ≤ H < 2.5 bits · uncolored

Effective vocabulary ≈ 1.4–6 tokens ($2^{0.5}$–$2^{2.5}$): the model has a clear
favourite but a small number of plausible alternatives.  This is the expected zone
for coherent open-ended generation.

**Intuition:** the model has a clear direction but acknowledges genuine alternatives.
Factual answers, conversational replies, and structured documents largely live here.
NORMAL entropy does **not** guarantee factual correctness — a model can confabulate
wrong facts with normal-range confidence.

#### HIGH-H — 2.5 ≤ H < 3.5 bits · cyan in annotated output

Effective vocabulary ≈ 6–11 tokens ($2^{2.5}$–$2^{3.5}$): the model sees no clearly
dominant next token.  The model is genuinely uncertain about what comes next.

**Intuition:** common at topic transitions, rhetorical questions, open-ended list
continuations, or when the model is on unfamiliar ground.  Isolated HIGH-H windows are
unremarkable.  Sustained HIGH-H from the very first window means the model never had
good footing on the prompt — any coherent-looking phrase it generates may trap it in a
COLLAPSING/LOCK attractor thereafter (*fragile start* pattern).

#### CHAOS — H ≥ 3.5 bits · magenta in annotated output

Effectively flat distribution over many tokens; next-token choice is close to random.

**Intuition:** an isolated CHAOS spike usually marks a hallucination risk or a hard
knowledge boundary.  Sustained CHAOS at the start means the model cannot handle this
prompt type (wrong model format, catastrophic quantisation, model–task mismatch, or a
base model given an instruction-style prompt).

#### HIGH-CONF (segment label only)

Not a zone but a segment-level classifier: a tight distribution (≥60% LOCK+COLLAPSING)
with *low* win-win Jaccard (< 0.28).  This distinguishes **confident correct generation**
(the model is certain because the content is factual and predictable) from a
**repetition loop** (the model is certain because it has locked onto a phrase cycle).
The win-win Jaccard is the discriminating signal: real loops recycle the same token
distribution window-to-window, raising the Jaccard; factual generation moves forward,
keeping the Jaccard low.

---

### Temporal flow

#### Phase sequence and streaks

The phase sequence shows the zone label for every window in order:

```
NOR→COL×2→LOC→COL→NOR×2→COL×3→LOC×4
```

Repeated labels are run-length compressed.  The longest streak per zone is listed
separately.  A LOC streak ≥ 3 almost always means a repetition loop is active.

#### H slope and top-1 slope

OLS regression over all windows gives a global trend:

- **H slope < 0** → entropy falling over time; model is converging toward a tighter
  distribution.  Slopes are normalised by the run's own mean entropy so the same
  threshold applies regardless of model or temperature.
- **top-1 slope > 0** → confidence rising over time, complementary confirmation of
  convergence.

Both slopes must be non-trivial and consistent with the phase breakdown before a
*converging* verdict is raised.

#### H autocorrelation (lag 1)

Pearson correlation between H(window i) and H(window i+1):

| Value | Meaning |
|-------|---------|
| ≈ 1.0 | Smooth, persistent trajectory — entropy drifts slowly |
| 0.4–0.7 | Moderate momentum — gradual convergence or structured document |
| 0–0.15 | No memory — rapid phase switching window-to-window |
| negative | Anti-persistent — entropy alternates high/low every window (oscillating loop) |

A near-zero autocorrelation with COLLAPSING-dominant phases reflects the *sentence
rhythm* of factual generation: low-H clause endings alternate with higher-H connective
words.  It is not a sign of instability.

#### Win-win Jaccard trajectory

While the token-to-token Jaccard compares the top-20 candidate sets of *individual
adjacent windows*, the win-win Jaccard compares the *most-frequent tokens aggregated
across entire windows*.  It measures whether the same token distribution is recycling.

Values above 0.45 (sustained) indicate the same candidate set is looping window-to-window
— a strong cycling signal.  Values below 0.28 in a COLLAPSING-dominant run support the
HIGH-CONF interpretation (confident but moving forward).

The trajectory is printed as a row of per-window values, colour-coded red (≥ 0.55) and
yellow (≥ 0.40):

```
tok ~ 10–100   --  0.33  0.33  0.21  0.25  0.43  0.33  0.48  0.33  0.25
tok ~110–200  0.18  0.33  0.25  0.29  0.33  0.21  0.21  0.33  0.54  0.43
```

---

### Early warning signals

Chronological per-window events with look-ahead confirmation (a signal only fires if
the *next* window supports the same direction):

| Icon | Meaning |
|------|---------|
| `[!]` | High-severity warning — action recommended |
| `[~]` | Note — monitor closely |
| `[PRED]` | End-of-run prediction about the trajectory |

Key signals:
- **NORMAL/HIGH-H → COLLAPSING** — distribution tightening; loop may form if it
  continues downward.
- **COLLAPSING → LOCK** — single-token attractor reached.
- **LOCK → COLLAPSING/NORMAL** — lock eased; still narrow distribution.
- **win-Jaccard crossed 0.45** — same candidate-set cycling window-to-window.
- **top-1 ≥ 95% for N windows** — near-deterministic; loop likely.
- **top-1 ≥ 88% for N windows** — sustained overconfidence; repetition risk.

---

### Regional timeline

The run is divided into segments of approximately 5 windows each.  Every segment gets a
label, mean entropy, phase composition, and win-win Jaccard:

```
[!] tok 101–150  H=0.60  COL×3  NOR×2  wjk=0.17  → COLLAPSING
```

| Icon | Segment label |
|------|--------------|
| `[!]` | LOCK, COLLAPSING, or CYCLING — structural issue |
| `[~]` | CHAOS, HIGH-H, or MIXED — elevated uncertainty |
| `[ ]` | NORMAL or HIGH-CONF — healthy |

**Segment label logic (in priority order):**

1. **LOCK** — ≥40% of windows in LOCK zone
2. **CYCLING** — ≥60% in LOCK+COLLAPSING *and* win-win Jaccard > 0.35
3. **HIGH-CONF** — ≥60% in LOCK+COLLAPSING *and* win-win Jaccard < 0.28
4. **COLLAPSING** — ≥60% in LOCK+COLLAPSING (Jaccard 0.28–0.35, ambiguous)
5. **CHAOS / HIGH-H / NORMAL** — dominant zone ≥40%/50%
6. **MIXED** — no dominant zone

The **Overall** line derives from the segment sequence without any global-average
dilution (which would mask late-onset collapses):

| Verdict | Condition |
|---------|-----------|
| SUSTAINED LOOP | All segments in LOCK/COLLAPSING/CYCLING |
| LATE-ONSET COLLAPSE | First half clean, second half ≥50% bad segments |
| DETERIORATING | More bad segments in second half than first |
| RECOVERED | Bad segments in first half, none in second |
| CLEAN | All segments NORMAL / HIGH-H / HIGH-CONF |
| MIXED | Otherwise |

---

### Health scorecard

A set of threshold checks printed with `[OK]`, `[NOTE]`, or `[WARN]`:

| Check | Fires when |
|-------|-----------|
| Median entropy | < 0.2 (very tight) or > 3.0 (very high) |
| Sustained chaos start | Longest CHAOS streak ≥ 5 in first 20% of run |
| Chaos fraction | > 50% WARN; > 10% NOTE |
| H autocorrelation | < 0.15 with ≥ 3 isolated CHAOS spikes → erratic; or < 0.15 with < 55% NORMAL |
| LOCK fraction | > 35% WARN; > 10% NOTE |
| Cycling / converging | Pattern flags (see above) |
| Late-onset collapse | First-half H / second-half H ratio and tight-second-half fraction |
| VH/H ratio | > 6.0 WARN (heavy tail); > 3.0 NOTE |
| Max logit std-dev | > 30 WARN (numerical overflow risk) |
| Token-to-token Jaccard | > 0.75 WARN (same top-20 every window) |

---

## Metrics reference

All metrics are computed per token, then averaged over a window (default 10 tokens).

### H — Shannon entropy (bits)

$$H = -\sum_{i} p_i \log_2 p_i$$

where $p_i = \text{softmax}(\text{logit}_i)$ over the full vocabulary.

**Range:** 0 (single token with probability 1) to log₂(vocab_size) ≈ 15–17 bits for
typical 32 k–128 k vocabulary models.

**What it measures:** the average number of bits needed to encode the model's next-token
prediction.  Low H = high certainty; high H = genuine uncertainty.

**Practical range for a healthy model:** roughly 0.3–3.0 bits depending on the content
type, temperature, and model size.

---

### VH — Varentropy

$$VH = \text{Var}\!\left(-\log_2 p_i\right) = \frac{1}{V}\sum_i \left(-\log_2 p_i - \overline{-\log_2 p_i}\right)^2$$

**What it measures:** the *spread* of the log-probability distribution.  While H
measures the average surprise, VH measures how uneven that surprise is across the
vocabulary.

**Interpretation:**
- Low VH with low H → tight, near-deterministic distribution (LOCK/COLLAPSING).
- High VH with moderate H → a few tokens carry most of the probability, the rest are
  near-zero (typical NORMAL/HIGH-H with a sharp peak and a long tail).
- Very high VH/H ratio (> 3) → heavy-tailed distribution; log-probability values are
  highly variable, which can indicate numerical instability or a very spiky softmax.

---

### top-1 probability

The softmax probability of the single highest-scoring token: $p_1 = \max_i p_i$.

**Interpretation:** directly related to H but easier to reason about in percentage
terms.  A top-1 probability above 0.95 (95%) for multiple consecutive windows
corresponds to near-LOCK behaviour and is a direct repetition risk signal.

---

### Effective vocabulary (eff vocab)

$$\text{eff\_vocab} = 2^H$$

The number of tokens a uniform distribution would need to match the same entropy.
Equivalently, the number of "meaningful choices" the model considers.

**Examples:**
- H = 0.0 → eff_vocab = 1.0 — only one real option
- H = 1.0 → eff_vocab = 2.0 — roughly two equally likely options
- H = 3.0 → eff_vocab ≈ 8 — eight options in play

---

### top-1/top-2 log margin

$$\text{margin} = \log_2(p_1) - \log_2(p_2) = \log_2\!\left(\frac{p_1}{p_2}\right)$$

**What it measures:** how decisively the top-1 token beats its nearest competitor.
Independent of the tail of the distribution.

**Interpretation:**
- margin ≈ 1 → top-1 is about twice as likely as top-2 (contested decision)
- margin > 5 → top-1 overwhelmingly dominates top-2 (highly confident single choice)
- margin > 10 → effectively only one option (deep LOCK territory)

---

### Token-to-token Jaccard (jaccard top-20)

$$J = \frac{|\text{top}_{20}(w) \cap \text{top}_{20}(w-1)|}{|\text{top}_{20}(w) \cup \text{top}_{20}(w-1)|}$$

Jaccard similarity between the top-20 highest-scoring token *indices* of consecutive
windows.

**What it measures:** how much the *candidate set* (the tokens under active
consideration) changes between adjacent windows.

**Interpretation:**
- High Jaccard (> 0.75) → the same 20 tokens are candidates in both windows — the
  model is stuck considering the same options.
- Low Jaccard (< 0.15) → the candidate set is changing rapidly — the content is
  genuinely evolving.
- Moderate Jaccard in a COLLAPSING-dominant run is ambiguous: it can mean either
  per-position diversity within a phrase cycle, or genuine topic variation.

**Pitfall:** long-phrase repetition loops (phrase length > window size) produce *low*
token-to-token Jaccard because consecutive windows capture different positions in the
cycle — the candidate sets differ even though the same phrase repeats.

---

### Win-win Jaccard

Jaccard similarity between the sets of most-frequently-occurring tokens *within* each
complete window (frequency-aggregated, not just comparing boundaries).

**What it measures:** whether the same token *distribution* recycles window-to-window.
Unlike token-to-token Jaccard, this is insensitive to phrase-internal position — if the
same phrase repeats, the same tokens are frequent regardless of where in the phrase each
window starts.

**Interpretation:**
- mean > 0.50 → strong cycling: the model is looping the same phrase or sentence.
- mean 0.28–0.50 → elevated: monitor transitions.
- mean < 0.28 in a COLLAPSING-dominant run → HIGH-CONF (confident but moving forward).

---

### H slope (OLS)

The gradient of the OLS regression line fitted to H values over token position.

Reported normalised as `h_slope / max(mean_H, 0.20)` (relative slope) for pattern
detection, but the raw value is displayed.

**Interpretation:**
- Flat (|slope| < 0.001/tok) → entropy is stable; model in a consistent state.
- Negative slope → entropy falling; model converging toward higher confidence over time.
- Positive slope → entropy rising; model becoming less certain (topic exhausted, context
  window pressure, or recovering from a loop).

---

### H autocorrelation (lag 1)

Pearson correlation between consecutive H values: $r = \text{corr}(H_i, H_{i+1})$.

Captures the *persistence* of the entropy signal:

| r | Pattern |
|---|---------|
| ≈ 1.0 | Smooth, monotone trajectory |
| 0.4–0.7 | Moderate inertia |
| 0–0.15 | Rapid switching (sentence rhythm, alternating phrases) |
| < 0 | Anti-persistent oscillation (commit → reconsider → commit) |

A near-zero autocorrelation is normal for factual generation (sentence endings are
COLLAPSING, connectives are NORMAL/HIGH-H).  A *negative* autocorrelation combined with
COLLAPSING-dominant phases is the fingerprint of an *oscillating loop* — the model
commits to a phrase, briefly reconsiders, then commits again.
