# `feedback_circuits_transformerlens_optimization_only.ipynb` — Developer Guide

> A `CLAUDE.md`-style reference for **one specific research notebook**:
> [`src/scripts/feedback_circuits_transformerlens_optimization_only.ipynb`](src/scripts/feedback_circuits_transformerlens_optimization_only.ipynb)
>
> 141 cells, ~660 KB of source. This file is the map you should read before editing or running any of it.
> Cell numbers throughout refer to this revision.

---

## 0. What This Notebook Is

This notebook studies **autoregressive language models as self-consistent dynamical systems**.

The central object is a **macroscopic state** `π` — a probability measure over the
*content of the context window* (either over single tokens, or over **K-grams**).
A transformer conditioned on a context whose empirical distribution is `π` induces a
**state-dependent Markov transition operator** `P_π` on next-states. The notebook asks:

> **What are the fixed points `π = π P_π`?**
> Those fixed points are the "attractors" of generation: repetition loops, topic lock-in,
> semantic basins, and degenerate babble modes.

Three complementary attacks on that question live in the notebook:

| Approach | Mechanism | Where |
|---|---|---|
| **A. Gradient fixed-point search** | Parameterize `π` on the simplex, minimize `JSD(π, πP_π)` by natural / exponentiated gradient descent | Cells 52, 54, 57, 60, 68 |
| **B. Damped fixed-point iteration** | Just iterate `π ← α·π + (1−α)·π P_π` | Cell 57 |
| **C. Spectral / Markov-state-model analysis** | Explicitly discover the closed K-gram support, extract a sub-stochastic matrix, run SCC + PCCA+ + spectral decomposition on it | Cells 72–81 |

Approach **C** is the most mature and is the one the later sections of the notebook
actually exercise on trajectories.

A second, supporting theme is **thermodynamic head profiling**: measuring, per attention
head, the *positional baseline* `μ_d` and the *semantic spread* `σ_d` of pre-softmax
attention logits as a function of relative distance `d`. These statistics feed the
**3-part partition function** used by the abstract forward pass (see §2.3).

---

## 1. Architecture & Dependency Map

### 1.1 Cell-level narrative flow

```
[0–3]    Environment: nvidia-smi, pip/mamba installs, sys.path hack, global imports
[4–13]   DEFINITIONS ONLY — no side effects except function/class binding
  ├─ [5]   General helpers (PCA plotting, attention-pattern plotting)
  ├─ [7]   Activation patching / knockout hooks + similarity metrics
  ├─ [9]   "Optimization and Black Box Functions" — the 1-LAYER, TOKEN-LEVEL machinery
  ├─ [10]  THE BIG CELL (86 KB) — head profiling + the MULTI-LAYER, K-GRAM machinery
  ├─ [11]  MetastableKGramEngine, extract_substochastic_matrix, MetastableMSMAnalyzer
  └─ [13]  plot_head_thermodynamic_profile
[15]     ⚠ LOAD MODEL + `torch.set_grad_enabled(False)`  ← global state landmine
[16–49]  EXPLORATORY SCRATCH: generation, knockout, DLA, QK/OV matrices, stable tokens
[50–58]  Fixed Point Finder — 1-layer / token-level optimization (3 variants)
[59–60]  Multi-layer optimization — ⚠ LEGACY, calls the pre-refactor signatures
[61–66]  Timescale profiling (μ, σ, expected attention, receptive fields)
[67–70]  Multi-layer optimization — CURRENT, 3-part partition function
[71–78]  K-gram exploration → sub-stochastic matrix → MSM spectral + PCCA+ analysis
[79–82]  Sweep the MSM pipeline along a generated trajectory; QSD affinity matrix
[83–87]  Trajectory diagnostics: JSD loss at positional vs. non-positional fixed points
[88–106] Wide semantic head optimization — one head per GPU over a real corpus,
         + result inspection, branching factors, Jacobian stability
[107–111] APPROXIMATION QUALITY BENCHMARK — how close is the abstraction to the real model?
  ├─ [108] DEFINITIONS (85 KB): estimators, approximate passes, the benchmark harness,
  │        `DiscountedUnigramContextEstimator` and `MultiTimescaleMeanFieldForwardPass`
  ├─ [110] Example run: partition-function abstraction on a Wikipedia corpus
  └─ [111] Result inspection: error histograms, error-vs-position, worst-case dump
[112–115] EXACT 1-LAYER / SINGLE-HEAD / POSITION-FREE PASS — the abstraction's zero point
  ├─ [113] `ExactSemanticHeadForwardPass`
  └─ [115] Example run + ablation table (measured JSD ≈ 1.6e-12)
[116–121] EXACT SINGLE-SEMANTIC-HEAD FIXED-POINT SEARCH — the cell-68 loop driven by the
          exact parameterization, with `MetastableKGramEngine` as the exploration phase
[122–127] Jacobian / linear-stability analysis of a found fixed point
[128–131] FULL MULTI-HEAD, POSITION-ENABLED ABSTRACTION  (§2.11–2.13)
  ├─ [129] Timescale profiling: measure E_pos(d) per head, fit γ, build `mtmf_full`
  └─ [131] Approximation quality + 10 feature ablations against the UNTOUCHED model
[132–135] The same two cells on `tiny-stories-1L-21M` (16 heads, an MLP, attn_scale = 1)
[136–140] WIDE MULTI-TIMESCALE MEAN-FIELD OPTIMIZATION  (§2.14, §6)
  ├─ [137] The sweep: one explore-then-optimize fixed-point search per (text, position)
  ├─ [138] Save `wide_mtmf_results` + its config to `results/wide_optimizations/*.pt`
  └─ [140] Single-run diagnostics: 8-panel figure + the headline numbers
```

Cell numbering is as of this revision of the notebook (141 cells). Cells 88–106 (the
token-level wide optimization) are only sketched here; cells 107–111 are documented in
full in §2.9 and §5, cells 112–115 in §2.10, cells 128–135 in §2.11–§2.13, and cells
136–140 in §2.14 and §6.

### 1.2 Dependency tree — Pipeline C (K-gram / multi-layer, the current one)

```
MetastableKGramEngine                              [cell 11]
├── __init__(model, k_value, frozen_context_{vals,keys}, forward_pass_kwargs, ...)
├── run_pruned_power_iteration(initial_kgrams, ...)        ← ENTRY POINT
│   └── abstract_forward_pass(...)                 [cell 10]   (called per minibatch)
│       └── single_layer_forward(...)  × n_layers  [cell 10]
│           ├── model.blocks[l].ln1.w              (γ only; σ is frozen)
│           ├── model.W_{Q,K,V,O}[l, active_heads] + b_{Q,K,V}, b_O
│           ├── sink_k[l], sink_v[l]               ← extract_bos_sink
│           ├── ln_avg_sigma_list[l]               ← extract_frozen_sigma
│           ├── L_ctx_list[l], C_far_list[l]       ← profile_thermodynamic_heads
│           │                                        / estimate_L_ctx_for_head
│           │                                        / estimate_c_far_for_head
│           └── model.blocks[l].mlp (if present)
└── run_dijkstra_support_discovery(initial_kgrams, ...)    ← ALTERNATE ENTRY POINT
    └── abstract_forward_pass(...)                 (same as above)

extract_substochastic_matrix(closed_support_keys, ...)     [cell 11]
└── abstract_forward_pass(...)
    → {"sparse_transition_matrix" (N+1 × N+1 COO), "canonical_support_keys",
       "sink_state_index", "leakage_per_state", "mean_leakage_rate"}

MetastableMSMAnalyzer(substochastic_matrix_dict)           [cell 11]
├── __init__  → torch COO → scipy CSR → Tarjan SCC → self.P_core_csr, self.core_keys
├── perform_spectral_decomposition(k_eigenvalues)
│   └── scipy ARPACK (spla.eigs) or dense LAPACK (la.eig) fallback
│       → {evals_complex, evecs_right_complex, lambda_quasi, r_quasi_positive,
│          qsd_distribution, basin_escape_rate, implied_timescales_tokens, spectral_gaps}
└── run_conditioned_pcca(spectral_results, num_clusters)
    ├── Doob h-transform  P_cond = (1/λ)·R⁻¹ P R
    ├── deeptime msm.MarkovStateModel(reversible=False) → π
    ├── Adjoint  P* = Π⁻¹ P_condᵀ Π ; symmetrize  P_sym = ½(P_cond + P*)
    └── deeptime msm.MarkovStateModel(reversible=True).pcca(n_metastable_sets)
        → {pcca_memberships, crisp_clusters, stationary_measure_pi,
           macrostate_mass_proportions, core_support_keys}
```

### 1.3 Dependency tree — Pipeline A/B (token-level, 1 layer)

```
Optimization loop  [cells 52 / 54 / 57]                    ← ENTRY POINTS
├── pi_to_pi_P_one_layer_topk(pi, heads, model, top_tokens, T, p)   [cell 9]  ← training path
├── pi_to_pi_P_one_layer(pi, heads, model, chunk_size, T, p)        [cell 9]  ← validation path
│   └── both wrap the same inner `compute_P_chunk` closure
├── pi_to_P_one_layer(...)                                          [cell 9]  ← materializes sparse P
├── JSD(pi1, pi2)                                                   [cell 9]
├── pi_from_context / pi_t_from_context / context_from_pi           [cell 9]
└── loss_from_pi_t(pi_t, model, ...)                                [cell 9]
    └── pi_to_pi_P_one_layer_topk + JSD
```

### 1.4 Dependency tree — Gradient loop over the K-gram machinery

```
"EXPLORE-THEN-OPTIMIZE" loop  [cells 60 (legacy) / 68 (current)]
├── PHASE 1 (no grad): compute_stationary_distribution(...)   → discover new K-grams
├── union-support merge via torch.unique(dim=0) + scatter_
├── PHASE 2 (grad):    compute_stationary_distribution(...)   → differentiable π P_π
│   └── torch.utils.checkpoint(single_power_iteration_step, ..., use_reentrant=False) × M
│       └── single_power_iteration_step
│           └── abstract_forward_pass  → topk → key-shift → unique → scatter_add_ → topk
├── loss = 1e5 · compute_aligned_jsd(π, keys_π, πP, keys_πP) / vocab_size
├── PHASE 3: Sherman–Morrison natural gradient on the simplex + grad clipping
└── PHASE 4: prune to top-N_tracking, migrate (pi_vals, pi_keys) to next iteration
```

### 1.5 Profiling sub-graph (feeds `forward_pass_kwargs`)

```
extract_bos_sink(model)              → sink_k, sink_v        [n_layers, n_heads, d_head]
extract_frozen_sigma(model, tok, l)  → scalar σ_LN for layer l
profile_thermodynamic_heads(model, K_list, ...)   → (L_ctx_list, C_far_list)
estimate_L_ctx_for_head / estimate_c_far_for_head → per-head scalars (analytic alternative)

profile_deep_heads_empirical(model, corpus_tokens, ...)          ┐
profile_deep_heads_empirical_instruct(model, prompts, ...)       ├→ (μ, σ, μ_bos, σ_bos)
profile_deep_heads_multigpu(model_name, prompts, gpu_ids, ...)   ┘   [n_layers,n_heads,seq_len]
  └── _profiling_worker (joblib/loky, one model per GPU)
        ↓
compute_marginalized_expected_attention(μ,σ,μ_bos,σ_bos)   → a_bar      [L,H,S]
compute_unmarginalized_expected_attention(...)             → a_bar_T    [L,H,S,S]
compute_attention_mass_horizons(post_softmax, thresholds)  → {thr: idx}
compute_adversarial_influence_horizon(μ, σ, mult, thr)     → [L,H]
plot_head_thermodynamic_profile(μ[l,h], σ[l,h], l, h)

load_and_tokenize_continuous_corpus(model, ...)   → 1-D corpus tensor
prepare_intact_instruction_corpus(model, ...)     → [ensemble_size, seq_len]
```

### 1.6 Data-flow contract (shapes)

| Symbol | Shape | Meaning |
|---|---|---|
| `pi_vals` / `context_vals` | `[N]` | Probability mass, sums to 1 (the **one** exception is `ExactSemanticHeadForwardPass(mass_in_L_ctx=False)`, which deliberately passes raw counts — §2.10.4) |
| `pi_keys` / `context_keys` | `[N, S_init]` | Integer token IDs; each row is one K-gram |
| `S_init` | scalar | `sum(K_list) - len(K_list) + 1` — the exact input length needed to produce **one** output token through all layers |
| `P_active` | `[N_queries, d_vocab]` | Rows sum to 1 |
| `μ, σ, μ_bos, σ_bos` | `[n_layers, n_heads, seq_len]` | Pre-softmax logit stats vs. relative distance |
| `a_bar_T` | `[n_layers, n_heads, seq_len(T), seq_len(d)]` | Expected attention at absolute pos `T`, rel. distance `d` |
| `L_ctx_list[l]`, `C_far_list[l]` | `[n_heads]` | Integration horizon, far-field baseline mass |
| `sink_k_list[l]`, `sink_v_list[l]` | `[n_heads, d_head]` | Frozen BOS key/value |
| sparse `P` | `[N+1, N+1]` COO | Row `N` is the absorbing sink |
| `pi_vals`, `pi_keys` from a `ContextDistributionEstimator` | `[N_c]`, `[N_c, S_init]` | Same contract as `context_vals` / `context_keys` |
| `p_real`, `p_approx` (benchmark) | `[d_vocab]` | Dense, sum to 1, after the shared sampling transform |
| `context_vals` (MTMF) | `[N_c]` **or** `[H, N_c]` | Unigram mean field; **every row sums to 1**. `[N_c]` ties all heads (the optimization case), `[H, N_c]` is per head (the `predict` case) |
| `context_keys` (MTMF) | `[N_c]` or `[N_c, 1]` | **Unigram token ids, not K-grams.** `forward` raises if you hand it `[N_c, K]` |
| `query_keys` (MTMF) | `[N_q, K]` | The explicit fast windows; `S_init == K` for this class |
| `far_mass` | `[H]` | `Z_far,h`, the effective far-token count. Travels *with* `context_vals`; only the product `far_mass[h] * context_vals[h, c]` enters the partition function |
| `L_ctx`, `gamma` (MTMF) | `[H]` | Per **active** head, in `active_heads` order — slice the profiler output, do not index it by absolute head id |
| `nu_vals`, `nu_keys` (MTMF state) | `[N]`, `[N, K]` | The tracked K-gram measure; its unigram marginal is the mean field |

### 1.7 Dependency tree — Pipeline D (approximation quality benchmark)

```
ApproximationQualityBenchmark(model, approximate_pass, ...)   [cell 108]  ← ENTRY POINT
└── run(corpus: List[str])
    ├── tokenize(text)                → [n_tokens]  (BOS-prepended, chopped to max_tokens)
    ├── sample_positions(n_tokens)    → List[int]   (seeded random.Random)
    └── evaluate_position(text, tokens, position)
        ├── real_probs(tokens, position)                         ← GROUND TRUTH
        │   └── model(tokens[:position+1])[0, -1]
        │       └── apply_sampling_transform(logits, T, top_p)
        ├── approximate_pass.predict(tokens, position)           ← UNDER TEST
        │   ├── PartitionFunctionApproximateForwardPass          [cell 108]
        │   │   ├── context_estimator.estimate(tokens[:position+1], model, S_init)
        │   │   │   ├── UnigramContextEstimator  → pi_from_context    [cell 9]
        │   │   │   └── KGramContextEstimator    → get_kgram_distribution_from_tokens  [cell 9]
        │   │   ├── _build_query_keys(tokens, position)  → [1, S_init]
        │   │   └── abstract_forward_pass(...)                   [cell 10]
        │   └── ExactSemanticHeadForwardPass                     [cell 113]   ← §2.10
        │       ├── __init__: sigma_table / W_E_unit / extract_bos_sink (W_pos zeroed)
        │       ├── _far_field(context, query_keys) → far_counts [d_vocab], N_far
        │       │     n_far[v] = total[v] − alive_local[v] − [v == bos_id]
        │       ├── _zero_pos_embeddings()  /  _rescaled_embeddings(sigma_query)
        │       └── abstract_forward_pass(..., L_ctx=[N_far+K+1], C_far=[1.0],
        │                                  ln_avg_sigma=[sigma_{x_t}])  [cell 10]
        ├── jsd_dense(p_real, p_approx)      → JSD               [cell 9]
        └── pi_from_context(...)             → unigram control baseline
    ↓
    RunningMoments (Welford) × 6 metrics
    ↓
    {"per_text": [...], "statistics": {...}, "skipped": [...], "config": {...}}

summarize_benchmark(results)                                   [cell 108]  ← pretty printer
```

### 1.8 Dependency tree — Pipeline E (multi-timescale mean field, the full-model lane)

```
MultiTimescaleMeanFieldForwardPass(model, L_ctx, ...)      [cell 108]   ← ONE class, two roles
│
├── DIAGNOSTIC / SETUP  (cell 129)
│   ├── positional_profile(query_token, d_max)      → E_pos(d) per head   [H, d_max+1]
│   └── fit_gammas_from_profile(query_token, ...)   → γ per head          [H]
│         └── L_ctx_h = 1 / (1 − γ_h)               ← read off the model, not guessed
│
├── THE OPERATOR
│   └── forward(context_vals, context_keys, query_keys, far_mass, query_position)
│       └── _forward_chunk(...)                     → P_active [N_q, d_vocab]
│           ├── 1. FAST WINDOW  — real W_E + W_pos, real ln1, all 4 QK terms  (exact)
│           ├── 2. BOS SINK     — extract_bos_sink, full (semantic+positional) query
│           ├── 3. MEAN FIELD   — the only approximation
│           │     ├── _pair_sigma_outer / frozen_sigma       ← LayerNorm for a far token
│           │     ├── k_far, v_far from W_E[context_keys]
│           │     ├── log_W_far = log E_pos(K) + log far_mass
│           │     └── far_positional_value: + discount-weighted mean W_pos in the VALUE
│           ├── 4. Z = ΣM_local + ΣM_far + M_sink, max-shifted
│           └── 5. residual → MLP (if any) → ln_final → W_U → softmax/T → top-p (STE)
│
├── FIXED-POINT API  (cell 137)
│   ├── kgram_to_unigram(vals, keys, marginal)      → the mean field, differentiably
│   ├── power_iteration_step(...)                   → one ν → νP_π step on K-grams
│   └── stationary(ν, keys, N, pruning_K, M, ...)   ← drop-in for compute_stationary_distribution
│
└── BENCHMARK API  (cells 131, 135)
    └── predict(tokens, position)                   ← ApproximateForwardPass contract
        └── DiscountedUnigramContextEstimator.estimate_per_head(...)  [cell 108]
              → (context_vals [H, N_c], context_keys [N_c], far_mass [H])

DiscountedUnigramContextEstimator                          [cell 108]
├── estimate_per_head(tokens, model, K, gammas, dead_ids)  ← the per-head path
└── estimate(tokens, model, S_init)                        ← ContextDistributionEstimator
                                                             contract (single γ, lifts to
                                                             [N_c, S_init], DROPS far_mass)
```

---

## 2. Core Method Implementations & Variations

The notebook is explicitly a **version museum**. Several methods exist in 2–4 forms,
with earlier versions left in place (often commented out, directly above the new one).
Knowing which is which is the single most important thing for editing safely.

### 2.1 `π → πP` for the 1-layer token-level model — three variants (cell 9)

All three share an identical inner `compute_P_chunk` closure:
LN₁(W_E) → Q/K/V → `QK_weighted = π ⊙ exp(QK/√d_head)` (the π-weighting **is** the
macroscopic conditioning) → OV → `+ E_chunk` residual → `ln_final` → `W_U` → softmax/T
→ **top-p nucleus with a straight-through estimator**.

| Variant | Returns | Cost | Gradient quality | Use when |
|---|---|---|---|---|
| **`pi_to_P_one_layer`** | full `[V, V]` sparse COO transition matrix | `O(V²)` — chunked, gradient-checkpointed | Full | You need the *matrix itself* (spectra, powers, visualization). Almost never used in the loops; kept for inspection. |
| **`pi_to_pi_P_one_layer`** | `[V]` dense vector `πP` | `O(V²)` FLOPs but `O(V·chunk)` memory; checkpointed per chunk | Exact | **Validation / ground truth.** Used at `i % val_iterations == 0` to compute `loss_full`. |
| **`pi_to_pi_P_one_layer_topk`** | `[V]` dense vector `πP` | `O(chunk_size · V)`, single un-checkpointed pass | Biased — only the top-`chunk_size` tokens act as queries | **Training.** This is what every gradient step actually calls. |

**Why the difference matters.** `pi_to_pi_P_one_layer` loops over the *entire* vocabulary
as queries — for a 48k–128k vocab that is hundreds of chunks per step, far too slow to put
inside a 3000-iteration loop. `_topk` exploits the fact that `π` is extremely sparse
in practice: only tokens with appreciable mass contribute meaningfully to `πP`, so it
restricts the query set to `pi.topk(chunk_size).indices`. The cost is that the loss
landscape is **discontinuous whenever the top-k membership changes**, and the gradient
w.r.t. tokens outside the top-k is exactly zero. The notebook mitigates this by tracking
both `losses` (biased) and `full_losses` (exact) on the same plot — if the two curves
diverge, the top-k truncation is the culprit.

`pi_to_pi_P_one_layer_topk` also deliberately **disables gradient checkpointing**
(the `checkpoint(...)` call is commented out) because with only `chunk_size` queries the
activations fit in VRAM and the recompute would be pure overhead.

**The STE trick.** All three do:
```python
chunk_probs = chunk_probs_sparse.detach() - chunk_probs_dense.detach() + chunk_probs_dense
```
Forward value = the top-p-filtered sparse distribution; backward gradient = the dense
softmax. Without this, nucleus filtering zeroes the gradient for every token outside the
nucleus and optimization stalls immediately. An `entmax15` alternative is present but
commented out in every copy.

### 2.2 Optimizers for the simplex — three variants (cells 52, 54, 57)

All three minimize the *same* objective, `1e5 · JSD(π, πP_π)` (sometimes `/ vocab_size`),
and all three log `losses`, `full_losses`, `pi_list`, `pi_P_list`.

| Cell | Parameterization | Update rule | Notes |
|---|---|---|---|
| **52** | `pi_logits = nn.Parameter(log π)`, `π = softmax(pi_logits)` | SGD + `CosineAnnealingLR`, gradient replaced by an **exact natural gradient** via Sherman–Morrison, then clipped to norm 1.0 | The canonical version. `heads = [0,3]`, `chunk_size = 64`, `lr = 1e0`. |
| **54** | `pi = nn.Parameter(π)` directly | **Exponentiated gradient**: `π ← softmax(log(π + 1e-12) − lr·g)`, done manually under `no_grad`, `pi.grad.zero_()` by hand | No optimizer object at all. `lr = 1e-2`, 5000 iters. Naturally stays on the simplex without a natural-gradient correction. |
| **57** | none — `pi` is a plain tensor | **Damped Picard iteration**: `π ← α·π + (1−α)·πP`, `α = 0.995`, everything under `no_grad` | Not optimization at all; a fixed-point *solver*. Cheapest, no autograd, but only converges to *stable* fixed points and can't escape a basin. |

**Why three?** The natural-gradient version (52) is the mathematically principled one
— the Fisher metric on the simplex is `diag(1/p)` plus a rank-1 correction, and
Sherman–Morrison inverts that in closed form:

```
A⁻¹ = 1/(π + λ)
natural_grad = A⁻¹g + (A⁻¹π · πᵀA⁻¹g) / (1 − πᵀA⁻¹π)
```

The damping `λ = 1e-10` is load-bearing: `π` is sparse, so `1/π` explodes for dead tokens.
Exponentiated GD (54) achieves a similar mirror-descent geometry *implicitly* and is far
more numerically forgiving, at the cost of a less well-scaled step. Picard (57) tells you
where the dynamics *actually go* rather than where the loss is minimized — use it to
sanity-check whether a fixed point found by 52/54 is attracting or repelling.

### 2.3 `single_layer_forward` — old vs. new (cell 10)

Both versions are in the cell; the old one is fully commented out above the live one.

**Old (2-part: semantic heads + positional heads)**
- Heads are partitioned into `sem_heads_list` / `pos_heads_list` per layer.
- Semantic heads attend only to the **final token of each frozen context K-gram**, weighted by `context_vals`.
- Positional heads attend over a local `K_i` sliding window with a standard `softmax` and `-1e4` masking.
- Positional embeddings were added into the residual stream (actually commented out, leaving them absent).
- MLP bypassed.

**New (3-part thermodynamic partition, unified head set)**
- No sem/pos split — one `active_heads` list per layer.
- The residual stream carries **pure semantics only**; positional embeddings are kept in a
  *separate stream* and only enter through the QK positional score.
- LayerNorm is **linearized** ("the freeze hack"): centering is applied exactly, but the
  per-token σ is replaced by a frozen constant `ln_avg_sigma` measured once via
  `extract_frozen_sigma`. Learned γ (`ln1.w`) is still applied. This makes the map linear
  in the semantic stream and lets semantic and positional contributions be separated.
- The attention denominator becomes an explicit **3-term partition function**:

```
M_global = (L_ctx − K_i − 1) · π · exp(q_s·k_s_global/√d) · C_far      # the far background
M_local  = exp(q_s·k_s_local/√d) · exp(q_p·k_p_local/√d) · window_mask  # explicit K-gram
M_sink   = exp(q_s·sink_k/√d)                                          # the BOS attention sink
Z        = ΣM_global + ΣM_local + M_sink
attn_out = (out_global + out_local + out_sink) / Z
```

**Why this matters.** The old version had no way to represent "the rest of the context":
the whole rest of the sequence was either in the `K_i` window or invisible. `M_global`
reintroduces it as a *mean-field* term — `L_ctx − K_i − 1` counts how many far tokens a
head effectively integrates over, `C_far` is the average exponentiated positional mass of
such a token, and `π` weights them by the macroscopic density. That is precisely what makes
the operator `P_π` **state-dependent** and gives the whole fixed-point story its content.
`M_sink` is separated out because the BOS sink dominates raw attention and would otherwise
poison both the local window and the far-field statistics.

The new version also adds **unified dead-token masking**: both `pad_token_id` and
`bos_token_id` occurrences are wiped from `M_local`, forcing all BOS mass to be resolved
by the dedicated `M_sink` term instead of being double-counted.

Two **dummy positional embedding** patches are marked in-line with `### ADDED`:
`ln2(shrunken_resid + pos_embeds_q)` before the MLP, and `ln_final(resid + pos_embeds_q)`
before unembedding. These exist because LayerNorm statistics of a purely-semantic residual
are out of distribution relative to what the model saw in training.

**Shrinking pyramid.** Every layer consumes `S_in` positions and emits `S_out = S_in − K_i + 1`.
Hence `S_init = sum(K_list) − len(K_list) + 1` is exactly the input length that collapses
to a single output position after the last layer. Change `K_list` and `S_init` must be
recomputed — the notebook does this in every setup cell.

### 2.4 `abstract_forward_pass`, `single_power_iteration_step`, `compute_stationary_distribution` — old vs. new

Each of these has a commented-out predecessor immediately above it. The **signatures changed
incompatibly** in the refactor:

```python
# OLD
compute_stationary_distribution(pi_vals, pi_keys, N, pruning_K, model,
                                K_list, sem_heads_list, pos_heads_list,
                                M, temperature=1.0, top_p=1.0)
# NEW
compute_stationary_distribution(pi_vals, pi_keys, N, pruning_K, model,
                                M, K_list, active_heads_list, ln_avg_sigma_list,
                                L_ctx_list, C_far_list, sink_k_list, sink_v_list,
                                temperature=1.0, top_p=1.0)
```

Note `M` moved from position 9 to position 6. **Cell 60 still calls the old positional
signature** — see §4.3.

`single_power_iteration_step` itself is unchanged in logic between versions:
`topk` per query → shift keys left and append the new token → flatten →
`torch.unique(dim=0)` + `scatter_add_` (differentiable aggregation of converging paths) →
global `topk(N)` → renormalize. `compute_stationary_distribution` wraps each of the `M`
steps in `torch.utils.checkpoint(..., use_reentrant=False)`; `use_reentrant=False` is
required so autograd can route gradients through the float tensors while ignoring the
integer key tensors.

### 2.5 Support discovery — power iteration vs. Dijkstra (cell 11)

`MetastableKGramEngine` offers two ways to find the closed K-gram support. They answer
different questions and are **not** interchangeable.

| | `run_pruned_power_iteration` | `run_dijkstra_support_discovery` |
|---|---|---|
| **Criterion** | Keeps a successor if its **fully aggregated flux** `Σ_parents p(u)·P(v|u) ≥ ε_power` | Keeps a successor if **cumulative surprisal** `Σ −log P ≤ S_max = −log(ε_dijkstra · N_init)` |
| **Semantics** | "What states carry real probability mass *right now*?" | "What states are reachable along any sufficiently likely path?" |
| **Halting** | `frontier_rate < threshold` **and** `leakage_rate < threshold` for `min_stable_steps` consecutive iterations | Heap exhausted / frontier action exceeds `S_max` — **exact**, no convergence heuristic |
| **Output extras** | `stationary_measure`, `history_{internal,frontier,leakage}_rates`, `is_converged` | `support_actions_nats` (min surprisal from seed), `total_forward_passes` |
| **Bias** | Depends on the seed measure and the number of iterations | Independent of measure; purely topological on the action manifold |
| **Cost** | Bounded per iteration by `suffix_chunk_size × vocab_size` VRAM | Unbounded — can explode combinatorially if `ε_dijkstra` is too small |

**Key implementation detail in the power iteration: post-aggregation pruning.**
The naive approach prunes each `p(u)·P(v|u)` edge before summing, which destroys states
reached by many weak paths. Here, active queries are first **sorted by their `K−1` suffix**
so that all parents converging on the same successor prefix are adjacent; then binary
search (`torch.searchsorted`) slices out exactly the parents belonging to each suffix
chunk; then a dense `[chunk, vocab]` accumulator is filled with `scatter_add_`. The
`ε_power` guillotine is applied **only after** the accumulator holds the exact total flux.
This also means the surviving successors are **guaranteed unique**, so no coalescing pass
is needed afterward. `suffix_chunk_size = 2048` is sized to keep that accumulator near
~411 MB — raise/lower it with VRAM, not with correctness in mind.

The **3-part thermodynamic audit** (internal / frontier / leakage rates) printed each
iteration is the diagnostic to watch:
- high **frontier** rate → the support is still growing, keep iterating;
- high **leakage** rate → `ε_power` is too aggressive, you are dissipating real mass;
- high **internal** rate + low other two → converged, metastable set isolated.

### 2.6 Head profiling — four variants (cell 10)

| Function | Input | Isolates | Cost |
|---|---|---|---|
| `estimate_c_far_for_head` / `estimate_L_ctx_for_head` | none (weights only) | **Pure positional** geometry: takes `W_pos`, linearizes LN, projects through this head's `W_Q/W_K` with **no biases** (they are treated as part of the semantic stream), and looks at `exp(score)` per diagonal | Instant, per head |
| `profile_thermodynamic_heads` | random token ensemble | **Post-softmax** mean attention pattern over uniform-random prompts, so `E[semantic] ≈ 0`. Computes `E(d)` per diagonal, excludes `k=0`, thresholds at `μ_far + 10σ_far` | `num_prompts` forward passes, `[n_heads, S, S]` accumulator per layer |
| `profile_deep_heads_empirical` | flattened real corpus | **Pre-softmax** `μ_d` and `σ_d` over a real text distribution, via running sum / sum-of-squares in **float64** | `ensemble_size / batch_size` passes |
| `profile_deep_heads_empirical_instruct` | pre-tokenized chat tensor | Same, but on **intact chat-templated** sequences (preserves system prompts, BOS, role headers) | Same, plus periodic `gc.collect()` |
| `profile_deep_heads_multigpu` | pre-tokenized tensor + `gpu_ids` | Same as `_instruct`, fanned out with `joblib`/`loky`; each worker loads its **own bfloat16 copy** of the model and returns **raw FP64 accumulators** for a global map-reduce | `ensemble_size / (num_gpus · batch)` |

**Why the analytic and empirical variants both exist.** The analytic pair
(`estimate_*_for_head`) needs no data and isolates positional structure perfectly, but it
ignores the fact that real token distributions shift the effective horizon.
`profile_thermodynamic_heads` averages over random tokens to cancel semantics — cheap,
returns exactly the `(L_ctx_list, C_far_list)` pair that `single_layer_forward` consumes,
and is what the live setup cells (68, 72) actually call. The `profile_deep_heads_*` family
returns the richer `(μ, σ, μ_bos, σ_bos)` used by the *analysis* side (expected attention,
receptive fields, adversarial horizons) rather than by the forward pass.

**The BOS guillotine** appears in every empirical profiler: for each sub-diagonal
`torch.diagonal(scores, offset=-d)`, element `[..., 0]` is always the query-to-BOS score
(because `q − k = d` and `k = 0` ⟺ `q = d`). It is recorded separately into the `*_bos`
accumulators and then **sliced out** (`diag[..., 1:]`) from the content statistics.
Without this, the sink's enormous logit inflates `σ_d` at every distance and the whole
log-normal MGF downstream becomes garbage.

The multi-GPU worker uses `return_type=None` in `run_with_cache` to skip the vocabulary
projection entirely — worth >1 GB of transient VRAM on a 128k-vocab model — and keeps its
accumulators on **CPU** to avoid VRAM creep over long runs.

### 2.7 Expected attention — marginalized vs. unmarginalized (cell 10)

Both convert `(μ, σ)` into expected attention via the **log-normal MGF**,
`E[exp(X)] = exp(μ + σ²/2)`:

- `compute_marginalized_expected_attention` → `a_bar[l, h, d]`: averages the conditional
  probability `M_content(d)/Z_T` over every valid query position `T > d`. A single curve
  per head; use it for "what does this head's attention profile look like on average".
- `compute_unmarginalized_expected_attention` → `a_bar_T[l, h, T, d]`: the full matrix,
  built by outer-broadcasting `M_content.unsqueeze(-2) / Z.unsqueeze(-1)`, masked with
  `tril(diagonal=-1)`, plus BOS mass placed on the main diagonal via `diag_embed`
  (for query `T`, BOS sits at relative distance exactly `d = T`). Use it when the horizon
  depends on absolute position — e.g. cell 64's `a_bar_T[:, :, 400]`.

The marginalized version uses a Python loop over `T`; the unmarginalized one is fully
vectorized but costs `[L, H, S, S]` memory (for `S = 1024`, 16 layers, 32 heads that is
~2 GB in fp32 — a real constraint).

### 2.8 `compute_aligned_jsd` — defined twice

Cell 9 and cell 60 both define it. The cell-60 version is identical **except** it appends
to a module-level `N_union_list`, which cells 69 and 79/81 read. Since cell 60 executes
after cell 9, the cell-60 definition wins — but only if cell 60 has been run. If you run
cells 68/72/79 on a fresh kernel without running cell 60, `N_union_list` will never be
populated and cell 69's second plot will be empty.

Mechanism: concatenate both key tensors, `torch.unique(dim=0, return_inverse=True)` to
build the union support, `scatter_` both value vectors onto it with `1e-10` for missing
keys, renormalize, then call `JSD`. This is what allows the loss to compare two sparse
distributions whose supports genuinely differ — and it is *why* the optimization can
discover new K-grams at all: a newly emitted key gets `1e-10` mass in `π` and real mass in
`πP`, producing a large gradient that pulls `π` toward it.

### 2.9 The approximation quality benchmark (cells 107–111)

Everything else in this notebook *assumes* the abstraction (`abstract_forward_pass`) is a
faithful stand-in for the concrete model. Cells 107–111 are the experiment that **measures
that assumption**. The question is deliberately narrow:

> Given a real text and a position `t` inside it, how far is the abstraction's next-token
> distribution from the one the actual transformer produces?

with the distance being `JSD(p_real, p_approx)`, averaged over a corpus.

#### 2.9.1 The measurement, end to end

For one text:

```
text  ──to_tokens(prepend_bos=True)[:max_tokens]──▶  tokens [n_tokens]
      ──rng.randrange(min_position, n_tokens)─────▶  position t

REAL:    model(tokens[:t+1]) → logits[0, -1]  ──apply_sampling_transform──▶ p_real   [d_vocab]
APPROX:  estimator(tokens[:t+1])              ─────────────────────────────▶ (pi_vals, pi_keys)
         query = tokens[t-S_init+1 : t+1]     ─────────────────────────────▶ query_keys [1, S_init]
         abstract_forward_pass(pi, query)     ─────────────────────────────▶ p_approx [d_vocab]

error = jsd_dense(p_real, p_approx)
```

Both sides predict the token at `t+1`; the token at `t` is the last one either side sees.
The ground truth is the *full* concrete forward pass over the whole `t+1`-token prefix —
all heads, all positions, real LayerNorm, real positional embeddings — so the JSD absorbs
**every** approximation at once: the mean-field `M_global` term, the frozen σ, the
truncated `K_i` window, the head selection, the unigram (or K-gram) state estimate. It is
a scalar verdict on the abstraction as a whole, not an ablation of its parts. If you want
to attribute the error, run the benchmark twice with one ingredient changed (see §5.4).

#### 2.9.2 Design decisions, and why

**Both sides go through the same sampling transform.** `abstract_forward_pass` applies
temperature and top-p internally and returns probabilities, not logits. If the real side
skipped that, the JSD would be dominated by the transform rather than by the
approximation. `apply_sampling_transform(logits, temperature, top_p)` was factored out of
the tail of `abstract_forward_pass` for exactly this reason and is called by both paths.
It is the same nucleus code (sort → cumsum → shift-right-by-one → scatter → renormalize)
**minus the straight-through estimator**, which only exists to keep gradients alive and is
meaningless under `no_grad`.

`ApproximationQualityBenchmark` defaults its `temperature` / `top_p` to whatever the
approximate pass carries, so the two sides cannot silently drift apart. **Keep
`top_p = 1.0` for a pure approximation-quality measurement**: any `top_p < 1` truncates
*both* distributions to a small shared support and flatters the JSD, because the two
nuclei overlap heavily even when the full distributions do not.

**There is a baseline, because a bare JSD is uninterpretable.** JSD in nats is bounded by
`ln 2 ≈ 0.6931`; hitting that bound means the two supports are disjoint. Every result
therefore also carries `jsd_baseline_context_unigram` = `JSD(p_real, pi_from_context(...))`
— the divergence between the true next-token distribution and the raw context token
histogram. That is the "we did not use the model at all" control. An abstraction that does
not beat it is not carrying any information about the transformer. On `attn-only-1l` with
`K_list=[1]`, `active_heads_list=[[3]]` a smoke run gave ≈ 0.38 nats against a ≈ 0.56 nat
baseline: better than nothing, and nowhere near good — which is the correct reading for a
one-head, one-token-window abstraction.

**Statistics are streaming (Welford), not stored-then-reduced.** `RunningMoments` keeps
`count / mean / M2 / min / max` and exposes `var`, `std`, and `sem`. The mean is the
headline, but `sem = std/√n` is the bar that belongs on it — with 50 texts the standard
deviation across texts says how heterogeneous the corpus is, while the SEM says how well
you have pinned down the corpus average. Non-finite values are silently dropped by
`update`, so one NaN cannot poison the run. `statistics["mean_loss"]` and
`statistics["std_loss"]` are aliases of the `loss_jsd` entries, named as the headline
numbers.

**JSD is computed in float64 with clamping.** The notebook's `JSD` takes `.log()` of both
arguments and uses `reduction='batchmean'`, so it needs a batch dimension and strictly
positive inputs. Top-p filtering produces *exact* zeros, and even without it fp32
softmax tails underflow. `jsd_dense(p, q)` therefore reshapes to `[1, d_vocab]`, casts to
double, clamps at `1e-15`, renormalizes, and only then calls `JSD`. Result is in **nats**.

**Per-position failures are isolated.** `run` wraps `evaluate_position` in a try/except and
pushes the repr of any exception into `results["skipped"]`. A 500-text corpus run is long
enough that one pathological text must not destroy it. Texts shorter than
`min_position + 1` tokens are also recorded in `skipped`, with the reason spelled out,
rather than silently dropped — otherwise `n_evaluated` quietly disagrees with `len(corpus)`
and you never learn why.

**The RNG is seeded per run, not per text.** `random.Random(self.seed)` is created inside
`run`, so two different approximations benchmarked with the same `seed`, `corpus`,
`max_tokens`, `min_position` and `n_positions` see **identical positions**. This is what
makes A/B comparisons meaningful; the commented-out sweep at the bottom of cell 111 relies
on it. Changing any of those five knobs changes the sampled positions.

**`max_tokens` is clamped to `model.cfg.n_ctx`.** Feeding a longer prefix to the concrete
model would index past `W_pos` and crash. The constructor also refuses
`min_position >= max_tokens` up front, with the clamped value in the message, because the
failure mode otherwise is "every text skipped, no explanation".

**Storage is a knob.** `store_probs="full"` keeps both `[d_vocab]` vectors per position on
**CPU** (≈ 2 × 48262 × 4 B ≈ 386 KB per position — 50 texts is ~19 MB, fine; 5000 is not).
`"topk"` keeps `(indices, values)` pairs of length `store_topk`, `"none"` keeps neither.
The estimated π is always kept, in sparse `(context_vals, context_keys)` form, because it
is small and it is the thing you will want to look at when a position goes wrong.

#### 2.9.3 Auxiliary metrics

JSD is the requested error, but it is a single number over a 48k-dimensional object.
Each result also carries, at negligible cost:

| Field | Meaning / why |
|---|---|
| `total_variation` | `½‖p_real − p_approx‖₁`. Linear rather than log — insensitive to the tail, where JSD spends a lot of its budget. |
| `top1_agreement` | Do both sides argmax to the same token? The only metric that speaks to *generation* behaviour rather than distributional fit. |
| `real_top1_rank_in_approx` | Where the true argmax lands in the approximation's ranking. Distinguishes "approximation is diffuse but ordered correctly" from "approximation is confidently wrong". |
| `real_entropy_nats` / `approx_entropy_nats` | The fastest way to see whether the abstraction is systematically over- or under-confident. Large JSD with matching entropies means *misplaced* mass; large JSD with an entropy gap means the partition function `Z` is mis-scaled. |
| `n_context_states` | `N_c` — how many rows the estimator produced. Drives both cost and fidelity. |

#### 2.9.4 `UnigramContextEstimator` and the K-gram lift

`abstract_forward_pass` requires context keys of shape `[N_c, S_init]`, but a unigram
state is a distribution over *single tokens*. The lift used here is the cheap one: each
surviving token id becomes one row whose **last column** is that token and whose earlier
`S_init − 1` columns are `fill_token_id` (the tokenizer's PAD, else BOS, else 0).

**For a 1-layer model this is exact.** `single_layer_forward` builds the global background
from `sem_resid_pre[:N_c, -1, :]` — the final column only — so the filler is never read.
The context rows' own `M_local` is computed and then discarded, costing a little compute
and changing nothing.

**For a multi-layer model it is not.** Layer 0 emits a shrunken context residual that
layer 1 consumes, and that residual *does* depend on the filler columns. PAD/BOS are
wiped from `M_local` by the unified dead-token mask, so each context row becomes an
isolated token with no local neighbourhood — a degenerate K-gram. Measured on
`attn-only-2l` with `K_list=[3,3]`, the unigram estimator returned **exactly `ln 2`**
(disjoint supports — the abstraction had collapsed), while `KGramContextEstimator` on the
same positions returned ≈ 0.325. **Use `KGramContextEstimator` for `n_layers > 1`**, or
design a better lift; the unigram class documents this in its own docstring.

Other knobs on `UnigramContextEstimator`:

- `context_window` — count only the last *n* tokens. This is how you test a finite sliding
  window hypothesis for the *state estimator* without touching the forward pass.
- `exclude_special` (default `True`) — BOS/EOS/PAD are dropped from the histogram. They are
  already represented by the dedicated `M_sink` term; counting them in π double-counts the
  sink and inflates `Z_global`.
- `top_n` — keep only the highest-mass tokens. The global term costs
  `O(heads · N_q · S_out · N_c)` in both time and VRAM, so this is the memory knob.
- `min_mass` — mass floor, an alternative to `top_n`.

`pi_from_context` is always called with `vocab_size=model.cfg.d_vocab` explicitly; its
default of `48262` only coincidentally matches `attn-only-1l` and is wrong for every other
model in the notebook (see §4.2).

`KGramContextEstimator` wraps `get_kgram_distribution_from_tokens` with `K = S_init` and
`offset = S_init − 1`, which keeps it out of that function's CPU-only left-padding branch
(§4.4), and moves the result to the model device itself.

#### 2.9.5 Query construction

`_build_query_keys` returns `tokens[position+1-S_init : position+1]` as `[1, S_init]`. When
the prefix is shorter than `S_init` it left-pads with the tokenizer's PAD id — which
`single_layer_forward` then masks out of `M_local`, so short prefixes degrade gracefully
instead of reading garbage. In practice `min_position ≫ S_init` and this branch never
fires; it exists so the class does not break when someone lowers `min_position`.

#### 2.9.6 Constructor validation — and the silent bug it catches

`PartitionFunctionApproximateForwardPass.__init__` validates `forward_pass_kwargs` against
the exact nine-key contract (missing *and* unexpected keys both raise, because the dict is
splatted with `**` and a typo otherwise becomes an opaque `TypeError` deep inside the
forward pass), and then checks that `L_ctx_list[l]` and `C_far_list[l]` have either
`len(active_heads_list[l])` entries or exactly 1 (a broadcast scalar, which several
commented-out configs in cell 72 use deliberately).

That second check exists because of a **real, silent corruption** in the existing pipeline.
`profile_thermodynamic_heads` returns one entry per head **in the model**, but
`single_layer_forward` indexes `L_ctx` / `C_far` by position **within `active_heads`**:

```python
L_global  = (L_ctx - K_i - 1).view(-1, 1, 1, 1)   # [n_heads_profiled, 1, 1, 1]
M_global  = L_global * pi_view * E_sem_global * C_far_view
#                                ^ [n_active_heads, batch, S_out, N_c]
```

With `n_active_heads == 2` and 8 profiled entries this raises a shape error. With
`n_active_heads == 1` it **broadcasts silently to 8**, the downstream einsum sums over
that phantom head axis, and the abstraction computes the same head eight times with eight
different integration horizons. Verified on `attn-only-1l`: `active_heads=[3]` with an
unsliced 8-entry `L_ctx` returns a perfectly normalized, perfectly wrong `[1, d_vocab]`.

Cell 72's live TinyStories/attn-only config (`active_heads_list = [[3]]` with
`L_ctx_list, C_far_list = profile_thermodynamic_heads(...)` unsliced) hits exactly this.
**Any cell-72 / cell-80 result produced with a narrow `active_heads_list` and unsliced
profiler output should be re-checked.** The fix is one line, and cell 110 shows it:

```python
L_ctx_list = [L_ctx_all[l][active_heads_list[l]] for l in range(model.cfg.n_layers)]
C_far_list = [C_far_all[l][active_heads_list[l]] for l in range(model.cfg.n_layers)]
```

#### 2.9.7 Return value

```python
{
  "per_text": [                       # one entry per (text, position) pair
     {"text", "text_index", "position", "n_tokens", "query_token",
      "loss_jsd",                     # ← the requested error
      "jsd_baseline_context_unigram", "total_variation", "top1_agreement",
      "real_top1_rank_in_approx", "real_entropy_nats", "approx_entropy_nats",
      "context_vals" [N_c], "context_keys" [N_c, S_init],   # ← the estimated pi
      "query_keys" [1, S_init], "n_context_states",
      "real_probs" [d_vocab], "approx_probs" [d_vocab]},    # if store_probs == "full"
     ...],
  "statistics": {                     # every metric → {count, mean, std, sem, min, max}
      "loss_jsd", "jsd_baseline_context_unigram", "total_variation",
      "top1_agreement", "real_entropy_nats", "approx_entropy_nats",
      "n_evaluated", "n_skipped",
      "mean_loss", "std_loss"},       # aliases of statistics["loss_jsd"]
  "skipped": [{"text_index", "position"?, "reason"}, ...],
  "config":  {"approximation", "max_tokens", "min_position", "n_positions",
              "temperature", "top_p", "seed"},
}
```

`summarize_benchmark(results)` prints the corpus-level block: the approximation's `name`,
evaluated/skipped counts, mean ± std (with SEM and range) for the JSD, the baseline it
must beat, total variation, top-1 agreement rate, and both mean entropies.

Cell 103 plots the error histogram against the baseline histogram, error versus position,
and dumps the worst case — its query token and the top-10 of both distributions side by
side, which is usually where you see *what* the abstraction got wrong rather than by how
much.

### 2.10 The *exact* 1-layer forward pass (cells 112–115)

`ExactSemanticHeadForwardPass` (cell 113) is not another approximation — it is the
**zero point of the error scale**. It drives the same `abstract_forward_pass` that every
other pipeline in this notebook uses, but parameterized so that it reproduces the concrete
model bit-for-bit. Measured on the benchmark: **JSD ≈ 1.6e-12 nats** (float32 round-off),
against a context-unigram baseline of ≈ 0.59 and a frozen-σ variant of ≈ 4.9e-3.

Its value to a future agent is diagnostic. If a benchmark run is bad, this cell tells you
whether the fault is in `single_layer_forward` (it is not — under these assumptions the
machinery is provably exact) or in how you parameterized it. It also converts every
residual error in the *general* case into an attributable quantity: turn one assumption
off at a time and watch the JSD climb.

#### 2.10.1 Why exactness is possible

Zero the positional embeddings of a 1-layer attention-only model and the pre-softmax score
between query position `t` and key position `j` collapses to

```
s(x_t, x_j) = (LN(W_E[x_t])·W_Q + b_Q) · (LN(W_E[x_j])·W_K + b_K) / sqrt(d_head)
```

— a function of the two **token ids** only. Position has dropped out. Therefore

```
Z       = Σ_j exp(s(x_t, x_j))            = Σ_v  n_v · exp(s(x_t, v))
attn_z  = Σ_j exp(s(x_t, x_j))·v_j  / Z   = Σ_v  n_v · exp(s(x_t, v))·v(v) / Z
```

where `n_v` is the number of times token `v` occurs in the context. The forward pass is an
exact function of the context unigram **counts**. Note *counts*, not the normalized
distribution: the total context length `T` genuinely matters, and carrying it is exactly
what `L_ctx` is for.

#### 2.10.2 Mapping that onto the 3-part partition function

`Z = Σ_c M_global[c] + Σ_s M_local[s] + M_sink`. Exactness is the requirement that the
three terms **partition the context exactly once** — every token position counted, none
counted twice.

| Term | What it covers, with `W_pos = 0` | What must be true |
|---|---|---|
| `M_sink` | exactly **one** BOS occurrence | free — with no positions `sink_k` *is* the BOS key, so `M_sink = exp(s(x_t, BOS))`. Requires `tokens[0] == bos_token_id` (the benchmark's `to_tokens(prepend_bos=True)` guarantees it) |
| `M_local` | the last `S_init` positions, **minus** every PAD/BOS position (`valid_mask` deletes those) | `exp(pos_score) = exp(0) = 1`; the window mask with `S_out == 1` admits all `S_init` positions, so each alive local token is counted once with its exact score |
| `M_global` | everything else — the far field | `L_global · π_c · C_far` must equal `n_far[c]` |

The tuning that follows:

```
C_far  = 1.0                  # C_far is the mean exponentiated POSITIONAL mass of a far
                              # token; with no positional embeddings that is exp(0).

L_ctx  = N_far + K_i + 1      # because single_layer_forward uses L_global = L_ctx - K_i - 1.
                              # The "-1" is the BOS/sink slot and the "-K_i" is the local
                              # window, so this parameterization is already the natural one.
                              # PER FORWARD PASS: it grows with the prompt. In the clean
                              # case (one BOS at position 0, no PAD) it is simply
                              # L_ctx = T = position + 1.

pi_c   = n_far[c] / N_far     # the FAR-FIELD-ONLY unigram, where
                              #   n_far[v] = total_count[v] - alive_local_count[v] - [v == bos_id]
                              # Using the whole-context unigram (what UnigramContextEstimator
                              # returns) double-counts the local window and the sink.

K_i    # NOT a tuning knob for exactness. It only shuttles mass between M_local and
       # M_global; K = 1 (query token alone in the window) and K = 8 give identical
       # results, verified. K = 1 is cheapest. What must be tuned is the TRIPLE
       # (K_i, pi, L_ctx) jointly, so that the partition stays exact.
```

The off-by-one that this accounting exists to prevent: if you leave `π` as the full context
unigram and set `L_ctx = T`, the tokens in the local window contribute **twice** (once via
`M_local`, once via their share of `M_global`) and BOS contributes twice (once via
`M_sink`, once via `π`). That is a silent, smooth error — the distribution still
normalizes, it is just wrong, and it grows as `K_i / T`.

#### 2.10.3 The one thing tuning cannot fix: the frozen LayerNorm

`single_layer_forward` linearizes LN with a single frozen scalar `ln_avg_sigma`
(§2.3, "the freeze hack"), while the true LN divides by a per-token `σ_v`. On this model
that alone costs **JSD ≈ 4.9e-3**, nine orders of magnitude above the ~1e-12 float32 floor
the rest of the construction reaches. It is the dominant error term in the 1-layer regime —
larger than everything the partition function does — and worth remembering when reading any
general benchmark number.

With no positional embeddings the layer-0 residual at position `j` is exactly `W_E[x_j]`,
so `σ_v` is a pure function of the token and can be made exact **without editing
`single_layer_forward`**: feed the model a rescaled embedding matrix.

```
W_E'[v] = W_E[v] · (s / σ_v)          with   ln_avg_sigma = s
```

Centering is linear and commutes with the per-row scalar, so
`(W_E'[v] − mean) / s == (W_E[v] − mean) / σ_v ==` the true LN output, for every `v`.
`σ_v` is precomputed once into `self.sigma_table` as
`sqrt(mean((W_E[v] − mean)²) + ln1.eps)` — the exact denominator TransformerLens uses.

The one place the **un**-normalized residual is still read is the residual-stream update
`shrunken_resid = combined_sem_resid[:, -S_out:, :] + attn_out`, which for a 1-layer model
only ever touches the **query** token. Choosing `s = σ_{x_t}` (the query token's own σ)
makes `W_E'[x_t] == W_E[x_t]`, so that read is exact too. The *context* rows' residuals are
left rescaled and therefore wrong — they are discarded after layer 0, which is precisely
why **this trick is restricted to `n_layers == 1`** and the constructor refuses anything
deeper.

#### 2.10.4 Constructor arguments

| Argument | Effect |
|---|---|
| `active_head` | the single head the abstraction models. The real side **must** ablate all the others |
| `K` | `K_list[0]`, and also `S_init`. Any value is exact; 1 is cheapest |
| `exact_layernorm` | `True` = the `W_E`-rescaling trick above. `False` = fall back to a scalar `ln_avg_sigma` (the mean of `sigma_table`), which is the ablation that isolates §2.10.3's 4.9e-3 |
| `mass_in_L_ctx` | *Only* `L_global · π_c` matters, so there are two factorizations of the same product. `True`: `π` is a normalized distribution and `L_ctx = N_far + K + 1`. `False`: `π` carries the raw counts and `L_ctx = K + 2` so `L_global = 1`. Mathematically identical — both measured at 1.644e-12, so the round-trip `False` avoids is below the float32 floor here. It would matter in fp16/bf16. The cost of `False` is that `context_vals` sums to `N_far`, not 1, which breaks every downstream consumer that assumes a simplex vector. **Leave it `True`** |
| `temperature`, `top_p` | keep at `1.0`; anything else measures the sampling transform (§5.5.2) |

#### 2.10.5 Preconditions — all enforced in `__init__` / `predict`

The class raises rather than silently degrading, because a "nearly exact" run is worse than
no run: it looks like a successful measurement.

- `model.cfg.n_layers == 1` (see §2.10.3) and no MLP (`attn_only`).
- `blocks[0].ln1` has no learnable **beta** — `single_layer_forward` applies γ but never β.
  The TransformerLens default `fold_ln=True` folds both into `W_{Q,K,V}` / `b_{Q,K,V}` and
  leaves a `LayerNormPre`, which satisfies this. **Loading with `fold_ln=False` — which
  §4.2 recommends for the K-gram pipeline — breaks exactness**, and the constructor says so.
- `model.cfg.attn_scale == sqrt(d_head)`, since `single_layer_forward` hardcodes it.
- `W_U` does not alias `W_E` (the rescaling would corrupt the unembedding).
- `tokens[0] == bos_token_id`, and no negative far-field count (the accounting self-checks).
- The **real** side of the benchmark must run with both hooks:
  `zero_head_hook` on every head except `active_head`, and `remove_pos_embed_hook`.
  This is not checkable from inside the class; cell 115 wires it correctly, copy from there.

#### 2.10.6 How it patches global state

`abstract_forward_pass` reads `model.W_pos` and `model.W_E` directly, not through hooks, so
the class patches `.data` under two context managers (`_zero_pos_embeddings`,
`_rescaled_embeddings`) with `try/finally` restores. Consequences:

- `approx_hooks` must stay **empty** — the abstraction patches the weights itself.
- It is not thread-safe and must not run concurrently with anything else touching the model
  (relevant if you ever fold it into the multi-GPU lane of cells 88–92).
- `_W_E_buffer` is a persistent `[d_vocab, d_model]` scratch tensor (~100 MB at
  48k × 512 fp32) held for the lifetime of the object, alongside `W_E_unit` of the same
  size. Budget ~200 MB of VRAM per instance; cell 115 builds four of them for the ablation
  table, so delete them when done.

#### 2.10.7 Reference numbers

`attn-only-1l`, head 3, 9 positions ≥ 300 on a small synthetic corpus of repeated
sentences. Cell 115 as written runs on cell 110's Wikipedia `corpus`, so the exact digits
will differ — what should reproduce is the *separation* between the rows:

```
exact K=1                                   JSD = 1.644e-12   top1 = 1.000
exact K=8                                   JSD = 1.598e-12   top1 = 1.000
counts in pi (mass_in_L_ctx=False)          JSD = 1.644e-12   top1 = 1.000
frozen sigma (exact_layernorm=False)        JSD = 4.919e-03   top1 = 1.000
baseline: JSD(real, raw context unigram)          5.897e-01
```

Read these as: the partition-function machinery is exact; `K` is free; the two
factorizations of `L_global · π` agree; and the frozen-σ linearization is the single
largest modelling error in the 1-layer regime.

---

### 2.11 `MultiTimescaleMeanFieldForwardPass` — the full-model abstraction (cell 108)

`ExactSemanticHeadForwardPass` (§2.10) buys its exactness by deleting everything that
makes the model hard: the positional embeddings, seven of the eight heads, and any
dependence on *where* in the context a token sits. `MultiTimescaleMeanFieldForwardPass`
(MTMF) is the first class in the notebook that approximates the concrete model **as it
actually is** — every head alive, positional embeddings on, MLP applied if there is one,
and **no hooks on the real side** when it is benchmarked.

It is also the notebook's **single source of truth** for the physics. The same object is

* the `ApproximateForwardPass` the benchmark scores (`predict`), **and**
* the operator the fixed-point optimization iterates (`forward`, `power_iteration_step`,
  `stationary`).

There is no second copy to keep in sync — which is the specific failure mode §2.4 and §4.3
document for the older `single_layer_forward` lane, where the exploration engine and the
gradient loop could silently drift apart.

#### 2.11.1 The model: three blocks, one partition function

For a query token `x_t` at a (dummy) absolute position `t*`, head `h` resolves its context
in three blocks:

```
Z_h = Σ_{d=0}^{K-1} E_h(x_{t-d}, t*-d)                              ← FAST WINDOW  (exact)
    + W_far_h · Σ_c π_h(c) · exp(q_h · k_s_h(c) / scale)            ← MEAN FIELD   (the only approximation)
    + exp(q_h · k_BOS_h / scale)                                    ← BOS SINK     (exact)
```

and the head output is the same three sums taken against the values, divided by `Z_h`.

* **The fast window** is computed with no approximation whatsoever: real token embeddings,
  real absolute positional embeddings, the real `ln1` module, and **all four QK cross
  terms** (`q_s·k_s`, `q_s·k_p`, `q_p·k_s`, `q_p·k_p`). The positional attention profile
  *inside* the window is therefore **generated by the model**, not imposed — there are no
  exponential weights here at all. This is the single biggest difference from
  `single_layer_forward`, whose `M_local` multiplies a factorized `exp(q_s·k_s)·exp(q_p·k_p)`
  and drops the two cross terms. (`local_cross_terms=False` reproduces the old behaviour
  as an ablation.)
* **The mean field** covers everything older than the window. Under a linearized LayerNorm
  `k(x, j) = k_s(x) + k_p(j)`, so `exp(q·k(x_j, j)) = exp(q·k_s(x_j)) · exp(q·k_p(j))`
  *exactly*. The approximation is the next step: the two factors are assumed
  **independent across the far field**, so the sum over far positions factors into
  (sum of positional weights) × (π-average of semantic weights). That is the *whole*
  approximation in the attention block.
* **The BOS sink** uses the true key and value of BOS at position 0 (`extract_bos_sink`),
  and scores it with the **full** query — semantics *and* position — because attention to
  the sink is strongly position-dependent.

#### 2.11.2 Per-head timescales: one number fixes everything

Each head carries its own effective context length `L_ctx[h]`, the total number of tokens
it integrates, **fast window included**. Everything else is derived:

```
γ_h     = 1 − 1 / L_ctx[h]                       geometric profile with mean horizon L_ctx
Z_far_h = Σ_{d≥K} γ_h^(d−K) = 1 / (1 − γ_h)      the effective far-token count
W_far_h = E_pos_h(K) · Z_far_h                   the mean-field block's total weight
```

`E_pos_h(K) = exp(q_h · k_p_h(t*−K) / scale)` is the **real** positional attention factor
at the first position outside the window. This is the load-bearing detail: the geometric
tail is **anchored to the explicit window by continuity at its edge** (`d = K`), so the
mean field picks up exactly where the exact block stops. If the true profile really is
geometric with ratio `γ_h`, the three blocks together integrate exactly `L_ctx[h]` tokens'
worth of attention mass — no double counting, no gap.

Consistently, `π_h` is the discounted far-field law **anchored at `d = K`** (the nearest
far token has weight `γ^0 = 1`); the `γ^K` separating it from the query lives in
`E_pos_h(K)`, not in `π_h`. The pair `(π_h, Z_far_h)` is the exact analogue of the
`(π, L_ctx)` split of `ExactSemanticHeadForwardPass(mass_in_L_ctx=True)`: only the
**product** `Z_far_h · π_h(c)` — the effective discounted *count* of token `c` in head
`h`'s far field — ever enters the partition function. **A truncated, unnormalized `π`
therefore silently shortens the context**, exactly as in §2.10.4.

#### 2.11.3 LayerNorm: pairwise σ, not a frozen scalar

`LN(e_v + p_j) = w · (e_v − μ + p_j − μ) / σ(v, j) + b`. Centering is linear, so the split
into a semantic and a positional stream is **exact provided both streams are divided by
the same joint σ(v, j)**. The class never uses one global frozen σ:

| Stream | σ used |
|---|---|
| fast window + query | the true `σ(token, position)` — in fact these rows go through the real `ln1` module, so they are exact |
| far-field semantic keys/values | `σ(c, j_ref_h)`, where `j_ref_h = t* − (K + γ_h/(1−γ_h))` is the **discount-weighted mean far position** |
| far positional stream (one vector per head) | a single per-head scalar: the `π_h`-average of `σ(c, j_ref_h)` |

`_pair_sigma_outer` computes `σ(c, j)` in closed form from `‖ê_c‖²`, `‖p̂_j‖²` and their
inner product, so it costs one matvec rather than an `[H, N_c, d_model]` tensor.

Two details worth not "fixing":

* The π-average that produces `sigma_pos` uses `context_vals.detach()`. π only picks
  *which* far tokens the σ average runs over; that is a device for summarizing the
  positional stream, not part of the operator, and detaching keeps the gradient w.r.t. π
  inside the mean-field weights alone.
* `layernorm_mode="frozen"` falls back to the notebook's original single scalar. It exists
  to **price** the pairwise treatment, not as a supported production setting — §2.10.3
  shows the frozen hack is the dominant error term in the exact lane.

#### 2.11.4 Numerical stability: the max shift

`log_W_far` carries `log(far_mass)`, which for `L_ctx ≈ 800` is `+6.7` on top of a raw QK
logit. The original `single_layer_forward` exponentiates raw scores and **will** overflow
here. `_forward_chunk` subtracts a per-`(head, query)` maximum from every exponent before
`exp()`:

```python
shift = max(local_logits.max(-1), far_exponent.max(-1), sink_logits).detach()
```

`Z` and the output numerator are both homogeneous of degree 1 in the masses, so the ratio
is untouched. The `.detach()` matters: the shift is a numerical device, and letting a
gradient flow through an `argmax`-selected element would add a spurious term.

#### 2.11.5 `attn_scale` is read off the module, not assumed

```python
self.scale = self._resolve_attn_scale(model, 0)
```

preference order: `model.blocks[0].attn.attn_scale` (already resolved and always correct),
then `cfg.attn_scale` if it is a real value, then `sqrt(d_head)`. **GPT-Neo-derived models
(`tiny-stories-*`) use `attn_scale = 1.0` and carry the scaling in the weights, and
`cfg.attn_scale` is `−1.0` there (a sentinel).** `single_layer_forward` hardcodes
`sqrt(d_head)` and is therefore *silently wrong* on those models; MTMF is not. This is one
of the reasons it is the source of truth, and it is why the TinyStories lane (§2.13.2) is a
real test rather than a rerun.

#### 2.11.6 State, and what "π" means at a fixed point

The tracked macroscopic state is a sparse measure over **K-grams** (the queries); the mean
field is its **unigram marginal**, because a far-field position is only ever read through
its token identity. `kgram_to_unigram` does that projection differentiably
(`torch.unique` + `scatter_add`), so a gradient on the K-gram measure flows through the
mean field too.

> **At a mean-field fixed point every head's `π_h` is the same distribution.**
> A discounted average of a stationary process has the same expectation for every `γ` —
> the timescale changes the *variance* of a realized estimate, not its mean. Per-head
> states differ only when the context is a concrete, finite prompt, which is exactly the
> benchmark (`predict`) case.

Hence `forward` accepts `context_vals` as either `[N_c]` (tied — the optimization case) or
`[n_heads, N_c]` (per head — the prediction case), and the per-head timescales still act on
the optimization through `W_far_h` alone. This is why §2.14's wide optimization uses a
**unified** π and is not thereby cutting a corner.

#### 2.11.7 Constructor arguments

| Argument | Default | What it does |
|---|---|---|
| `model` | — | `HookedTransformer`, **`n_layers == 1`** (enforced: a unigram mean field is only a sufficient state for one layer; a second layer would read the far field's own neighbourhood). An MLP is allowed and is applied. |
| `L_ctx` | — | Effective context length **per active head**, window included. float, list or `[n_heads]` tensor. Must exceed `K + 1` for every head. |
| `active_heads` | `None` = all | Heads the abstraction models. The real model must ablate the rest for a like-for-like benchmark. |
| `K` | `1` | Fast window width; also `S_init`, also the width of a query key. |
| `gamma` | derived | Override `1 − 1/L_ctx`. `gamma == 1.0` means "no decay", and then `far_mass` must be supplied explicitly (`1/(1−γ)` diverges). |
| `query_position_offset` | `n_ctx − 1` | The dummy absolute position `t*`. Everything positional is measured relative to it. Must satisfy `K ≤ t* < n_ctx`. |
| `use_real_query_position` | `False` | `predict` uses the prompt's true position instead of `t*`. Off by default so `predict` and the optimization loop run the *same* operator. |
| `local_cross_terms` | `True` | Keep all four QK terms inside the window. `False` reproduces the old factorized local block. |
| `far_positional_value` | `True` | Add the discount-weighted mean positional embedding to the far-field **value**. The value is linear in the residual, so this is free and strictly more faithful. |
| `layernorm_mode` | `"pairwise"` | `"frozen"` = one scalar σ for the far field (§2.11.3). |
| `frozen_sigma` | RMS of a typical embedding | The scalar for `layernorm_mode="frozen"`. |
| `mask_dead_local` | `True` | Wipe BOS/PAD positions out of the fast window so the sink is not double-counted. |
| `ablate_sink` | `False` | Drop the BOS sink entirely — no key mass in `Z`, no value in the output. The JSD gap to `full` is how much of the forward pass the sink carries. |
| `query_chunk_size` | `None` | Split the query batch when forming the `[H, N_q, N_c]` score tensor. `None` = one shot. |
| `temperature`, `top_p` | `1.0`, `1.0` | Sampling transform on the returned probabilities, with the same straight-through estimator as §2.1. |
| `context_estimator` | `DiscountedUnigramContextEstimator(window=K)` | Used by `predict` only. |

#### 2.11.8 The verified limit — the acceptance test to re-run after any edit

Zero `W_pos`, set `gamma = 1.0`, `active_heads = [h]`, `K = 1`, and pass the realized
`far_mass` — i.e. drive MTMF into the regime of `ExactSemanticHeadForwardPass` — and it
reproduces the concrete model (other heads ablated) to

```
JSD = 1.3e-12 nats on attn-only-1l, K-invariant across K = 1, 2, 8
```

That is float32 round-off, and it is the evidence that the three blocks **partition the
context exactly once**: window + far field + sink, no token counted twice and none
dropped. K-invariance is the sharp part of the test — it says mass moved between
`M_local` and `M_far` without changing the answer.

⚠ **No cell in the notebook currently reproduces this.** The number lives in the class's
own block comment; the nearest live acceptance test, cell 118, checks the *batched path of
`ExactSemanticHeadForwardPass`*, not MTMF. If you edit `_forward_chunk`, reconstruct the
limit by hand — zero `W_pos`, one head, `gamma=1.0`, `K=1`, realized `far_mass`, compare
against the concrete model with the other heads ablated — before trusting anything
downstream. Adding that as a permanent cell would be a cheap, high-value contribution.

---
### 2.12 `DiscountedUnigramContextEstimator` (cell 108)

The third `ContextDistributionEstimator` (after `UnigramContextEstimator` and
`KGramContextEstimator`, §2.9.4), and the only one that knows about timescales.

For a context `x_0 … x_t` and a window of the last `K` tokens, the far field is
`x_0 … x_{t−K}`, and head `h` weights position `j` by `γ_h^((t−K) − j)`: **the nearest far
token has weight 1** and the weights decay geometrically into the past. The normalizer

```
Z_far_h = Σ_{j far} γ_h^((t−K)−j)  →  1 / (1 − γ_h)   (long prompt)
```

is the effective number of far tokens head `h` integrates, and **it must travel with
`π_h`** — `MultiTimescaleMeanFieldForwardPass.forward` multiplies the mean-field block by
it. Returning a normalized `π` and a separate mass is the same split as
`ExactSemanticHeadForwardPass(mass_in_L_ctx=True)`.

#### 2.12.1 Two entry points, and why the contract one is lossy

| Method | Returns | Use |
|---|---|---|
| `estimate_per_head(tokens, model, K, gammas, dead_ids)` | `vals [H, N_c]`, `keys [N_c]`, `far_mass [H]` | **The real path.** One distribution per head (one per discount), over a shared support. This is what `MTMF.predict` and the wide optimization call. |
| `estimate(tokens, model, S_init)` | `vals [N_c]`, `keys [N_c, S_init]` | The `ContextDistributionEstimator` contract, so a discounted state can be A/B'd against the plain unigram one inside `PartitionFunctionApproximateForwardPass`. Needs a scalar `gamma`. |

> ⚠ **`estimate` discards `far_mass`.** The contract has nowhere to put it. If you use that
> path, the caller's `L_ctx` must be set by hand to `1/(1−γ) + K + 1` or the weighting is
> silently wrong. Prefer `estimate_per_head` wherever the consumer can accept it.

#### 2.12.2 Why dead tokens are dropped

BOS / EOS / PAD are removed from the counts (`exclude_special=True`). **BOS is already
carried exactly once by the sink term**, and counting it again in `π` double-counts the
sink — the same rule the exact lane enforces via `drop_dead_states` (§2.10). This is not a
cleanliness preference; it changes the partition function.

#### 2.12.3 Truncation semantics

`top_n` keeps the highest-mass tokens and renormalizes `π` over the survivors, while
`far_mass` keeps the **full realized mass**. So truncation *redistributes* the dropped mass
instead of shortening the effective context — the opposite of what a naive slice would do.
`min_mass` drops far tokens below a post-normalization threshold, applied before `top_n`.

#### 2.12.4 Implementation notes

* Weights are computed as `exp(e · log γ)` rather than `γ ** e`: stable over the hundreds
  of powers a long prompt needs, and exact at `γ == 1`.
* The support is the union over heads (`counts.sum(dim=0) > 0`), so all `H` rows share one
  `keys` tensor and `forward` can form a single `[H, N_q, N_c]` score tensor.
* It raises rather than guesses when the context is not longer than `K` (no far field) or
  when the far field is entirely special tokens.

---

### 2.13 The full-abstraction benchmark and the feature ablations (cells 128–135)

#### 2.13.1 `attn-only-1l` (cells 128–131)

**Cell 129 — timescale profiling.** This is the cell that makes `L_ctx` a *measurement*
rather than a guess. `positional_profile(query_token, d_max)` evaluates the real
positional attention factor

```
E_pos_h(d) = exp(q_h · k_p_h(t*−d) / scale)
```

for `d = 0 … d_max`, and `fit_gammas_from_profile` least-squares-fits
`log E_pos_h(d) = a + d·log γ_h` over the far field. Then `L_ctx_h = 1/(1−γ_h)`.

Design decisions in that cell that are easy to undo by accident:

* **Probe tokens are real corpus tokens**, not arbitrary ids, and special tokens are
  filtered out. `E_pos` depends on `q`, so it depends on the query token.
* **The average over probe tokens is geometric, not arithmetic** — the profile is a
  product of exponentials, and an arithmetic mean would be dominated by whichever query
  token happens to attend hardest.
* **γ is the median across probe tokens**, which is robust to the handful of tokens whose
  fit is degenerate.
* **`L_ctx` is clamped into `[K+2, T_STAR]`** and the clamp is printed. A head whose
  profile is flat or rising over the fit range has no geometric horizon at all; clamping it
  is a decision, and the print is there so it is a *visible* one.
* **The plot is the point, not decoration.** The right-hand panel draws the measured tail
  against `γ^(d−K)`. If a head's curve is not straight on a log axis, its geometric tail is
  a summary, not a description, and every number derived from it inherits that.

The cell ends by building `mtmf_full` — all heads, `K = K_WINDOW`, `t* = T_STAR`,
`temperature = 1`, `top_p = 1` — which is the object every downstream cell consumes.

**Cell 131 — approximation quality + feature ablations.** The ground truth is the
**untouched** model: `real_hooks = []`, every head alive, positional embeddings on. Unlike
the single-head / position-free lane, nothing has been removed from the model to meet the
abstraction halfway, so the JSD is the real cost of replacing the context with
*(fast window + per-head discounted mean field + BOS sink)*.

Each row turns off exactly **one** ingredient, so the gap to `full` is that ingredient's
contribution:

| Label | Override | What it prices |
|---|---|---|
| `full` | — | the reference |
| `frozen LayerNorm` | `layernorm_mode="frozen"` | the pairwise σ of §2.11.3 |
| `no QK cross terms` | `local_cross_terms=False` | the two cross terms the old `M_local` dropped |
| `no far pos. value` | `far_positional_value=False` | the mean positional embedding in the far-field value |
| `no BOS sink` | `ablate_sink=True` | how much of the forward pass the sink carries |
| `true query position` | `use_real_query_position=True` | the cost of the dummy `t*` |
| `uniform L_ctx` | `L_ctx=L_ctx_used.mean()` | **whether per-head timescales matter at all** |
| `K=1` / `K=8` / `K=16` | `K=…` | where the window/mean-field split should sit |

The control `jsd_baseline_context_unigram` (the raw context histogram, no model at all)
comes free with every run and is drawn as the line every bar must clear.

> **Read the `uniform L_ctx` row first.** If it matches `full`, the multi-timescale story
> is not carrying its weight on this model and the whole per-head apparatus is decoration.
> If `K=1` matches `full`, the fast window is not either.

Because `ApproximationQualityBenchmark` seeds its RNG per run, **all rows are evaluated at
the same positions**, so the differences are paired and the SEM bars are comparable. That
also means the pitfalls of §5.5 (3) apply in reverse: do *not* change `seed`,
`min_position`, `max_tokens` or `n_positions` between rows.

#### 2.13.2 `tiny-stories-1L-21M` (cells 132–135)

The same two cells on a harder target, and a genuine test rather than a rerun:

* **16 heads instead of 8** — 16 mean-field weights have to be right at once.
* **It has an MLP.** The abstraction runs it on the reconstructed residual.
* **It is GPT-Neo-derived, so `attn_scale == 1.0`, not `sqrt(d_head)`** (§2.11.5). MTMF
  reads the divisor off the module and is correct; `single_layer_forward` — and therefore
  every other pipeline in this notebook — hardcodes `sqrt(d_head)` and would be silently
  wrong on this model.

Two deviations from the `attn-only-1l` config, both forced by the model and both applied to
every ablation row equally:

* `n_ctx = 512`, so `T_STAR = 450` instead of 800 (the query position must be `< n_ctx`).
* the benchmark window moves with it: `max_tokens = 512`, `min_position = 400`.

So the TinyStories columns are comparable **with each other**, not with the `attn-only-1l`
table (different context length, different corpus).

Two further details specific to this lane:

* **The corpus is joined with blank lines, not `<|endoftext|>`.** This tokenizer has
  `bos == eos == pad == 50256`; an interior EOT would be masked out of the local window
  *and* dropped from the far field while the sink still fires exactly once — those
  positions would fall out of the partition entirely.
* **The `K=20` / `K=40` rows pass `L_ctx=ts_L_ctx_used.clamp(min=K+2)`.** Widening `K` past
  a head's own horizon is not legal: the far field would be empty and `L_global` would go
  negative. Clamped heads become pure window heads with a vestigial mean field, which is
  the honest reading of "this head only ever looked `K` tokens back anyway".

---
### 2.14 Wide multi-timescale mean-field optimization (cells 136–140)

The wide fixed-point search of §"Wide Semantic Head Optimization" (cells 88–106), re-run on
the **full** MTMF operator instead of the 1-head token-level one. One independent
explore-then-optimize run per `(text, position)` pair, single GPU — there is nothing to fan
out over heads, because every head is in play at once.

#### 2.14.1 What changes relative to cells 89/91

| | Cells 89/91 (token-level) | Cells 136–140 (MTMF) |
|---|---|---|
| Operator | `pi_to_pi_P_one_layer_topk`, one head, no positions | `MultiTimescaleMeanFieldForwardPass`, all heads, positions on |
| State | unigram, dense `[vocab]` | **K-gram**, sparse `[N, K]` + `[N]` |
| Mean field | the state itself | the state's **unigram marginal** via `kgram_to_unigram` |
| Initial condition | `pi_from_context` — the raw context histogram | `DiscountedUnigramContextEstimator`, lifted to K-grams |
| Exploration | none | no-grad power iteration of the same operator |
| Parallelism | one head per GPU, `joblib`/`loky` | single process, single GPU |
| Result key | `wide_results[head][i]` | `wide_mtmf_results[i]` (flat list) |

Two of those deserve spelling out.

**Why the state has to be K-grams.** `forward` needs an explicit `[N_q, K]` fast window per
query — that is what makes the window block exact. So the tracked measure lives on K-grams:
K-grams choose the active queries and carry the dynamics (the key shift), and their unigram
marginal is what the mean-field block integrates against. Both projections happen every
step, and `kgram_to_unigram` is differentiable, so the gradient couples them.

**Why exploration is a plain power iteration, not `MetastableKGramEngine`.** The engine
drives `abstract_forward_pass`, which is a *different operator* from this class
(§2.11.5 alone is enough to make them disagree on some models). Mixing them would let the
exploration and gradient phases explore different dynamics — precisely the drift §2.11
exists to prevent. `mtmf.stationary(...)` under `torch.no_grad()` is the exploration phase,
and it is the same code path the gradient phase uses.

The cost is real and worth naming: the engine prunes on **fully aggregated flux**, so
states reached by many weak paths survive (§2.5). A top-k rollout does not. If the support
looks anaemic, that trade is the first place to look.

#### 2.14.2 What `π` means here

`π` is the **far-field** measure: the law of the context *outside* the `K` most recent
tokens and *outside* the BOS sink, because that is the only thing `forward` reads as a mean
field. Consequences, all enforced by `drop_dead_kgrams`:

* BOS and PAD are banned from `π`'s support, on **both** sides of the JSD. Filtering only
  one side leaves a permanent JSD floor and a gradient that forever pulls `π` toward BOS.
* `π` handed to the forward pass must sum to 1, because `far_mass · π(c)` is the expected
  *count* of `c` in the far field. `top_mass` is logged every iteration for exactly this
  reason (§6, `N_ACTIVE`).
* Do not compare `π*` against the raw unigram of a generated sample without stripping
  special tokens first.

#### 2.14.3 The initial condition

`DiscountedUnigramContextEstimator.estimate_per_head` gives the discounted far-field law of
a real prompt, one row per head, plus the realized `far_mass`. The row of the
**longest-timescale head** (largest `L_ctx`, `γ` closest to 1) is taken as "the" discounted
local distribution.

That row is a unigram; the operator needs K-grams. The lift weights every realized
far-field K-gram by the *same* geometric discount `γ^((t−K−1) − j)`, `j` = index of the
K-gram's last token. By construction the `"last"`-token marginal of that K-gram measure **is**
the estimator's unigram, except for the first `K−1` positions, which no K-gram window can
reach and which carry the most-discounted mass of all. Both forms are stored in the result
dict (`pi_init_vals`/`pi_init_keys`, `pi_init_unigram`, `pi_init_estimator_per_head`) so the
identity can be checked rather than believed.

`far_mass` is the **realized** `Z_far` of that prompt, not the saturated `1/(1−γ)`, and it
is held fixed for the whole run. It is the physical context length of the initial
condition — the analogue of `L_PROMPT` in cell 117, which §2.10 calls *the* physical
parameter of the problem rather than a nuisance one.

#### 2.14.4 The loop

Structurally identical to cell 119 (§2.10's exact lane), phase for phase:

```
PHASE 1  exploration    mtmf.stationary(...) under no_grad, M = deep/shallow
                        → drop dead K-grams → cap at N_TRACKING
PHASE 1b union merge    torch.unique(dim=0) + scatter_, new states enter at 2·eps
                        (or + ETA_EXPLORE · flux, a damped-Picard hybrid)
PHASE 2  gradient       pi_logits = log π ; π = softmax(pi_logits)
                        mean field = kgram_to_unigram(π over the FULL union)
                        queries    = top-N_ACTIVE of π, renormalized
                        πP         = mtmf.stationary(..., M = M_ITERATIONS, checkpointed)
                        loss       = LOSS_SCALE · compute_aligned_jsd(π, πP)
PHASE 3  natural grad   Sherman–Morrison on the simplex, λ = 1e-10, clip to 1.0
PHASE 4  prune/migrate  top-N_TRACKING, drop dead, renormalize
PHASE 5  logging
```

One deliberate difference from cell 119: **the mean field is taken over the whole union
support, not the top-k query slice.** It is a `scatter_add`, not a forward pass, so it is
nearly free, and it is the object `forward` integrates against — truncating it would
truncate the physics rather than just the query set.

The best iterate is recorded at `pi_vals_active`, the iterate the loss was *measured* at,
not the post-step iterate of phase 4.

#### 2.14.5 The result dict

`wide_mtmf_results` is a flat list; each entry carries everything needed to re-analyze the
run without the notebook state that produced it.

| Group | Keys |
|---|---|
| What was optimized | `text`, `text_index`, `position`, `n_tokens`, `context_tokens` (the exact prompt), `active_heads` |
| Operator parameters | `K`, `t_star`, `gamma [H]`, `L_ctx [H]`, `far_mass [H]`, `far_mass_saturated`, `init_head`, `init_head_slot`, `init_gamma`, `operator_name` |
| Initial condition | `pi_init_vals [≤S, ]` / `pi_init_keys [≤S, K]`, `pi_init_unigram` (sparse COO `[vocab]`), `pi_init_estimator_per_head [H, N_c]`, `pi_init_estimator_keys [N_c]`, `n_states_init` |
| Result | `pi_final_vals` / `pi_final_keys`, `pi_final_unigram`, `pi_best_vals` / `pi_best_keys`, `n_states_final` |
| Traces | `losses`, `initial_loss`, `final_loss`, `best_loss`, `best_iteration`, `n_iterations`, `top_mass_history`, `n_union_history`, `explore_size_history` |

Cell 138 saves `{"wide_mtmf_results": …, "config": …}` to
`results/wide_optimizations/wide_mtmf_results_<timestamp>.pt`, with the config carrying
every knob in §6 plus the operator's own settings. The load pattern of cell 93 works
unchanged apart from the top-level key.

#### 2.14.6 Cell 140 — single-run diagnostics

Reads one entry and prints the headline numbers (initial → final → best loss, support sizes
in both spaces, entropy and effective support size, `JSD(π₀, π_final)`, the fraction of
final mass still on K-grams the prompt itself realized, truncation mass, per-head
`far_mass`), then draws an 8-panel figure:

| | |
|---|---|
| loss vs. iteration (log y, best marked) | support size vs. iteration (union / exploration) |
| truncation mass vs. iteration, zoomed | K-gram rank–mass curve, `π₀` vs `π_final` |
| top-20 final K-grams (+ their initial mass) | top-20 final unigrams (+ their initial mass) |
| top-20 initial K-grams | unigram displacement scatter, init vs final |

It touches nothing but `model.to_string`, so it is safe against reloaded results with no
optimization in flight. §6 says which panel diagnoses which knob.

---

## 3. Goals & Use Cases

### 3.1 Goal by component

| Component | Goal |
|---|---|
| `pi_to_pi_P_one_layer*` | Compute the induced next-token measure for a **1-layer, position-free, token-level** abstraction of the model. Cheapest possible instantiation of `P_π`. |
| `abstract_forward_pass` | Compute `P(next token \| K-gram query, frozen π)` for a **multi-layer** model with positional and sink structure explicitly modeled. The workhorse. |
| `compute_stationary_distribution` | Differentiably apply `P_π` `M` times to a sparse measure over K-grams while keeping the support at size `N`. |
| `MetastableKGramEngine` | Find the (approximately) closed set of K-grams the model recirculates through under a fixed context — i.e. *the attractor's support*. |
| `extract_substochastic_matrix` | Turn that support into an explicit `(N+1)×(N+1)` sparse Markov matrix with an absorbing sink that collects all escaping flux. |
| `MetastableMSMAnalyzer` | Extract the physics: quasi-stationary distribution, basin escape rate, mixing timescales, spectral gaps, and PCCA+ macrostates ("topics"). |
| `profile_*_heads*` | Measure the per-head positional/semantic statistics that parameterize the abstraction. |
| DLA / knockout / QK-OV cells | Classic mechanistic-interpretability scratch work on the *concrete* model, used to pick `active_heads_list` and `K_list`. |
| `ContextDistributionEstimator` | Turn a realized context into the macroscopic state `π` in the `[N_c, S_init]` sparse format. Pluggable: unigram, K-gram, windowed. |
| `ApproximateForwardPass` | Predict `p(next token \| context)` *without* running the concrete model. Pluggable: the partition-function abstraction, or any future variant. |
| `ExactSemanticHeadForwardPass` | **Calibrate** the benchmark: drive `abstract_forward_pass` so that it reproduces the concrete model exactly (1 layer, 1 head, no positions). The zero point every other JSD is read against. |
| `ApproximationQualityBenchmark` | **Validate** the abstraction: JSD between the real and approximated next-token distributions over a text corpus, with running mean/std. This is the cell that tells you whether anything else in the notebook is trustworthy. |
| `DiscountedUnigramContextEstimator` | Turn a realized context into a **timescale-aware** state: the exponentially discounted far-field law, one row per head, plus the realized `Z_far`. The only estimator that knows the far field is not the whole context (§2.12). |
| `MultiTimescaleMeanFieldForwardPass` | The abstraction of the model **as it is** — all heads, positions on, MLP applied. Simultaneously the benchmarked `ApproximateForwardPass` and the operator the fixed-point search iterates, so the two cannot drift apart (§2.11). |
| `mtmf.positional_profile` / `fit_gammas_from_profile` | **Measure** `L_ctx` off the model's own positional embeddings instead of guessing or profiling it from data (§2.13.1). |
| Cells 136–140 | Run the fixed-point search of the full operator from a real prompt's discounted state, over a corpus, and store everything a later analysis needs (§2.14, §6). |

### 3.2 Which method should I use?

**"I want to find a fixed point of a 1-layer attention-only model over tokens."**
→ Cell 52 (natural gradient) if you want the principled optimizer; cell 54 (exponentiated
GD) if 52 is numerically unstable; cell 57 (damped Picard) if you want to know which fixed
point the dynamics actually *fall into*. Use `pi_to_pi_P_one_layer_topk` in the loop and
`pi_to_pi_P_one_layer` for periodic validation.

**"I want to find a fixed point of a multi-layer model over K-grams."**
→ Cell 68 (current, 3-part partition). Do **not** start from cell 60.

**"I want to characterize the attractor, not just find it."**
→ Cells 72–77: `MetastableKGramEngine.run_pruned_power_iteration` →
`extract_substochastic_matrix` → `MetastableMSMAnalyzer`. This gives you eigenvalues,
timescales in tokens, the QSD, and PCCA+ topic clusters — much more informative than a
single fixed-point vector.

**"Power iteration or Dijkstra for support discovery?"**
→ Power iteration when you have a meaningful seed measure and want the states that carry
mass. Dijkstra when you want an exhaustive, measure-independent reachability set, or when
power iteration's frontier rate refuses to converge. Cell 72 contains commented-out code
to run both and compute the overlap — a good diagnostic.

**"Which head profiler?"**
→ `profile_thermodynamic_heads` if you just need `(L_ctx_list, C_far_list)` for the
forward pass. `estimate_L_ctx_for_head` / `estimate_c_far_for_head` if you want per-head
analytic values with zero data (or want to hand-override a subset, which cell 72 does).
`profile_deep_heads_empirical` for `(μ, σ)` on a base-model corpus;
`..._instruct` for chat models where the template matters;
`..._multigpu` when the ensemble is large enough that one GPU is the bottleneck.

**"Is my abstraction any good?"**
→ Cells 107–111. Build a `PartitionFunctionApproximateForwardPass` around your
`forward_pass_kwargs` and run `ApproximationQualityBenchmark` on a corpus. Check the mean
JSD **against the `jsd_baseline_context_unigram` control** — an abstraction that does not
beat the raw context histogram is not using the model. Do this *before* trusting a
fixed point, a QSD, or a PCCA+ macrostate, because every one of them is computed through
`abstract_forward_pass`.

**"My JSD is bad — is `single_layer_forward` wrong, or is my parameterization wrong?"**
→ Cells 112–115 (§2.10). `ExactSemanticHeadForwardPass` drives the *same*
`abstract_forward_pass` to a JSD of ~1e-12 on a 1-layer, single-head, position-free model.
So the machinery is not the problem; `π`, `L_ctx`, `C_far`, `K_list`, `active_heads_list`
or the frozen σ are. Reproduce the exact run first, then reintroduce your assumptions one
at a time and watch where the JSD jumps — §5.4 is the table of what to change.

**"How much does the frozen-LayerNorm hack actually cost me?"**
→ `ExactSemanticHeadForwardPass(exact_layernorm=False)`. On `attn-only-1l` it is the
difference between 1.6e-12 and 4.9e-3, i.e. it is the dominant error term in the 1-layer
regime — larger than everything the partition function does. §2.10.3.

**"Which context estimator should I use in the benchmark?"**
→ `UnigramContextEstimator` for `n_layers == 1`, where the lift to `[N_c, S_init]` is
exact. `KGramContextEstimator` for anything deeper — the unigram lift degenerates there
(§2.9.4). Use `context_window=` on either to test a finite-memory hypothesis.

**"I want to approximate the model as it actually is — all heads, positions on."**
→ `MultiTimescaleMeanFieldForwardPass` (§2.11). Cell 129 measures `L_ctx` per head off the
model's own `W_pos`; cell 131 benchmarks it against the **untouched** model with no hooks
at all. This is the only lane where the ground truth has not been bent toward the
abstraction, so it is the only JSD that answers "is the mean-field picture true?".

**"Which of the abstraction's ingredients actually matter?"**
→ Cell 131's ablation table (§2.13.1). Each row disables exactly one feature, all rows
share the same sampled positions, and the context-unigram control is drawn as the line
every bar must clear. Read `uniform L_ctx` and `K=1` first: if either matches `full`, the
corresponding apparatus is decoration on this model.

**"I want fixed points of the FULL model, not of a single head."**
→ Cells 136–140 (§2.14). K-gram state, unigram mean field, initial condition from a real
prompt's discounted far-field law, explore-then-optimize with pure power-iteration
exploration. §6 is the manual for every knob.

**"Which fixed-point lane should I use?"**
→ Cell 68 for the multi-layer `abstract_forward_pass` operator. Cells 116–121 when you want
the *exact* single-head position-free operator and `MetastableKGramEngine`'s
aggregated-flux exploration. Cells 136–140 when you want the full model. They optimize
**different operators**, so their fixed points are not comparable — only the JSDs of §2.13
put them on one axis.

**"How do I study a whole generation, not one snapshot?"**
→ Cell 79. It slides a window of `effective_context_window_size = 200` tokens along a
generated trajectory at `positions = range(20, 830, 20)`, rebuilds `π` from the local
K-gram histogram, and re-runs the full MSM pipeline at each position, recording
`λ₀`, `λ₁`, implied timescales, SCC size, QSD support size, and `JSD(QSD, π_context)`.
Cell 81 then builds a pairwise JSD affinity matrix between the QSDs at different positions
— this is how you see topic transitions as *changes in the attractor*, not just in the text.

---

## 4. Developer Guide & Execution Details

### 4.1 Required execution order

Cells 3 → 5 → 7 → 9 → 10 → 11 → 13 are pure definitions and must all run first.
Cell **108** is a fourth definitions-only cell (the approximation benchmark, the two
context estimators and `MultiTimescaleMeanFieldForwardPass`) and depends on cells 9 and 10;
it can be run at any point after them. Cell **113** is a fifth
(`ExactSemanticHeadForwardPass`, §2.10) and depends on 10 and 100. Cell 115 additionally
needs `zero_head_hook` / `remove_pos_embed_hook` from cell 7 and a `corpus` — cell 110's
will do.
Cell 15 loads the model. Everything after that assumes `model`, `device`, and the
definitions above are live. Beyond that the notebook is **not** linearly runnable —
sections 50–58, 59–60, 61–66, 67–70, 71–82, 88–106, 107–111, 112–115, 116–121, 122–127,
128–131, 132–135 and 136–140 are alternative experiments
that each redefine overlapping globals (`losses`, `pi_vals`, `pi_keys`, `heads`, `K_list`,
`temperature`, `p`/`top_p`, `chunk_size`, `forward_pass_kwargs`). Pick one lane and run it
top to bottom. The benchmark lane is 108 (definitions) → 110 (run) → 111 (inspect); cell
110 rebuilds `forward_pass_kwargs` from scratch, so it will overwrite whatever cell 72 or
cell 80 left behind. The multi-timescale lane is 108 (definitions) → 129 (profiling, builds
`mtmf_full`) → 131 (ablations) or 137 (wide optimization) → 138 (save) → 140 (diagnostics);
it needs nothing from cells 61–82 at all, because `MultiTimescaleMeanFieldForwardPass`
carries its own physics (§2.11).

### 4.2 The global-state landmines

**⚠ `torch.set_grad_enabled(False)` in cell 15.**
This is set globally right after model loading. Every gradient-based cell (52, 54, 60, 68)
will fail at `loss.backward()` until you re-enable it. There is no cell in the notebook
that turns it back on — you must do it manually.

**⚠ `model.cfg.use_attn_result = True` in cell 15.**
Materializes per-head `hook_result` tensors of shape `[batch, seq, n_heads, d_model]`.
Needed by the DLA cells (24–28) and the `hook_result` knockout in cell 22, but it
multiplies attention activation memory by `n_heads`. Turn it off before any large
profiling run.

**⚠ Model/pipeline mismatch.** Cell 15 loads `meta-llama/Llama-3.2-1B-Instruct`, but
`single_layer_forward` and `abstract_forward_pass` read `model.W_pos`. TransformerLens
**does not create a `pos_embed` module at all** when `cfg.positional_embedding_type ==
"rotary"` (see `HookedTransformer.__init__`), so `model.W_pos` raises `AttributeError` on
Llama. The K-gram pipeline as written requires a model with **learned absolute positional
embeddings** — GPT-2, `attn-only-{1,2}l`, `gelu-2l`, `tiny-stories-1L-21M`, Pythia. The
commented-out `model_name` lines in cell 15 and the `### tinystories-1L` block in cell 72
are the configurations these sections were actually developed against. Llama is the right
choice only for the profiling sections (61–66) and `profile_deep_heads_*`.

**⚠ `fold_ln=False` is mandatory — except for cells 112–115.** `extract_frozen_sigma`, the
LN linearization inside `single_layer_forward`, and the `ln1.w` access all assume LayerNorm
weights are still present as separate modules. Loading with the default `fold_ln=True`
silently changes the semantics. The exception runs the other way: `single_layer_forward`
applies γ but **never β**, so `ExactSemanticHeadForwardPass` (§2.10) requires the
`fold_ln=True` model, where β has been folded into `b_{Q,K,V}`, and its constructor raises
otherwise. The two lanes want different loads; reload the model when you switch.

**⚠ `L_ctx_list` / `C_far_list` must be sliced to the ACTIVE heads.**
`profile_thermodynamic_heads` returns one entry per head in the model;
`single_layer_forward` indexes by position within `active_heads`. With 1 active head and
8 profiled entries the mismatch **broadcasts silently** and the abstraction sums the same
head eight times with eight different horizons — a normalized, plausible-looking, wrong
answer. With 2 active heads it raises an opaque shape error instead. Cell 72's live config
has this bug. Always slice:
`L_ctx_list = [L_ctx_all[l][active_heads_list[l]] for l in range(model.cfg.n_layers)]`.
`PartitionFunctionApproximateForwardPass` (cell 108) validates this and refuses to
construct; see §2.9.6.

**⚠ Hardcoded vocabulary size.** `pi_from_context` and `pi_t_from_context` default to
`vocab_size=48262`. That is not Llama's (128256) nor GPT-2's (50257). Any call that
doesn't pass `vocab_size` explicitly produces a wrong-length `π`. Cells 19, 54, 57, 83–85
all call it without the argument.

**⚠ Variable shadowing: `p`.** In cells 52/54/57 the nucleus parameter is named `p`, and
the same cells run `for p in model.parameters(): p.requires_grad = False` **before**
assigning `p = 0.9`. The order happens to be safe as written, but any reordering silently
turns the nucleus threshold into a `torch.nn.Parameter`.

**⚠ `loss_full` / `losses_full` staleness.** In cells 52/54/57 the expensive validation
loss is computed only when `i_iteration % val_iterations == 0`, but appended to the history
list **every** iteration. The plotted "full loss" curve is therefore a step function of
stale values, not a per-iteration measurement. It also means a `NameError` if you ever set
`val_iterations` such that iteration 0 is skipped.

**⚠ `N_union_list` is a module-level accumulator.** Defined at the top of cells 60 and 68,
appended to inside `compute_aligned_jsd` (cell-60 definition). It is **never cleared
between runs** — re-running an optimization loop without re-running the setup produces a
concatenated history. Cell 69 plots it.

### 4.3 Cell 60 is legacy and will crash

Cell 60 ("Multi-Layer + Positional Black Box Optimization") calls:

```python
compute_stationary_distribution(
    top_pi_vals, top_pi_keys, N_tracking, K_pruning, model,
    K_list, sem_heads_list, pos_heads_list,
    M_iterations, temperature, top_p
)
```

against the **new** signature, where argument 6 is `M` and arguments 7–13 are
`K_list, active_heads_list, ln_avg_sigma_list, L_ctx_list, C_far_list, sink_k_list, sink_v_list`.
`M_iterations` lands in `C_far_list`, `K_list` lands in `M`, and so on. It will fail.

To use cell 60 you must uncomment the old `single_layer_forward`, `abstract_forward_pass`,
`single_power_iteration_step`, and `compute_stationary_distribution` in cell 10 and comment
out the new ones. **Cell 68 is the maintained equivalent** and should be preferred.

Other scratch cells reference undefined names and are dead as written:
cell 29 (`slow_dla`, `fast_dla`), cell 46 (`sem_stable_tokens`).

### 4.4 Data-format requirements

- **`context_keys` / `query_keys`**: `torch.long`, shape `[N, S_init]`, on `model.cfg.device`.
  Every row must have exactly `S_init = sum(K_list) − len(K_list) + 1` entries. If you
  change `K_list`, recompute `S_init` **and** regenerate all K-grams.
- **`context_vals`**: float, shape `[N]`, must sum to 1. The pipeline renormalizes in
  several places but `single_layer_forward` uses it raw as an integration measure.
- **`corpus_tokens`** for `profile_deep_heads_empirical`: a flat 1-D tensor with **no
  special tokens** (`load_and_tokenize_continuous_corpus` enforces
  `add_special_tokens=False`). The function chunks it into `seq_len − 1` pieces and
  prepends BOS itself. It raises if the corpus is shorter than `ensemble_size × (seq_len−1)`.
- **`prompts`** for `profile_deep_heads_empirical_instruct` / `_multigpu`: `[ensemble_size,
  seq_len]`, already chat-templated and BOS-bearing. Length is validated.
- **`get_kgram_distribution_from_tokens(tokens, K, offset, pad_id)`**: returns
  `(values [N_active], unique_kgrams [N_active, K])`. If `offset < K − 1` it left-pads with
  `pad_id` via `torch.full((K − offset − 1,), pad_id)` — **created without a `device`
  argument**, so it lands on CPU and will raise on a CUDA `tokens` tensor. Pass `offset >=
  K − 1` (cell 72 passes `offset=S_init`) or move tokens to CPU first.
- **`corpus`** for `ApproximationQualityBenchmark.run`: a plain `List[str]` of raw text.
  Each string is tokenized with `prepend_bos=True` and chopped to
  `min(max_tokens, model.cfg.n_ctx)`. Texts yielding fewer than `min_position + 1` tokens
  are reported in `results["skipped"]`, not silently dropped — with `min_position=700` a
  corpus filtered at `len(text) > 5000` characters is about right for English.
- **`forward_pass_kwargs`**: the dict handed to `MetastableKGramEngine` and
  `extract_substochastic_matrix`. It must contain exactly the non-positional parameters of
  `abstract_forward_pass`: `K_list, active_heads_list, ln_avg_sigma_list, L_ctx_list,
  C_far_list, sink_k_list, sink_v_list, temperature, top_p`. It is splatted with `**`, so a
  typo becomes an opaque `TypeError`.

### 4.5 Research hacks worth knowing before you "fix" them

1. **`position_offset = 100`** in `abstract_forward_pass`. Positional embeddings are read
   from `model.W_pos[100 : 100 + S_init]` rather than from position 0. Early positions have
   anomalous embeddings dominated by the sink; offsetting puts the abstraction in the
   "bulk" of the positional manifold. The same offset appears as a default in
   `estimate_{c_far,L_ctx}_for_head`.
2. **Dummy positional embeddings before LN.** Marked `### ADDED DUMMY POSITION EMBEDDING`
   in two places. Deliberate: a pure-semantic residual has out-of-distribution LN statistics.
3. **`score_matrix[:, 0] = -1e4`** in `estimate_L_ctx_for_head` ("Banish the Attention Sink").
4. **`threshold = μ_far + 10·(σ_far + 1e-7)`** in `profile_thermodynamic_heads` — a 10σ
   cut, versus 3σ in `estimate_L_ctx_for_head`. The post-softmax ensemble is much less
   noisy, so the stricter cut is intentional, not a typo.
5. **`L_ctx` has a hand-tuned buffer**: `active_distances.max() + 1 + 2 [+ K_i]`. The `+2`
   is an explicit safety margin to catch the tail; there is a `# TODO` questioning it.
6. **`L_ctx_list` may be hand-written.** Cell 72's live TinyStories config hardcodes
   `torch.tensor([40, 100, 150, 100, 40, ...])` rather than calling a profiler — these
   were read off plots. `C_far_list` is computed analytically in the same cell. Mixing
   hand-set and computed values in one config is normal here.
7. **`torch.relu(E[X²] − μ²)`** before `sqrt` in every profiler — guards against tiny
   negative variances from float64 cancellation.
8. **Sink self-loop.** `extract_substochastic_matrix` always appends
   `P[sink, sink] = 1.0`, making the matrix a proper absorbing chain of size `N+1`.
   `MetastableMSMAnalyzer` then slices the sink back off (`P_full_csr[:sink_idx, :sink_idx]`)
   before Tarjan — the sink exists for conservation bookkeeping, not for the spectrum.
9. **Tarjan shear.** `MetastableMSMAnalyzer.__init__` keeps **only the largest strongly
   connected component**. This routinely discards a large fraction of the discovered
   support. It is required for Doob's h-transform to produce an exactly stochastic matrix,
   but it means `analyzer.core_keys` ≠ `results_power["closed_support_keys"]`. Cell 73
   explicitly checks `initial_k_grams_in_core` — if your seed fell outside the core, the
   analysis is about a basin your seed does not belong to.
10. **ARPACK fallback.** `perform_spectral_decomposition` switches to dense LAPACK when
    `core_size < max(k_eigenvalues + 2, 20)`, because ARPACK fails for `k ≥ N − 1`. Small
    cores are common and this path fires often.
11. **`epsilon_dijkstra · N_init ≥ 1.0` raises.** `S_max = −log(ε · N_init)`, so a large
    seed set with a loose epsilon makes the barrier non-positive. Scale `ε` down as your
    seed set grows.

### 4.6 Performance and memory notes

- `suffix_chunk_size` (default 2048) controls a dense `[chunk, vocab]` float32 accumulator
  — roughly 411 MB at vocab ≈ 50k. This is usually the peak allocation in the power iteration.
- `query_batch_size` (default 1024) controls the residual-stream batch inside
  `abstract_forward_pass`; peak there scales with `query_batch_size × S_init × d_model ×
  n_heads`. Lower this first if you OOM with a long `S_init`.
- `chunk_size` in the 1-layer path is the number of **query tokens** (top-k) in
  `_topk`, but the number of **vocabulary rows per chunk** in the full versions. Same name,
  different meaning across the two functions — a genuine footgun.
- The `for d in range(seq_len)` loop inside the empirical profilers runs once per layer per
  batch — it dominates runtime at `seq_len = 1024`, not the forward pass.
- In the benchmark, cost per position is dominated by the global term inside
  `single_layer_forward`, which is `O(n_heads · N_q · S_out · N_c)` with `N_q = 1`. `N_c`
  is set by the estimator, so `UnigramContextEstimator(top_n=...)` is the knob to reach
  for if a benchmark run OOMs — not `query_batch_size`, which is irrelevant at one query.
- `store_probs="full"` costs ≈ `2 · d_vocab · 4` bytes per evaluated position on CPU
  (~386 KB at `d_vocab = 48262`). Switch to `"topk"` beyond a few hundred positions.
- The real forward pass is a full `model(tokens[:t+1])` at `t ≈ 700–1000`, i.e. one
  ordinary forward pass per position. On a 1-layer model the benchmark runs at tens of
  texts per second; the concrete model, not the abstraction, is usually the bottleneck.
- `profile_deep_heads_multigpu` uses `backend="loky"` specifically because it uses
  cloudpickle and therefore survives being defined in a notebook cell; the default
  multiprocessing backend fails to pickle the closure.
- Every profiler uses `names_filter` to cache only what it needs, and the instruct/multigpu
  variants add `return_type=None` to skip the unembedding projection entirely.

### 4.7 External dependencies beyond the usual stack

Installed by cell 1: `transformer_lens`, `circuitsvis`, `deeptime`, `pygpcca` (conda),
`infomap` (conda), `geomloss[full]`, `pydiffmap`, `joblib`, `fast-langdetect`.

Actually used by the live code paths: `transformer_lens`, `einops`, `deeptime.markov.msm`
(PCCA+), `scipy.{sparse, linalg, sparse.linalg, sparse.csgraph}`, `joblib` (multi-GPU
profiling), `datasets`, `sklearn.decomposition.PCA`, `seaborn`.
`pygpcca`, `Infomap`, `geomloss.SamplesLoss`, `fast_langdetect`, and the sklearn regression
imports are imported at the top but unused in this notebook — inherited from the sibling
`feedback_circuits_transformerlens_experiments.ipynb`. `DEEPTIME_AVAILABLE` is set to a
literal `True` rather than guarded by a try/except, so a missing `deeptime` fails at import
time, not at the `run_conditioned_pcca` check.

Cell 3 prepends `/home/galk/LanguageDynamics/src` to `sys.path` — an absolute path that
must be edited on any other machine. `models_path` is likewise absolute and unused here.

### 4.8 Checklist before running the K-gram pipeline end-to-end

1. Model has **learned absolute positional embeddings** and was loaded with `fold_ln=False`.
2. `torch.set_grad_enabled(True)` if you are going to optimize.
3. `K_list`, `active_heads_list` chosen; `S_init = sum(K_list) − len(K_list) + 1` recomputed.
4. `sink_k_list` / `sink_v_list` from `extract_bos_sink`.
5. `ln_avg_sigma_list` from `extract_frozen_sigma` per layer (needs `dummy_tokens` of
   length `model.cfg.n_ctx`).
6. `L_ctx_list` / `C_far_list` — profiled, estimated, or hand-set. Must be per-layer
   tensors of length `len(active_heads_list[l])`, on the model device.
7. `forward_pass_kwargs` assembled with exactly the nine expected keys.
8. `pi_keys` / `pi_vals` built from a real context via `get_kgram_distribution_from_tokens`
   with `K = S_init`, deduplicated, normalized.
9. `initial_k_grams` shaped `[*, S_init]` — typically the last `S_init` tokens of the prompt.
10. Watch the internal/frontier/leakage audit line; only trust results where leakage is
    small and the frontier has gone quiet.
11. After `MetastableMSMAnalyzer`, check `initial_k_grams_in_core` before interpreting
    anything.
12. **Before trusting any of it**, run the approximation quality benchmark (§2.9, §5) with
    the *same* `forward_pass_kwargs` and confirm the mean JSD beats
    `jsd_baseline_context_unigram`.

### 4.9 Checklist before running the multi-timescale lane

This lane shares almost nothing with §4.8 — `MultiTimescaleMeanFieldForwardPass` carries
its own physics and does not read `forward_pass_kwargs` at all.

1. **`n_layers == 1`**, learned absolute positional embeddings, loaded with `fold_ln=False`
   (the class applies `ln1.w` *and* `ln1.b`, so either load works, but the rest of the
   notebook wants `fold_ln=False`). An MLP is fine.
2. Cell 108 has been run, and `corpus` exists.
3. Cell 129 has been run: `mtmf_full`, `L_ctx_used`, `K_WINDOW`, `T_STAR` are live.
   **Look at the profile plot** before trusting `L_ctx` — a head whose measured tail is not
   straight on a log axis has no geometric horizon, and the printed clamp notice tells you
   which heads were forced into `[K+2, T_STAR]`.
4. `K ≤ t* < n_ctx` and `L_ctx[h] > K + 1` for every head (both enforced in `__init__`).
5. For a benchmark: `real_hooks = []`, `top_p = 1.0`, and `model.cfg.use_attn_result = False`.
6. For an optimization: `torch.set_grad_enabled(True)`, all model parameters
   `requires_grad = False`, and `MIN_POSITION` large enough that the slowest head is
   saturated (§6, `MIN_POSITION`).
7. Re-run the acceptance test (§2.11.8 / cell 118) if you have touched `_forward_chunk`.
8. **Before trusting any fixed point from this lane**, read cell 131's `full` row against
   `jsd_baseline_context_unigram`. The fixed point is a property of the operator, and the
   ablation table is the only evidence that the operator is the model.

---

## 5. Extending the Approximation Benchmark

The benchmark has exactly **two** extension points, and they are independent. Adding a new
hypothesis means writing one subclass; nothing else in cells 107–111 changes, and the
metrics, statistics and result format stay comparable across every variant.

```
ContextDistributionEstimator   ── how is the macroscopic state π estimated?
ApproximateForwardPass         ── how is the forward pass itself abstracted?
ApproximationQualityBenchmark  ── fixed. Do not subclass unless you are changing the
                                  *measurement* (different divergence, different position
                                  sampling), not the thing being measured.
```

### 5.1 A new context estimator

Subclass when the question is *"what should π be?"* — a different window, a different
order of statistics, a smoothed or learned estimate, a fixed point imported from cell 68.

```python
class MyContextEstimator(ContextDistributionEstimator):
    name = "my-estimator"                      # shows up in results["config"]["approximation"]

    def estimate(self, tokens, model, S_init):
        """
        Args:
            tokens: [n_context] the context INCLUDING the query token, on any device.
            model:  HookedTransformer (for cfg.d_vocab, cfg.device, tokenizer).
            S_init: required key width, = sum(K_list) - len(K_list) + 1.
        Returns:
            vals: [N_c] float, on model.cfg.device, MUST sum to 1.
            keys: [N_c, S_init] torch.long, on model.cfg.device.
        """
        ...
        return vals, keys
```

Contract, all of which `single_layer_forward` relies on:

1. `vals` sums to 1. It is used raw as an integration measure in `M_global`; an unnormalized
   π silently rescales the global term against the local and sink terms, i.e. it changes
   the effective attention temperature rather than raising an error.
2. `keys` is exactly `S_init` columns wide, `torch.long`, on `model.cfg.device`.
3. `keys` rows should be unique. Duplicates are not detected and simply double-count mass.
4. Set `self.name` — it is the only thing distinguishing two runs in the results.

Worked example — an estimator that mixes the context histogram with a uniform floor
(a crude smoother, so unseen tokens are not exactly zero in the background):

```python
class SmoothedUnigramContextEstimator(UnigramContextEstimator):
    def __init__(self, alpha: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.name = f"smoothed-unigram(alpha={alpha})"

    def _sparsify(self, pi_dense):                       # [d_vocab] -> [N_c], [N_c]
        pi_dense = (1 - self.alpha) * pi_dense + self.alpha / pi_dense.numel()
        return super()._sparsify(pi_dense)
```

`UnigramContextEstimator` deliberately splits `estimate` (build the dense histogram) from
`_sparsify` (threshold / top-n / renormalize) so subclasses can override just one half.

That example also illustrates the trap: a uniform floor lifts **every** token above zero,
so `N_c` jumps to `d_vocab` (48262 here) and the global term explodes — but adding `top_n`
to keep it tractable prunes the floor straight back off, and the run becomes bit-identical
to the unsmoothed one. Verified: `alpha=0.01` with `top_n=2048` gives exactly the same
mean JSD as the baseline. Smoothing the context distribution is only meaningful if you
also let `N_c` grow, i.e. it is a VRAM decision before it is a modelling one. Order matters
too — override `_sparsify` (after thresholding) rather than `estimate` if you want the
floor applied only to the surviving support.

### 5.2 A new approximate forward pass

> Two reference implementations now exist: `ExactSemanticHeadForwardPass` (§2.10), which
> drives `abstract_forward_pass` through a patched model, and
> `MultiTimescaleMeanFieldForwardPass` (§2.11), which implements its own forward pass end
> to end and is also the operator of a fixed-point search. If your variant needs to be
> optimized as well as benchmarked, copy the second one's shape — `forward` /
> `power_iteration_step` / `stationary` alongside `predict` — so the benchmarked and
> optimized operators stay the same object.

Subclass when the question is *"what should the abstraction be?"* — this is where "no
positional embeddings at all", "a finite sliding context window", "attention-only, MLP
bypassed", "head subset X only" live.

```python
class MyApproximateForwardPass(ApproximateForwardPass):
    name = "my-approximation"

    def predict(self, tokens, position):
        """
        Args:
            tokens:   [n_tokens] the full BOS-prepended token sequence for this text.
            position: index of the query token; you are predicting position + 1.
                      You may read tokens[:position+1] and NOTHING beyond it.
        Returns:
            probs: [d_vocab] on model.cfg.device, sums to 1, already temperature/top-p'd.
            info:  dict merged verbatim into the per-text result.
        """
        ...
        return probs, info
```

Contract:

1. **Never read `tokens[position+1:]`.** There is no guard; leaking the answer produces a
   beautiful JSD and a meaningless experiment.
2. Return probabilities, not logits, and apply the same `temperature` / `top_p` the
   benchmark will apply to the real side. Expose them as `self.temperature` /
   `self.top_p` — `ApproximationQualityBenchmark` reads those attributes to default its
   own, which is what keeps the two sides in sync. Use `apply_sampling_transform`.
3. Keys put into `info` must not collide with the benchmark's own result keys
   (`loss_jsd`, `position`, `text`, `real_probs`, …); `info` is merged with `dict.update`
   and would overwrite them.
4. Put anything you will want to plot into `info` — it is the only channel out.

Worked example — the "no positional embeddings" ablation. It reuses the whole
partition-function pass and only zeroes `W_pos`, which is the single cleanest way to ask
"how much of the abstraction's fidelity comes from position?":

```python
class NoPositionalApproximateForwardPass(PartitionFunctionApproximateForwardPass):
    """Identical to the partition-function pass, with W_pos zeroed for the duration."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "no-positional/" + self.name

    def predict(self, tokens, position):
        saved_W_pos = self.model.W_pos.data
        self.model.W_pos.data = torch.zeros_like(saved_W_pos)       # [n_ctx, d_model]
        try:
            return super().predict(tokens, position)
        finally:
            self.model.W_pos.data = saved_W_pos                     # always restore
```

Worked example — a finite sliding context window. Note this is the *forward pass* window
(`K_i`, what the abstraction can see locally), which is a different question from
`UnigramContextEstimator(context_window=...)` (what the *state estimate* is built from).
Sweeping the two separately is how you tell them apart:

```python
def sliding_window_pass(model, base_kwargs, K_list, context_estimator):
    kwargs = dict(base_kwargs, K_list=K_list)
    return PartitionFunctionApproximateForwardPass(
        model=model, context_estimator=context_estimator, forward_pass_kwargs=kwargs)

# ⚠ changing K_list changes S_init = sum(K_list) - len(K_list) + 1, and L_ctx enters
# single_layer_forward as (L_ctx - K_i - 1). Re-profile or re-slice L_ctx/C_far to match,
# and keep L_ctx > K_i + 1 or the global term goes negative.
```

Worked example — **per-forward-pass `L_ctx` / `C_far` / `π`**. `ExactSemanticHeadForwardPass`
(cell 113, §2.10) is the reference for this pattern: nothing says `forward_pass_kwargs` has
to be constant across positions, and for anything whose horizon depends on the realized
prompt length it must not be. It rebuilds `L_ctx_list`, `C_far_list`, `ln_avg_sigma_list`
and the context measure inside `predict`, and patches `model.W_pos` / `model.W_E` around
the call under `try/finally`. Copy its structure whenever the abstraction's parameters are
a function of the prompt rather than of the model.

A genuinely non-`abstract_forward_pass` baseline — useful as a sanity floor — is just as
easy, because the ABC knows nothing about the K-gram machinery:

```python
class TruncatedContextForwardPass(ApproximateForwardPass):
    """Not an abstraction at all: the REAL model on only the last `window` tokens.
    Isolates 'how much does long context matter' from 'how good is the abstraction'."""
    def __init__(self, model, window=64, temperature=1.0, top_p=1.0):
        self.model, self.window = model, window
        self.temperature, self.top_p = temperature, top_p
        self.name = f"truncated-real(window={window})"

    def predict(self, tokens, position):
        prefix = tokens[max(0, position + 1 - self.window): position + 1]   # [<=window]
        logits = self.model(prefix.view(1, -1).to(self.model.cfg.device))   # [1, L, d_vocab]
        probs = apply_sampling_transform(logits[0, -1], self.temperature, self.top_p)
        return probs, {"window": self.window}
```

### 5.3 Changing the measurement itself

Only touch `ApproximationQualityBenchmark` if you are changing what "error" means.

- **A different divergence** — override `evaluate_position`, call `super()`, and add your
  key; then add that key to the `accumulators` dict in `run` so it gets mean/std. The
  statistics machinery keys off that dict and nothing else.
- **Deterministic rather than random positions** — override `sample_positions(n_tokens, rng)`.
  Returning `list(range(min_position, n_tokens, stride))` turns the benchmark into a
  position sweep, which is the natural companion to cell 87's JSD-versus-position plot.
  `n_positions` is then ignored, which is fine, but note `config` will still report it.
- **Comparing against something other than the concrete model** — override `real_probs`.

### 5.4 Attributing the error

The single JSD number is a verdict on the whole abstraction. To find out *which*
approximation is costing you, run the benchmark repeatedly with one ingredient changed,
holding `seed`, `corpus`, `max_tokens`, `min_position` and `n_positions` fixed so every
variant sees identical positions:

| Question | Change |
|---|---|
| Does the mean-field global term help at all? | `C_far_list` → zeros (kills `M_global`) |
| How much does position matter? | `NoPositionalApproximateForwardPass` |
| Is the local window long enough? | sweep `K_list` |
| Is the state estimate the bottleneck, or the forward pass? | swap `UnigramContextEstimator` ↔ `KGramContextEstimator` at fixed `forward_pass_kwargs` |
| Does the model actually need long context? | `TruncatedContextForwardPass` at several windows |
| Are the chosen heads the ones carrying the behaviour? | sweep `active_heads_list` (remembering to slice `L_ctx`/`C_far` — §4.2) |
| How much does the frozen-σ LayerNorm linearization cost? | `ExactSemanticHeadForwardPass(exact_layernorm=False)` vs. `True` (§2.10.3) |
| Is any of the residual error the *machinery* rather than the parameters? | `ExactSemanticHeadForwardPass` — if it does not reach ~1e-12, your preconditions (§2.10.5) are violated, not your `π` |

For a 1-layer, single-head, position-free model the error floor is **zero**, and
`ExactSemanticHeadForwardPass` reaches it. Start every attribution study from that run and
add assumptions back one at a time; each step's JSD increment is that assumption's cost.

The commented-out sweep at the bottom of cell 111 is the template: rebind
`benchmark.approximate_pass` and call `benchmark.run(corpus)` again.

### 5.5 Pitfalls when extending

1. **`torch.set_grad_enabled(False)` (cell 15) is your friend here** — the benchmark is
   pure evaluation and `run` already wraps each position in `torch.no_grad()`. If you have
   re-enabled grad for an optimization lane, the benchmark still works but the STE branch
   inside `abstract_forward_pass` will build graphs you do not need.
2. **Do not compare runs with different `top_p`.** Nucleus filtering truncates both sides
   to a shared support and systematically lowers the JSD. Benchmark at `top_p = 1.0`.
3. **Do not compare runs with different `seed`/`min_position`/`max_tokens`/`n_positions`** —
   the sampled positions change and the corpus means are no longer paired. If you need a
   paired test, the position lists are in `results["per_text"][i]["position"]`; check they
   match before differencing.
4. **Watch `n_skipped`.** A variant that quietly raises on most texts will report an
   excellent mean over the three positions that survived. `summarize_benchmark` prints the
   skip count first for this reason.
5. **A JSD at exactly `ln 2 ≈ 0.6931` is not a bad score, it is a broken variant** —
   it means the supports are disjoint, which in practice means the abstraction collapsed
   (see the multi-layer unigram case in §2.9.4).
6. **`model.cfg.use_attn_result = True`** (cell 15) multiplies attention activation memory
   by `n_heads` in the *real* forward pass. Turn it off before a long benchmark run.
7. **An abstraction that patches model weights needs `approx_hooks = []`.**
   `ExactSemanticHeadForwardPass` swaps `model.W_pos` and `model.W_E` in place because
   `abstract_forward_pass` reads them directly rather than through hook points. Adding
   `approx_hooks` on top layers a second, independent intervention on the same forward
   pass; adding the *same* intervention twice (e.g. `remove_pos_embed_hook`) is harmless
   but misleading in `config`. Always restore patched `.data` in a `finally` — an
   exception mid-`predict` that leaks a rescaled `W_E` silently corrupts every later cell
   in the kernel.
8. **Exactness claims need `fold_ln=True`.** §4.2 tells you to load with `fold_ln=False`
   for the K-gram pipeline; that leaves a learnable `ln1.b` which `single_layer_forward`
   never applies, so the exact class refuses to run. The two lanes want different model
   loads — reload the model between them rather than trusting one.

---

## 6. Manual — every parameter of the wide multi-timescale optimization

The parameters of cells 136–140, one at a time. Each entry answers the same four
questions:

* **Controls** — what the number physically is.
* **Effect** — what moves when you change it, in the optimization *and* in the result.
* **Tuning** — how to pick it.
* **Diagnosis** — how to tell, from the run's own output, that it is wrong.

Everything in §6.1 comes from the profiling cell (129) and is inherited by the
optimization through `mtmf_full`; §6.2–§6.6 are the optimization cell's own knobs.

> **The one-line version.** `MIN_POSITION` and `L_ctx` decide whether the physics is
> right. `N_ACTIVE`, `N_OUTPUT` and `N_TRACKING` decide whether the *numerics* are honest.
> `LR_MAX` and `N_ITERATIONS` decide whether it converged. Nothing else usually matters,
> and three of those five have a printed diagnostic that tells you when they are wrong.

### 6.0 What to touch first

| If you are… | Touch |
|---|---|
| running this for the first time | nothing — run it, then read `top_mass_history` and the loss curve |
| seeing `top_mass` well below 1 | `N_ACTIVE` ↑ |
| seeing the loss plateau above ~1e-2 | `N_OUTPUT` ↑, then `N_ITERATIONS` ↑, then `LR_MAX` ↓ |
| seeing the support grow without bound | `N_TRACKING` ↓ or `EXPLORE_N` ↓ |
| out of VRAM | `QUERY_CHUNK_SIZE` ↓ first, then `K_PRUNING` ↓, then `CTX_TOP_N` set |
| unsure the fixed point means anything | go back to cell 131's ablation table, not to these knobs |

---

### 6.1 The operator's parameters (inherited from cell 129)

These are *not* free parameters of the optimization — they define **which operator** you
are finding a fixed point of. Changing one changes the answer, not the convergence.

#### `K_WINDOW` (→ `mtmf.K`, the K-gram width)

* **Controls** the split between the exactly-resolved fast window and the mean field: the
  last `K` tokens are computed with no approximation, everything older is mean-field. It is
  simultaneously the **state-space width** — π lives on K-grams of exactly this length.
* **Effect.** Larger `K` moves work from the approximate block into the exact one, so the
  abstraction gets more faithful — and the state space gets exponentially larger, the
  exploration slower to close, and the K-gram measure sparser for a fixed `N_TRACKING`.
  `K = 10` on `attn-only-1l` means a "state" is a 10-token phrase, which is already long
  enough that most states are seen once in a prompt.
* **Tuning.** Read it off cell 131's `K=1 / K=8 / K=16` rows: pick the smallest `K` whose
  JSD is within noise of the best. Do **not** raise `K` past any head's horizon without
  also clamping `L_ctx` to `K+2` (§2.13.2).
* **Diagnosis.** `n_states_final` barely above `N_ACTIVE`, and top K-grams that are all
  unique phrases from the prompt with near-equal mass → `K` is too large for the amount of
  mass you are tracking; the measure never aggregates. Conversely a `K` that is too small
  shows up in cell 131, not here.

#### `T_STAR` (→ `mtmf.t_star`)

* **Controls** the dummy absolute query position. Every positional quantity — the window
  embeddings, the sink score, `E_pos(K)` — is measured relative to it.
* **Effect.** It sets *which regime* you are modelling: `t* = 800` means "a query deep in a
  full context". Because `use_real_query_position=False`, the optimization and `predict`
  use the same `t*`, which is what makes the benchmarked operator and the optimized
  operator identical.
* **Tuning.** Pick a position representative of where you care about the dynamics, subject
  to `K ≤ t* < n_ctx`. For a fixed-point study the answer is "deep", because a fixed point
  is a statement about a long context.
* **Diagnosis.** Cell 131's `true query position` row prices the dummy. If that row is far
  from `full`, `t*` is not representative of the benchmark's positions and the fixed point
  describes a regime the model is not in.

#### `L_ctx` / `gamma` (per head)

* **Controls** each head's effective integration horizon, window included;
  `γ_h = 1 − 1/L_ctx[h]`, `Z_far_h = 1/(1−γ_h)`, `W_far_h = E_pos_h(K)·Z_far_h`. One number
  per head fixes the entire mean-field weighting (§2.11.2).
* **Effect.** Large `L_ctx` → the mean-field block dominates the partition function, the
  fixed point is driven by the bulk statistics of the context. Small `L_ctx` → the fast
  window and the sink dominate, and π barely matters. This is the single parameter that
  decides *how state-dependent* `P_π` is, i.e. whether there is an interesting fixed-point
  problem at all.
* **Tuning.** Do not guess it: cell 129 fits it from the model's own `W_pos`. If you must
  override, `uniform L_ctx` in cell 131 tells you how much per-head resolution is worth.
* **Diagnosis.** Three signals, in order. (1) The clamp notice printed by cell 129 — a head
  clamped to `T_STAR` has no fitted horizon and its `L_ctx` is a fiction. (2) The right-hand
  profile panel — a curve that is not straight on a log axis is not geometric, and `L_ctx`
  is a summary of it at best. (3) In the optimization, `far_mass` printed per run: if the
  realized `Z_far` is far below `1/(1−γ)`, the prompt is too short for that head (see
  `MIN_POSITION`).

#### `PROFILE_D_MAX`, `FIT_D_MAX`, `N_PROBE`, `PROBE_SEED`, `L_CTX_CEILING` (cell 129)

* **Controls** the measurement of `E_pos(d)`: how far back it is measured, over what range
  the geometric tail is fitted, how many query tokens the (geometric-mean) average runs
  over, and the ceiling the fitted `L_ctx` is clamped to.
* **Effect.** `FIT_D_MAX` is the one that matters: fitting over a range where the profile
  has already decayed into numerical noise gives a slope of nothing. `N_PROBE` only reduces
  the variance of the median γ.
* **Tuning.** Fit over `[K_WINDOW, FIT_D_MAX]` with `FIT_D_MAX` chosen from the left-hand
  plot as the distance where the curves are still above ~1e-6 of their peak. `N_PROBE` in
  the high hundreds is plenty; the median is robust.
* **Diagnosis.** Re-run with a different `PROBE_SEED` and compare `gamma_fit`. If the
  medians move by more than a few percent, `N_PROBE` is too small. If the dashed fit in the
  right-hand panel does not overlay the solid measurement, no `N_PROBE` will fix it — the
  ansatz is wrong for that head.

---

### 6.2 Corpus and initial conditions

#### `N_TEXTS`

* **Controls** how many texts are optimized — one run per text at `N_POSITIONS_TEXT = 1`.
* **Effect.** Purely statistical: it is the sample size for "what do fixed points reached
  from real prompts look like?". It does not affect any individual run.
* **Tuning.** 10 for a smoke test. For a claim about the *distribution* of fixed points
  (how many distinct basins, how the final loss is distributed), you want the same order as
  the token-level lane's 90.
* **Diagnosis.** The summary block prints `initial` and `final` loss medians and ranges. If
  the range spans orders of magnitude with 10 texts, you cannot say anything about the
  population yet.

#### `MAX_TOKENS`

* **Controls** where each text is chopped.
* **Effect.** It bounds the admissible positions and therefore the realized `far_mass`. It
  **may exceed `n_ctx`**: the real model is never run in this cell, and everything
  positional is measured from `t*`, not from the true index.
* **Tuning.** At least `MIN_POSITION + N_SEPARATION + 1`. Raising it costs nothing but
  tokenization time and admits longer, better-saturated contexts.
* **Diagnosis.** The "only N of M texts reach …" warning means `MAX_TOKENS` (or the
  corpus's length filter) is too small for the `MIN_POSITION` you asked for.

#### `MIN_POSITION` (default `None` → derived)

* **Controls** the earliest context length an initial condition may be taken from. The
  user-facing statement of the requirement *"not too close to the beginning, so that all
  heads are fully active"*.
* **Effect.** This is the **most load-bearing parameter in the cell**. The mean field is
  built from a *discounted* sum, and the discounted sum of a short prompt has not
  saturated: the realized `Z_far` is `(1 − γ^D)/(1 − γ)` rather than `1/(1 − γ)`. Start too
  early and the slowest head is integrating a fraction of the context it is supposed to,
  so the operator you optimize is not the operator you benchmarked.
* **Tuning.** Leave it `None`. The cell derives
  `SATURATION_FACTOR · L_ctx_max + K + 1`; with `SATURATION_FACTOR = 2` the slowest head
  has realized `1 − e⁻² ≈ 86%` of its far-field mass, with 3 it is 95%. Raise the factor if
  you want a stricter saturation, and raise `MAX_TOKENS` with it.
* **Diagnosis.** The cell prints the realized saturation percentage whenever
  `MIN_POSITION < required_position`, which happens exactly when a head's `L_ctx` was
  clamped to `T_STAR` in cell 129 (there is then *no* position in an `n_ctx`-long prompt
  that saturates it). That warning is not cosmetic: treat it as "the slowest head's
  timescale is not measurable on this model", and either drop that head from
  `active_heads` or accept that its mean field is truncated.

#### `SATURATION_FACTOR`

* **Controls** how many integration horizons of context count as "fully active";
  saturation is `1 − e^(−factor)`.
* **Effect / Tuning / Diagnosis.** See `MIN_POSITION`. 2 is a reasonable default, 3 is
  strict, below 1.5 the mean field of the slowest head is visibly truncated.

#### `N_POSITIONS_TEXT`, `N_SEPARATION`, `POSITION_SEED`

* **Controls** how many independent initial conditions are drawn per text, the minimum
  token separation between them, and the RNG seed. Uses `sample_optimization_positions`
  from cell 89, so the sampling is identical to the token-level lane.
* **Effect.** More positions per text is cheaper than more texts (one tokenization, one
  file) but the initial conditions are correlated — overlapping contexts share most of
  their far field. `N_SEPARATION` bounds that correlation.
* **Tuning.** 1 position per text for an unbiased sample of prompts. If you specifically
  want "does the fixed point depend on *where* in this text I start?", raise
  `N_POSITIONS_TEXT` and set `N_SEPARATION` to at least `L_ctx_max` so the two far fields
  barely overlap.
* **Diagnosis.** Two runs from the same text converging to the same π with
  `N_SEPARATION < L_ctx_max` is not evidence of a basin — it is evidence that they read
  nearly the same context. Compare `context_tokens` before concluding anything.

#### `INIT_HEAD` / `INIT_GAMMA` (derived: `argmax L_ctx`)

* **Controls** which head's discount builds the initial K-gram measure.
* **Effect.** A larger γ spreads the initial mass further back into the prompt (a flatter,
  higher-entropy π₀); a smaller γ concentrates it near the window edge. It only sets the
  *starting point* — the fixed point itself does not depend on it, if the optimization
  converges.
* **Tuning.** The derived choice (the longest timescale) is the right default: it gives the
  broadest support, which is the most forgiving seed for the exploration phase.
* **Diagnosis.** Cell 140's "entropy (K-gram)" line, initial vs final. A π₀ with much lower
  entropy than π_final means the optimization had to *discover* most of its support, which
  is slow and exploration-limited; consider a longer timescale or a larger `EXPLORE_N`.

---

### 6.3 The sparse forward passes

These are the numerics. None of them changes the operator; all of them change how
faithfully it is evaluated, and every one has a printed diagnostic.

#### `N_ACTIVE` — queries per forward pass

* **Controls** how many K-grams are used as queries in each `forward` call: the top
  `N_ACTIVE` states of π, renormalized to sum to 1.
* **Effect.** This is the **top-k truncation of §2.1, on the K-gram state space**. States
  outside the top `N_ACTIVE` contribute nothing to `πP` and receive **exactly zero
  gradient**. The loss landscape is discontinuous wherever top-k membership changes. It is
  also the dominant term in the per-iteration cost: the `[H, N_q, N_c]` score tensor and
  the `[N_q, |V|]` top-k both scale linearly in it.
* **Tuning.** Raise it until `top_mass` sits near 1. 256 is the notebook's standing value
  (cells 117/119); with `K = 10` and a broad π you may need more, because K-gram mass is
  spread over far more states than unigram mass.
* **Diagnosis.** **`top_mass_history` is the diagnostic, and it is plotted** (cell 140,
  panel 3). `top_mass` is the fraction of π carried by the active query set. Before the
  renormalization in phase 2 it is the effective rescaling of the context: at
  `top_mass = 0.7` you are evaluating the operator as if the far field were 30% shorter
  than it is. Want ≳ 0.95. If it *falls* over the run, π is spreading faster than
  `N_ACTIVE` can follow.

#### `N_OUTPUT` — states kept in `πP`

* **Controls** the size of the support of `πP` after each power-iteration step.
* **Effect.** Subtle and important. The loss is `compute_aligned_jsd(π, πP)`, which puts
  states missing from one side at `1e-10` and renormalizes. So **any mass of π that lies
  outside `πP`'s `N_OUTPUT` states is a JSD floor the optimizer cannot get under** by
  redistributing; it can only get under it by *concentrating π onto those states*. That is
  a real pressure on the answer, not just on the convergence rate: too small an `N_OUTPUT`
  biases the fixed point toward being more concentrated than it should be.
* **Tuning.** Keep `N_OUTPUT ≈ N_ACTIVE` (the notebook's default) and let `N_TRACKING`
  exceed both — the gradient then genuinely has to concentrate π, which is the intended
  dynamics. If you want a broad fixed point, raise `N_OUTPUT` and `N_ACTIVE` together.
* **Diagnosis.** A loss curve that drops fast and then plateaus on a hard floor, while
  `n_states_final` sits pinned near `N_OUTPUT` and the K-gram rank–mass curve (cell 140,
  panel 4) shows a cliff exactly at rank `N_OUTPUT`. That cliff is the parameter, not the
  physics.

#### `K_PRUNING` — successors kept per query

* **Controls** how many next-tokens survive per query inside `power_iteration_step`
  (`topk` over the `[N_q, |V|]` transition rows, before the key shift).
* **Effect.** Bounds the branching factor of the discovered dynamics. Too small and the
  support can only ever grow along the model's most likely continuations, which biases the
  measure toward low-entropy paths. The intermediate tensors are
  `N_ACTIVE × K_PRUNING × K` integers, so it is the main *memory* term of the power
  iteration.
* **Tuning.** 512 is the notebook default. Compare it against the model's actual branching
  factor at these positions — cell 101 (`measure_branching_factor`) computes it. If the
  typical nucleus is 50 tokens wide, 512 is generous and can be lowered for speed.
* **Diagnosis.** Raise `K_PRUNING` 2× and re-run one text. If the final loss or the top
  K-grams move, it was binding.

#### `N_TRACKING` — cap on `|support(π)|`

* **Controls** how many K-gram states survive the prune-and-migrate step at the end of
  each iteration, and also caps the exploration's contribution.
* **Effect.** The memory of the run. It bounds the union support (and hence the size of the
  `pi_logits` parameter, the mean-field `scatter_add`, and the `torch.unique` in the union
  merge). Too small and the exploration's discoveries are thrown away before the gradient
  can act on them; too large and you pay for tens of thousands of states carrying `2·eps`.
* **Tuning.** 8192 is the default here. Cell 117 uses `d_vocab // 4` for a *unigram* state
  space; that is far too generous for K-grams, where almost all states have negligible
  mass. Set it a few times `N_ACTIVE`.
* **Diagnosis.** `n_union_history` (cell 140, panel 2). If it saturates flat at
  `N_TRACKING + EXPLORE_N`, you are pruning every iteration and the cap is binding — check
  that the pruned mass is negligible by comparing `n_states_final` with the rank–mass
  curve's tail. If it is still climbing at the end of the run, the support has not closed
  and the fixed point is not one.

#### `QUERY_CHUNK_SIZE`

* **Controls** how many queries `forward` processes at a time when forming the
  `[H, N_q, N_c]` mean-field score tensor. Set on a `copy.copy` of `mtmf_full`, so the
  benchmarked object is not mutated.
* **Effect.** Pure memory/speed trade; the result is bit-identical (the chunks are
  concatenated, not reduced). Peak VRAM for that tensor is
  `H × QUERY_CHUNK_SIZE × N_c × 4 bytes`, times a small constant for the intermediates.
* **Tuning.** Largest value that fits. `None` (one shot) is fastest when `N_c` is small.
* **Diagnosis.** OOM inside `_forward_chunk` → lower it. Nothing else changes.

#### `CTX_TOP_N` — cap on the mean field's support

* **Controls** how many unigram states the mean field keeps (`None` = all of them).
* **Effect.** The mean field is the `N_c` axis of every score tensor, so this is the other
  memory lever. **But it is not free**: truncating π's unigram marginal and renormalizing
  redistributes the dropped mass onto the survivors, which is a different operator. The
  same warning as §2.12.3, one level up.
* **Tuning.** Leave it `None` unless memory forces it. If you must set it, make it large
  enough that the dropped mass is ≪ the loss you are trying to reach.
* **Diagnosis.** The `|mean field|` figure in the periodic log line tells you what `N_c`
  actually is. If it is a few thousand, `None` is cheap and you should not be truncating.

---

### 6.4 The gradient loop

#### `N_ITERATIONS`

* **Controls** gradient steps per run, and (through the cosine schedule) the learning-rate
  trajectory — the schedule is `N_ITERATIONS`-normalized, so halving it does not just stop
  early, it **anneals twice as fast**.
* **Effect.** Wall-clock is linear in it.
* **Tuning.** 1000 here (10 runs); cell 117 uses 3000 for a single run. Raise it before you
  raise `LR_MAX`.
* **Diagnosis.** The loss curve in panel 1. Still descending at the right-hand edge → too
  few. Flat for the last half → you are paying for nothing, and `best_iteration` will tell
  you exactly where it stopped improving.

#### `LR_MAX`, `LR_MIN`

* **Controls** the cosine-annealed step size on `pi_logits`, from `LR_MAX` at iteration 0 to
  `LR_MIN` at `N_ITERATIONS`.
* **Effect.** After the Sherman–Morrison natural-gradient correction and `clip_grad_norm_`
  to 1.0, the step is a **bounded** move in the natural (Fisher) metric — so `LR_MAX = 1.0`
  is a step of order one *in KL*, not in Euclidean distance. That is why these values look
  enormous compared to a normal SGD learning rate.
* **Tuning.** `1e0 → 1e-1` is the notebook's standing pair across cells 52, 89 and 117. Do
  not change it before you have ruled out `N_OUTPUT` and `N_ACTIVE` as the cause of a bad
  loss.
* **Diagnosis.** A loss curve that is noisy and non-monotonic at the start and only settles
  once the schedule anneals → `LR_MAX` too high. A loss that descends smoothly but has not
  arrived by the end → `LR_MAX` too low *or* `N_ITERATIONS` too small; prefer raising
  iterations, since the clipped natural gradient makes larger steps mostly wasted motion.

#### `M_ITERATIONS` — power iterations per gradient step

* **Controls** how many times `P_π` is applied inside the differentiated `stationary` call.
  The mean field is derived **once, outside** the loop, so it stays frozen for all `M`
  steps while remaining differentiable in π.
* **Effect.** `M = 1` makes the loss `JSD(π, πP)`, a one-step self-consistency residual.
  `M > 1` makes it `JSD(π, πP^M)`, which is a **weaker** condition (it tolerates period-`M`
  cycles) but has a better-conditioned gradient, and costs `M` checkpointed forward passes
  plus their recomputation in backward.
* **Tuning.** Keep `M = 1`. Raise it only if you suspect the optimizer is being deflected
  by short-lived transients, and then interpret the result as a statement about `P^M`.
* **Diagnosis.** If `M > 1` gives a much lower loss than `M = 1` from the same start, you
  have found a cycle, not a fixed point.

#### `LOSS_SCALE`

* **Controls** the multiplier on the JSD; `1e5 / d_vocab` here, matching cell 117.
* **Effect.** **It is redundant with `LR_MAX`** — the natural gradient is linear in the
  loss, and the only non-linearity is `clip_grad_norm_`. So `LOSS_SCALE` mostly decides
  *whether the clip is active*, and therefore whether you are doing natural-gradient
  descent or normalized natural-gradient descent.
* **Tuning.** Leave it. Its real job is making the printed loss a readable number that is
  comparable with cells 68, 117 and 119.
* **Diagnosis.** Print `grad_max` (the periodic log line does). If it is pinned at the clip
  bound for the entire run, every step has the same length and the loss magnitude is doing
  nothing — that is usually fine, but it means `LR` is the *only* step-size control.

#### `LMBDA`, `MAX_GRAD_NORM`

* **Controls** the damping in `A⁻¹ = 1/(π + λ)` and the gradient-norm clip.
* **Effect.** π is extremely sparse, so `1/π` explodes for the `2·eps` states the union
  merge just introduced. `λ = 1e-10` caps that at `1e10`; the clip then bounds the step.
  Both are load-bearing — without them a single dead state can dominate the update.
* **Tuning.** Do not. They are the same values as cells 52, 89 and 117.
* **Diagnosis.** `grad_max` ≫ `grad_mean` by many orders of magnitude, or a non-finite
  gradient warning (the loop breaks and keeps the best iterate), means the damping is being
  overwhelmed — usually because `2·eps` states are entering the top-`N_ACTIVE` query set.
  Raise `N_TRACKING`'s pruning pressure or lower `EXPLORE_N` rather than raising `λ`.

#### `PRINT_ITERATIONS`

* **Controls** logging cadence only. No effect on the result.

---

### 6.5 The exploration phase

Exploration is what lets the support **grow**; the gradient can only redistribute mass over
the states it is given. All of these are no-grad.

#### `EXPLORE_EVERY`, `EXPLORE_DEEP_M`, `EXPLORE_SHALLOW_M`

* **Controls** the cadence: `EXPLORE_SHALLOW_M` power iterations every step, and
  `EXPLORE_DEEP_M` every `EXPLORE_EVERY` steps.
* **Effect.** A shallow step adds the immediate successors of the current top states — one
  token of look-ahead. A deep step runs the operator to (approximate) convergence and can
  discover states many transitions away, at `EXPLORE_DEEP_M ×` the cost.
* **Tuning.** `1 / 10 / 50` is a reasonable default (cell 117 uses `1 / 30 / 50` with the
  aggregated-flux engine). Raise `EXPLORE_DEEP_M` if the support is still opening up late
  in the run; lower `EXPLORE_EVERY` if the loss drops in visible steps synchronized with
  the deep explorations.
* **Diagnosis.** `explore_size_history` and `n_union_history` plotted together (panel 2).
  Sawtooth spikes at multiples of `EXPLORE_EVERY` that do not decay → the deep exploration
  keeps finding new states and the support has not closed. Both curves flat from early on →
  exploration is doing nothing and you can turn the cadence down.

#### `EXPLORE_N`

* **Controls** how many states the exploration keeps per step before the union merge.
* **Effect.** The width of the frontier. Larger means more candidates offered to the
  gradient each step, a larger union support, and more `2·eps` states diluting π.
* **Tuning.** A few times `N_ACTIVE`. 2048 with `N_ACTIVE = 256` is a wide frontier.
* **Diagnosis.** If `n_union_history` is dominated by `EXPLORE_N` (i.e. `N_union ≈
  |support(π)| + EXPLORE_N` every step) and the loss is not improving, the frontier is
  being generated and discarded — lower `EXPLORE_N` or raise `ETA_EXPLORE`.

#### `ETA_EXPLORE`

* **Controls** the damped-Picard blend: newly discovered states enter π with
  `ETA_EXPLORE × their exploration flux` instead of `2·eps`.
* **Effect.** **`0.0` (the default) is pure gradient descent on JSD.** Any positive value
  makes the method a **hybrid** — gradient descent plus a Picard step — which converges
  faster but changes what the fixed point is a fixed point *of*: you are no longer purely
  minimizing the self-consistency residual.
* **Tuning.** Keep it 0 for a clean result. Raise it (1e-2 … 1e-1) only when the gradient
  demonstrably cannot lift new states out of `2·eps` in the iterations you have, and say so
  when reporting the result.
* **Diagnosis.** Compare a run at `ETA_EXPLORE = 0` and at `0.05`. If the final π differs
  materially, the answer depends on the hybrid and the pure-gradient run is the one to
  report.

---

### 6.6 Storage and diagnostics

#### `STORE_TOP_K`

* **Controls** how many K-gram states are kept per stored measure (`pi_init`, `pi_final`,
  `pi_best`). Note the **unigram** measures are stored in full, as sparse COO.
* **Effect.** File size and the resolution of any later tail analysis. `4096` states × `K`
  int64 ≈ 0.3 MB per measure.
* **Tuning.** Raise it if you intend to study the tail of π (rank–mass slopes, support
  overlap between runs); the default is comfortably above `N_OUTPUT`.
* **Diagnosis.** `n_states_final > STORE_TOP_K` in the result dict means the stored measure
  is truncated relative to the one that was optimized — the rank–mass curve will end at
  `STORE_TOP_K` rather than at the true support size.

#### `EPS`

* **Controls** the floor mass given to states entering the union support (`2·eps`).
* **Effect / Tuning.** `1e-10`, same as cells 60 and 117. It interacts with `LMBDA`: a
  state at `2e-10` gets a natural-gradient prefactor of ~`5e9`. Do not lower it.

#### Cell 140: `RUN_INDEX`, `TOP_KGRAMS`, `TOP_UNIGRAMS`, `LABEL_CHARS`, `BAR_HEIGHT`

Presentation only — which run to summarize, how many bars per histogram, and how the
decoded K-gram labels are truncated. None of them touches the data.

---

### 6.7 A tuning recipe

1. **Fix the operator first.** Run cell 129, look at the profile plot, and run cell 131.
   If `full` does not comfortably beat `jsd_baseline_context_unigram`, stop — no setting in
   §6.2–§6.6 will make a fixed point of a bad operator mean anything.
2. **Run one text** with the defaults and `N_ITERATIONS = 200`.
3. **Read `top_mass_history`.** Raise `N_ACTIVE` until it sits near 1.
4. **Read the loss curve and the rank–mass curve.** A plateau with a cliff at rank
   `N_OUTPUT` → raise `N_OUTPUT` (and `N_ACTIVE` with it).
5. **Read `n_union_history`.** Still climbing → the support has not closed: raise
   `EXPLORE_DEEP_M` or `N_ITERATIONS`. Flat at the cap → raise `N_TRACKING` or accept it.
6. **Only now** touch `N_ITERATIONS` and `LR_MAX`, in that order.
7. **Re-run the same text with a different `POSITION_SEED`.** A fixed point that does not
   survive a different initial condition from the same text is a local artifact.
8. Scale to `N_TEXTS`.

### 6.8 Symptom → parameter

| Symptom | Most likely cause |
|---|---|
| `top_mass` < 0.9 and falling | `N_ACTIVE` too small |
| loss plateaus on a hard floor; rank–mass cliff at `N_OUTPUT` | `N_OUTPUT` too small |
| `n_union_history` still climbing at the last iteration | support not closed: `N_ITERATIONS`, `EXPLORE_DEEP_M` |
| `n_union_history` pinned at `N_TRACKING + EXPLORE_N` | `N_TRACKING` binding |
| loss noisy and non-monotonic early | `LR_MAX` too high |
| non-finite gradient warning | `2·eps` states reaching the query set: `EXPLORE_N` ↓, `N_TRACKING` ↓ |
| `far_mass` ≪ `1/(1−γ)` for some head | `MIN_POSITION` too early, or that head's `L_ctx` was clamped |
| saturation warning at run start | `L_ctx` clamped to `T_STAR` in cell 129 — the head has no measurable horizon |
| final π ≈ initial π, loss barely moved | the operator is barely state-dependent: check `L_ctx` (§6.1) before blaming the optimizer |
| every run converges to the same π regardless of text | either a genuine global attractor or `N_OUTPUT` is so small that only the operator's own top states survive — check the rank–mass cliff |
| OOM in `_forward_chunk` | `QUERY_CHUNK_SIZE` ↓ |
| OOM in `power_iteration_step` | `K_PRUNING` ↓ or `N_ACTIVE` ↓ |

### 6.9 Cost model

Per gradient iteration, the dominant terms:

```
exploration :  EXPLORE_M × [ forward(N_ACTIVE queries) + topk(N_ACTIVE × |V|) ]      no grad
gradient    :  M_ITERATIONS × forward(N_ACTIVE queries) × ~2                          checkpointed
                                                          ^ recomputed in backward
forward mem :  H × min(N_ACTIVE, QUERY_CHUNK_SIZE) × N_c × 4 bytes × O(1) intermediates
power it.mem:  N_ACTIVE × |V| × 4 bytes  (the dense transition rows before topk)
             + N_ACTIVE × K_PRUNING × K × 8 bytes  (the shifted keys, before unique)
union merge :  torch.unique over [(N_TRACKING + EXPLORE_N), K]
```

Wall-clock is `N_TEXTS × N_ITERATIONS ×` the above; the exploration is roughly
`(EXPLORE_SHALLOW_M + EXPLORE_DEEP_M / EXPLORE_EVERY)` forward passes per step on average,
so the deep cadence is cheap unless `EXPLORE_EVERY` is small.

---

# 7. Reduced mean-field fixed points: hunting saddles instead of repetitions

> **Cells 142–155.** A second fixed-point lane for `MultiTimescaleMeanFieldForwardPass`.
> It solves the *same* operator as §2.14 — `mtmf.forward` is called exactly as the
> benchmark calls it, byte for byte — but changes **what is tracked** and **how the
> fixed point is solved for**, so that *unstable* and *saddle* fixed points become
> reachable. Depends on cells 108 and 129 only (`mtmf_full`, `L_ctx_used`, `corpus`).

## 7.1 Why the K-gram lane can only find repetitions

This is a counting argument, and it is the reason cells 136–140 converge to repetition
loops no matter how the optimizer is tuned.

A fixed point with unigram entropy `H₁` and per-token conditional entropy
`h = H(next | window, π)` has a K-gram measure of entropy

```
H(π_K) ≈ H₁ + (K−1)·h          [nats]
```

and a measure of entropy `S` needs `≈ e^S` explicit states before `top_mass` approaches
1. Cell 137's budget is `N_ACTIVE = 512` (ln = 6.24 nats) and
`N_TRACKING = d_vocab/4 = 12065` (ln = 9.40 nats). Cell 145 measures `H₁` and `h`
directly. On `attn-only-1l` with a real Wikipedia prompt:

| state | `H₁` | `h` | `H(π_K)` at K=10 | states needed | best possible `top_mass` at `N_ACTIVE=512` |
|---|---|---|---|---|---|
| a real prompt's discounted far field | 4.73 | 4.53 | **45.5 nats** | 10^19.8 | **9e-18** |
| the repetition fixed point it falls into | 2.51 | 1.38 | 14.9 nats | 10^6.5 | 1.7e-4 |
| a pure `'.'` loop (text 1) | 0.03 | 0.03 | 0.3 nats | 10^0.1 | **1.0** |

Inverting the inequality `ln N_TRACKING ≥ H₁ + (K−1)h` gives the honest ceiling

```
K_max ≈ 1 + (ln N_TRACKING − H₁) / h
```

which on real prompts comes out at **K_max ≈ 1.85 – 2.10**, against the `K_WINDOW = 10`
cell 129 actually uses. **A sparse list of K-grams can represent a fixed point only if
that fixed point is nearly deterministic**, i.e. only if it is a repetition loop. The
optimizer is not failing; the state space excludes the answer.

## 7.2 The fix: put the state in Δ(V), keep the operator

`forward` reads the mean field only through, per head `h` and query `q`,

```
A_h(q) = Σ_c m(c)·exp(q_h·k_h(c)/s)                 (scalar)
B_h(q) = Σ_c m(c)·exp(q_h·k_h(c)/s)·v_h(c)          (d_head vector)
```

both **linear in `m`**. At frozen `m` the K-gram chain is an ordinary linear Markov
operator, so its stationary measure is a *slaved* variable. The only genuinely nonlinear,
self-consistent equation is on the unigram marginal:

```
m = G(m),    G(m) = Σ_q w_q(m) · P(· | q, m),    w_q(m) ∝ m(last token of q)
```

`ReducedMeanFieldOperator` implements exactly that. The window contents come from the
product ansatz `π(x_{t−K+1..t}) = ∏ m(x_i)`, realized as a **frozen active set**: every
`topk` and every sample happens in `refresh()` and nowhere else, so `G` is exactly smooth
in `m` between refreshes — which is what makes it Newton-solvable. Two variance controls
matter and are not optional: rows are allocated **proportional to `m`** (query weight has
an ESS of order 10, so uniform allocation wastes almost every draw), and the filler slots
are **stratified quantiles** of `m`, not i.i.d. draws (error `O(1/R)`, not `O(1/√R)`).
With `n_query=256, n_fill=8` the residual SAA noise measures **σ(|r|) ≈ 8e-4**, and it is
reported as the honest error bar on every solve.

## 7.3 What the reduction buys

| | cells 136–140 (K-gram) | cells 142–155 (reduced) |
|---|---|---|
| state | sparse `[N, K]` K-gram measure | dense `m ∈ Δ(V)` |
| `top_mass` on a real prompt | ≤ 1e-17 (§7.1) | **1.00** |
| smooth in the state? | no (top-k churn, STE) | yes, between refreshes |
| solver | natural-gradient descent on `JSD(π, πP)` | **JFNK** (Newton + GMRES on JVPs) |
| residual reached | plateaus ~1e-2 (scaled) | **8e-8**, quadratic |
| finds saddles? | no — descent on `‖r‖²` is flattest exactly along the near-critical directions | **yes** — Newton is stability-agnostic |
| cost of one `G` | one explore + one gradient pass | ~0.05 s |

## 7.4 The two-timescale reading (what "stationarity" means here)

The discounted mean field obeys an *exact* recursion,
`m_h(t+1) = m_h(t) + ε_h(δ_{x_{t+1}} − m_h(t))` with `ε_h = 1/L_ctx,h`. That is
constant-step-size **stochastic approximation**, whose ODE limit is

```
ṁ = G(m) − m
```

Consequences used throughout this section:

* **Fixed points of `G` are the equilibria of the real generation dynamics.**
* **The stability test is `Re λ(DG) < 1`, not `|λ(DG)| < 1`.** A mode at `λ = −3` is
  Picard-unstable but ODE-stable. `picard()` below is damped precisely so that it *is*
  the Euler discretization of the ODE and cannot be fooled this way.
* **The non-decaying "buzzing" is an O(√ε) fluctuation**, not a failure to converge:
  around `m*`, `m ≈ m* + √ε ξ` with `ξ` an Ornstein–Uhlenbeck process driven by
  `A = DG − I`. Variance is amplified in near-critical directions — critical slowing
  down is an observable signature of being near a saddle.
* **`1/L_ctx` is the temperature of topic switching.** Escape from a basin goes as
  `exp(−ΔΦ·L_ctx)`, so the barrier at the saddle is what sets how hard the model locks
  into a topic or a repetition loop.
* The assumption the whole picture rests on is `τ_mix ≪ L_ctx`. Cell 147 measures both
  and prints the ratio — the notebook computes each number elsewhere and never divides
  them.

`split_spectra` decomposes the Jacobian into the two physically distinct pieces:

```
D_u G_inner  -> how the FAST window law relaxes at frozen m   -> tau_mix
D_m G_inner  -> the MEAN-FIELD LOOP GAIN                      -> is there a problem at all?
DG_slaved = (I - D_uG)^-1 D_mG  -> the Jacobian of the true slow dynamics
```

**Read the loop gain first.** If it is ≈ 0 the operator is barely state-dependent, there
are no saddles to find, and §6.8's "final π ≈ initial π" symptom is the physics rather
than the optimizer.

## 7.5 Preconditions

1. Cells 3, 5, 7, 9, 10, 11, 13 and **108** have been run; **129** has been run so
   `mtmf_full`, `L_ctx_used`, `K_WINDOW`, `T_STAR` and `corpus` are live.
2. `torch.set_grad_enabled(True)` — the JVPs need forward-mode autograd. Model
   parameters stay `requires_grad = False`.
3. **`top_p = 1.0` and `temperature = 1.0`.** Nucleus filtering makes the operator
   non-smooth, biases the straight-through gradient, and manufactures artificial
   reducibility; the Jacobian would simply be wrong. `mtmf_full` is already built this
   way by cell 129.
4. A fixed point is a property of the operator, so cell 131's `full` row must
   comfortably beat `jsd_baseline_context_unigram` before any of this means anything.

## 7.6 Known limitation of the Jacobian

`_forward_chunk` deliberately detaches the `context_vals` that feed `sigma_pos`
(§2.11.3). The JVP therefore reproduces the *σ-frozen* Jacobian, not the full one.
Measured against central finite differences on a direction inside `supp(m)`, the gap is
**≈ 5 %, and it is eps-independent** (4.92e-2, 4.87e-2 at eps = 1e-4, 1e-5), which
confirms it is that systematic term and not non-smoothness. It is harmless for Newton
(inexact Newton still converges — the runs below reach 8e-8) but it puts a ~5 % error bar
on every eigenvalue quoted here. Eigenvalues near the stability boundary `Re λ = 1`
should be treated as "marginal", not as decided.

## 7.7 What these cells found on `attn-only-1l` (K=10, t*=800, all 8 heads)

Measured with `N_QUERY=256, N_FILL=8, CTX_TOP_N=2048`, `top_p = temperature = 1`, on the
Wikipedia corpus of cell 110. Runtimes on one RTX 5000: cell 146 ≈ 40 s, cell 147 ≈ 200 s,
cell 148 ≈ 20 min, cell 149 ≈ 6 min, cell 150 ≈ 8 min, cell 151 ≈ 12 min, cell 152 ≈ 25 min.

### (a) The representability audit — why the K-gram lane finds only repetitions (cell 145)

| state | `H₁` | `h` | `H(π_K)` @ K=10 | states needed | best `top_mass` @ `N_ACTIVE`=512 | `K_max` |
|---|---|---|---|---|---|---|
| text 0, real prompt | 4.729 | 4.531 | **45.5** | 10^19.8 | **8.8e-18** | **2.03** |
| text 2, real prompt | 4.597 | 4.946 | 49.1 | 10^21.3 | 2.4e-19 | 1.97 |
| text 4, real prompt | 4.804 | 5.425 | 53.6 | 10^23.3 | 2.6e-21 | 1.85 |
| text 0, fixed point | 2.507 | 1.379 | 14.9 | 10^6.5 | 1.7e-4 | 6.00 |
| text 1, fixed point (`'.'` loop) | 0.032 | 0.031 | 0.3 | 10^0.1 | **1.00** | 300 |

Read the last column against `K_WINDOW = 10`. **On a real prompt the honest ceiling is
`K_max ≈ 1.85 – 2.03`.** The only row a sparse K-gram list can carry is the pure
repetition loop, which is exactly what cells 136–140 return.

### (b) The solve (cell 146)

```
SAA (filler) noise sigma(|r|) at m_init     9.1e-4
JVP vs central finite differences           5.7e-2 .. 6.9e-2, eps-INDEPENDENT (the sigma_pos detach)
damped Picard   |r|  7.7e-2 -> 1.2e-4       150 steps
JFNK            |r|  5.0e-3 -> 9.2e-8       8 steps, quadratic
```

JFNK reaches four orders below Picard's plateau and below the SAA bar, so the surrogate
is solved essentially exactly and the residual error is the *operator's* sampling error,
not the solver's.

### (c) Attractors — one operator, many basins (cell 149)

`t*` and `far_mass` do not depend on the prompt, so every text is a different initial
condition of the **same** map:

| text | fixed point | `H₁` | ESS |
|---|---|---|---|
| 0 | `'ists' .29 / ' anarch' .28 / ',' .19` | 2.50 | 12.2 |
| 1 | `'.' .995` | 0.03 | 1.0 |
| 2 | `'ic' .48 / ' Ital' .48` | 1.00 | 2.7 |
| 3 | `' Al' 1.000` | 0.004 | 1.0 |
| 4 | `'us' .48 / ' Ze' .48` | 1.01 | 2.7 |
| 5 | `' Lincoln' .90` | 0.74 | 2.1 |

The two-token ones are **period-2 cycles** (`' Ital'→'ic'→' Ital'`); a 2-cycle appears in
the mean field as two tokens at ≈½ each, `H₁ ≈ ln 2 = 0.69`.

### (d) An UNSTABLE fixed point (cells 150–151)

Between attractor A = `' Lincoln'` (text 5) and B = `'.'` (text 1), `μ(c)` crosses zero
three times: at both endpoints (`dμ/dc < 0`, the attractors) and **once in the interior
with `dμ/dc = +0.51`**. Bisecting that crossing and polishing with JFNK:

```
state          ' Lincoln' 0.784 | '.' 0.175 | ',' 0.007 | ' and' 0.006
residual       |r| = 6.2e-8          (JFNK: 1.8e-2 -> 6.2e-8, and it STAYED on the saddle,
                                      JSD(start,end) = 8.5e-4)
H = 0.8095     ESS = 2.25            HIGHER entropy than either neighbour (0.736 and 0.033)
MORSE INDEX    1                     exactly one unstable direction -> an index-1 saddle
max Re lambda  +1.40 .. +1.71        > 1  => UNSTABLE
loop gain      1.44 .. 1.75          > 1  => locally expanding, which is REQUIRED for two
                                      attractors to coexist (a globally contracting map
                                      has a unique fixed point)
|<eigenvector, phi>|  = 0.9999       the unstable direction IS the A-B tilt
eigenvector    '.' -0.717 , ' Lincoln' +0.697 , everything else < 0.004
untied growth  Re mu = +4.0e-3 .. +9.0e-3 / token  ->  escape time ~ 110-250 tokens
```

The **untied** multi-timescale system (cell 153) agrees: `max Re μ = +9.0e-3` per token
at the saddle against `−5.0e-3` at the attractor, i.e. unstable and stable respectively.
Every complex pair found was **damped** — no Hopf bifurcation on this model, so the
heterogeneous horizons (`L_ctx` 18.9 → 108.3) do not by themselves produce oscillatory
topic drift here. The apparatus to detect one is in place if another model does.

This is the **mixture saddle**: a tilted blend of the two basins, higher entropy than
both, with the tilt as the single unstable coordinate. A second one was found between
text 0 and text 1 (`Morse = 1`, `max Re λ = +1.010`, loop gain `0.996`) — that one is
*marginal*, inside the ~5 % Jacobian error bar, and should be reported as such.

**Robustness to the sampling budget** (cell 152 — the check that matters, since the
product ansatz is sampled):

| `n_query` | `n_fill` | rows | `\|r\|` | `H` | ESS | loop gain | max Re λ | Morse | `⟨v,φ⟩` |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 8 | 2 250 | 8.5e-8 | 0.7750 | 2.171 | 1.444 | +1.405 | 1 | 0.9999 |
| 256 | 32 | 8 324 | 4.1e-7 | 0.7852 | 2.193 | 1.472 | +1.432 | 1 | 0.9999 |
| 512 | 16 | 8 568 | 1.5e-7 | 0.7856 | 2.194 | 1.470 | +1.430 | 1 | 0.9999 |
| 256 | 64 | 16 461 | 3.0e-8 | 0.7879 | 2.199 | 1.471 | +1.429 | 1 | 0.9999 |
| 512 | 64 | 32 981 | 1.4e-7 | 0.7913 | 2.206 | 1.480 | +1.439 | 1 | 0.9999 |

Over a **15× range of sampling budget** the entropy moves 2 % (0.775 → 0.791) and the
unstable eigenvalue 2.4 % (1.405 → 1.439), both **converging** rather than drifting; the
Morse index and the eigenvector are identical throughout. **The saddle is not a sampling
artifact.** Best estimate `max Re λ = 1.44 ± 0.02`; a single active-set refresh in the
cell-151 run returned 1.71, so quote the converged sweep, not one refresh, and in either
case the qualitative claim (`> 1`, index 1) is untouched.

### (e) Time-scale separation — it holds, but NOT uniformly (cell 147)

`τ_mix` is measured from `|λ(D_u G_inner)|`, the relaxation of the fast window law:

| state | `τ_mix` (tokens) | worst `τ_mix / L_ctx` | loop gain | verdict |
|---|---|---|---|---|
| `'.'` attractor (ESS 1.0) | 0.23 | 0.012 | 0.165 | separation holds by 2 orders |
| `' Lincoln'/'.'` saddle | 0.45 | 0.024 | 1.75 | holds |
| anarchism attractor (ESS 12.2) | **4.20** | **0.222** | 0.625 | **marginal** |
| text-0/`'.'` saddle (marginal) | **6.07** | **0.32** | 0.996 | **marginal** |

**This is the honest answer to "does the time-separation assumption survive?".** It is
excellent at the degenerate repetition fixed points and *degrades to marginal exactly at
the richer states and near the marginally-stable saddle* — i.e. precisely in the regime
worth studying. That is critical slowing down: as a mode approaches `Re λ = 1`, the frozen
chain's own relaxation lengthens, and the two timescales stop separating. Any claim about
a near-critical fixed point has to carry this ratio next to it. **The notebook computes
`implied_timescales_tokens` and `L_ctx` in different lanes and never divides them; cell
147 does.**

The same cell reports the fluctuation scale: at the anarchism attractor the leading slow
mode is `λ = +0.583` (`μ = −0.417`), giving relaxation times of 45–164 tokens per head and
a stationary fluctuation of `√(1/L_ctx) = 0.12–0.23`. **That is the "buzzing":
constant-step-size stochastic approximation does not converge to `m*`, it fluctuates
around it at `O(√(1/L_ctx))`, forever. It is a prediction, not a convergence failure.**

### (f) Continuation in the mean-field gain (cell 148)

`λ` multiplies `far_mass`. At `λ = 0` the operator is state-independent and its unique
fixed point is broad and disordered. Walking `λ` up (18 values, Newton-continued, spectra
at each):

| `λ` | `H` | ESS | `τ_mix` | loop gain | max Re λ |
|---|---|---|---|---|---|
| 0.00 | 4.177 | 65.2 | 0.93 | 0.000 | +0.000 |
| 0.02 | 3.813 | 45.3 | 0.83 | 0.276 | +0.279 |
| 0.05 | 3.014 | 20.4 | 0.62 | 0.346 | +0.372 |
| 0.10 | 2.187 | 8.9 | 0.55 | 0.303 | +0.355 |
| **0.16** | 1.765 | 5.8 | 0.62 | **0.533** | **+0.551** |
| 0.20 | 0.902 | 2.5 | 0.62 | 0.521 | +0.512 |
| 0.25 | 0.395 | 1.5 | 0.33 | 0.112 | +0.112 |
| 0.50 | 0.162 | 1.2 | 0.25 | 0.045 | +0.043 |
| 1.00 | 0.019 | 1.0 | 0.13 | 0.070 | +0.070 |
| 1.20 | 0.019 | 1.0 | 0.12 | 0.073 | +0.073 |

Three things to read off. The **order parameter collapses monotonically** (`H`: 4.18 → 0.02)
— by `λ = 1` this branch has become a single-token repetition. The **loop gain peaks at
≈ 0.53 near `λ = 0.16`** and then *falls back* to ≈ 0.07: the feedback **self-limits**,
because as `m` concentrates the far field stops carrying information. And **nothing ever
crosses `Re λ = 1`, and `λ(s)` never turns around**, out to `λ = 1.2`.

So on *this* branch the ordering is a fast **crossover, not a bifurcation**, and the real
operator at `λ = 1` sits deep in the ordered, strongly-stable regime. The other attractors
of §7.7(c) are on branches **not reachable from `λ = 0` along this path** — which is why
the pairwise saddle hunt of cells 150–151, and not continuation, is what produced the
unstable fixed points here. (`arclength_continuation` is provided for the case where a
branch *does* fold; it was not needed on this model.)

### (g) A negative result worth keeping

Between the `' Ital'/'ic'` and `' Ze'/'us'` attractors the straight-line coordinate
`φ = m_A − m_B` produced **zero interior sign changes of `μ(c)`**. The two period-2 cycles
are not connected by a saddle along that particular chord. The method finds what lies on
the chord you choose; a negative result means "not on this line", not "no saddle".

## 7.8 Caveats, in the order they will bite

1. **The product ansatz is an extra approximation.** `π(window) = ∏ m(x_i)` ignores
   correlations inside the fast window. It is the same *class* of approximation the far
   field already makes, but it is not free: at a true fixed point the window tokens are
   not independent. `n_fill` controls the sampling error of the ansatz, never the ansatz
   itself — §7.7(d) prices the former, nothing here prices the latter. An order-1 (Markov)
   ansatz, `π(x₁..x_K) = ν(x₁)∏Q(x_{i+1}|x_i)` with low-rank `Q`, is the natural next step.
2. **The Jacobian is the σ-frozen one** (§7.6), ~5 % off, and the spread across active-set
   refreshes is wider still (~20 % on the saddle's eigenvalue). Do not call an eigenvalue
   within that of `Re λ = 1` decided — the text-0 saddle at `+1.010` is exactly such a case.
3. **The active set is frozen during a solve.** That is what makes `G` smooth, but the
   converged state is a fixed point of a *surrogate*. Always refresh and re-measure — cell
   146 does. The SAA bar is **state-dependent and much larger at concentrated states**
   (9e-4 at `m_init` with ESS 113, but 1.5e-2 at the saddle with ESS 2.2), because a
   near-atomic product measure makes whole windows flip between stratification cells.
4. **`top_p` must stay 1.0.** Nucleus filtering breaks smoothness and the straight-through
   estimator makes the Jacobian wrong outright.
5. **`n_layers == 1`.** `MultiTimescaleMeanFieldForwardPass` enforces it, so every claim
   here is a one-layer claim. A unigram mean field is not a sufficient statistic deeper.
6. **A fixed point of a bad operator is meaningless.** Cell 131's `full` row must clear
   `jsd_baseline_context_unigram` first.

## 7.9 What to do next

1. **Report `τ_mix / L_ctx` beside every fixed point** — it is a one-line addition and it
   is the number that says whether the two-timescale picture applies at that state.
2. **Drop `K_WINDOW` to 2–3 if you keep the K-gram lane**, or keep `K = 10` in `forward`
   and stop tracking the joint K-gram measure (this section).
3. **Order-1 window ansatz** to price caveat 1.
4. **Seed the saddle finder from real trajectories**: cell 81's QSD affinity matrix already
   flags topic transitions, and a trajectory crossing a basin boundary passes near a
   saddle. That gives initial guesses no chord has to be chosen for.
5. **The barrier is the physics.** With `1/L_ctx` as the temperature and the saddle as the
   transition state, `exp(−ΔΦ·L_ctx)` predicts how fast generation falls into a repetition
   basin. The untied escape rate measured here (`+4.0e-3` per token, ≈ 250 tokens) is the
   first number of that story.

---

# 8. Replication record — the fixed points that were actually found

Everything needed to reproduce §7.7 from a cold kernel: the exact configuration, the
exact procedure, the states themselves, and what they appear to mean about the model.
The measures are archived at
`results/meanfield_fixed_points/meanfield_fixed_points.pt` (top-400 sparse form, plus
the unstable eigenvectors), alongside a standalone copy of the cell-143 module as
`results/meanfield_fixed_points/meanfield_reference.py`.

## 8.1 Exact configuration

```python
# --- model ------------------------------------------------------------------------
model = HookedTransformer.from_pretrained("attn-only-1l", device="cuda:0")
#   = NeelNanda/Attn_Only_1L512W_C4_Code, TransformerLens DEFAULTS
#     (fold_ln=True, center_writing_weights=True, center_unembed=True).
#     n_layers 1, n_heads 8, d_head 64, d_model 512, d_vocab 48262, n_ctx 1024.
#   NOTE: doc 4.2 requires fold_ln=False for the K-GRAM lane. This lane does not care --
#   MultiTimescaleMeanFieldForwardPass applies ln1.w AND ln1.b, so either load works.
model.cfg.use_attn_result = False          # not needed here, and it costs n_heads x memory
for p in model.parameters(): p.requires_grad = False
torch.set_grad_enabled(True)               # the JVPs need forward-mode autograd

# --- corpus -----------------------------------------------------------------------
it = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
ds = Dataset.from_list(list(it.take(400)))
corpus = [t for t in ds["text"] if len(t) > 5000][:40]
#   corpus[0] = "Anarchism", corpus[1] = "Autism", corpus[2] = "Albedo"(Italian-heavy),
#   corpus[3] = "A", corpus[4] = "Achilles"(Zeus), corpus[5] = "Abraham Lincoln"

# --- timescale profiling (cell 129, with N_PROBE reduced from 1024 to 256) ---------
K_WINDOW, T_STAR      = 10, 800
PROFILE_D_MAX, FIT_D_MAX = 512, 256
N_PROBE, PROBE_SEED   = 256, 0
# fitted, then clamped into [K+2, T_STAR] -- NO head was clamped:
L_ctx_used = [68.2, 26.9, 18.9, 75.2, 108.3, 26.3, 66.6, 26.1]     # heads 0..7
gamma      = [0.98533, 0.96285, 0.94716, 0.98670, 0.99076, 0.96200, 0.98498, 0.96168]
mtmf_full  = MultiTimescaleMeanFieldForwardPass(
    model, L_ctx=L_ctx_used, active_heads=None, K=K_WINDOW,
    query_position_offset=T_STAR, temperature=1.0, top_p=1.0)

# --- the reduction (cell 144) ------------------------------------------------------
rmf = ReducedMeanFieldOperator(mtmf_full, n_query=256, n_fill=8,
                               ctx_top_n=2048, query_chunk=512, far_scale=1.0, seed=0)
#   -> 1995 query rows; coverage (query, context) = (0.9952, 1.0000)

# --- initial conditions (cell 144/149) ---------------------------------------------
POSITION  = 900
estimator = DiscountedUnigramContextEstimator(window=10)
init_head = argmax(L_ctx) = 4          # gamma 0.99076, L_ctx 108.3
#   realized far_mass matched the saturated 1/(1-gamma) to 0.1 on every head, so
#   POSITION = 900 is comfortably inside the autonomous regime (doc 6.2, MIN_POSITION).
```

`N_PROBE = 256` instead of cell 129's 1024 is the only deviation from the notebook's own
profiling defaults; γ is a median over probe tokens and is insensitive to it (doc 6.1).

## 8.2 The procedure

| step | cell | call | settings |
|---|---|---|---|
| 1. attractors | 149 | `picard` then `newton_krylov` | `n_steps=150, alpha=0.15, refresh_every=25`; then `max_newton=8, tol=1e-10, refresh_every=0` |
| 2. reaction coordinate | 150 | `mu_scan` | `φ = (m_A − m_B)/‖·‖`, 28 values of `c` from `c_B` to `c_A`, `max_newton=14, gmres_maxiter=40, tol=1e-9`, warm-started |
| 3. bracket | 150 | sign changes of `μ(c)` | keep the one with `dμ/dc > 0` |
| 4. bisect | 151 | `solve_constrained` | 14 bisections; **stalls at `|μ| ≈ 2e-2`** (the constrained branch folds in `c`) |
| 5. **polish** | 151 | `newton_krylov` | `max_newton=16, tol=1e-9, refresh_every=0, gmres_maxiter=60, gmres_tol=1e-4` → `|r| ≈ 4e-9` |
| 6. classify | 151 | `split_spectra` | `k=8`, `n_neumann=8` |
| 7. validate | 152 | re-solve at `n_fill ∈ {8, 32}`, `n_query ∈ {256, 512}` | Morse index and eigenvector must not move |
| 8. untied | 153 | `tied_reduced` / `untied_reduced` | dense projection on a ~26-vector basis |

Step 5 is not optional: bisection alone leaves `|r| ≈ 2e-2`, two orders above the noise
bar, and the spectra computed there are wrong by ~20 %. Step 4 → 5 is also the cleanest
demonstration of the section's central claim: **JFNK takes `|r|` from 1.6e-2 to 4.2e-9 in
16 steps and moves the state by `JSD = 8.1e-4`, i.e. it converges to an index-1 saddle
and stays on it.**

Runtimes on one Quadro RTX 5000 (16 GB): cell 146 ≈ 50 s, 147 ≈ 200 s, 148 ≈ 20 min,
149 ≈ 5–6 min, 150 ≈ 8 min, 151 ≈ 8 min, 152 ≈ 25 min. Peak VRAM ≈ 2 GB.

## 8.3 The stable fixed points (six initial conditions, one operator)

All reached `|r| ≤ 1e-7`. `H` in nats, ESS `= e^H`.

| text | article | fixed point (top of `m*`) | `H` | ESS | `h` | `K_max` |
|---|---|---|---|---|---|---|
| 0 | Anarchism | `'ists' .289  ' anarch' .277  ',' .186  ' and' .056  '.' .039` | 2.499 | 12.2 | 1.379 | 6.0 |
| 1 | Autism | `'.' .995  '._' .005` | 0.033 | 1.03 | 0.031 | 300 |
| 2 | Albedo | `'ic' .484  ' Ital' .480` | 0.996 | 2.71 | 0.298 | 29 |
| 3 | A | `' Al' 1.000` | 0.004 | 1.00 | 0.004 | 2124 |
| 4 | Achilles | `'us' .482  ' Ze' .477` | 1.006 | 2.73 | 0.308 | 28 |
| 5 | Abraham Lincoln | `' Lincoln' .899  '.' .027` | 0.736 | 2.09 | 0.693 | 13.5 |

Every initial condition was a real prompt with `H ≈ 4.6–4.8`, `h ≈ 4.3–5.4`. **The
dynamics loses 2–5 nats of entropy on the way to its fixed point** — `JSD(π₀, m*)` runs
0.42–0.65.

Three kinds of attractor appear:

* **fixed-point loops** (texts 1, 3): one token at mass ≈ 1. `' Al'` and `'.'`.
* **period-2 cycles** (texts 2, 4): `' Ital'→'ic'→' Ital'` and `' Ze'→'us'→' Ze'`. A
  2-cycle shows up in a *mean field* as two tokens at ≈ ½ each, `H ≈ ln 2 = 0.693`. The
  mean field cannot distinguish a 2-cycle from a genuine 50/50 mixture — which is a real
  limitation of a unigram state, and one the K-gram lane would not have.
* **a small recurrent set** (text 0): five tokens forming the phrase fragment
  `' anarch' 'ists' ',' ' and'` — a stuttering list construction, `H = 2.50`.

## 8.4 The unstable fixed points

### Saddle A — `' Lincoln'` ↔ `'.'` (the one to trust)

Found between attractor 5 and attractor 1. Converged by bisect + JFNK to `|r| = 4.2e-9`,
and reproduced independently on a second GPU with a fresh active set (`H = 0.8092` vs
`0.8095`).

```
m*        ' Lincoln' 0.784 | '.' 0.174 | ',' 0.007 | ' and' 0.006 | ' 1' 0.003
H = 0.809 nats, ESS = 2.25            ABOVE both neighbours (0.735 and 0.033)
JSD to A = 0.037,  to B = 0.446       it sits close to the shallow basin, far from the deep one

MORSE INDEX      1                    exactly one unstable direction
max Re lambda    +1.44 +- 0.02        converged over a 15x sampling-budget sweep
                                      (one refresh in cell 151 gave 1.71 -- quote the sweep)
loop gain        1.44 .. 1.48         > 1, as coexistence of two attractors requires
lambda(slaved)   1.639, 0.114, 0.077, 0.046+-0.012j, 0.042      -- ONE mode above 1, rest tiny
tau_mix          0.40 - 0.48 tokens   worst tau_mix/L_ctx = 0.021 -> separation holds
untied growth    Re mu = +4.0e-3 .. +9.0e-3 / token -> escape time ~ 110-250 tokens
                 (untied attractor for contrast: Re mu = -5.0e-3, stable)
                 all complex pairs DAMPED -> no Hopf, no limit cycle on this model

unstable eigenvector (|<v, phi>| = 0.9996 -- it IS the A-B tilt)
    '.'         +0.716
    ' Lincoln'  -0.698
    ','         +0.005      everything else below 0.004
```

Robustness (cell 152): over a **15× range of sampling budget** (2 250 → 32 981 query
rows) `H` moves 2 % and `max Re λ` 2.4 %, both converging; Morse index stays 1 and
`⟨v, φ⟩` stays 0.9999 throughout.

### Saddle B — the anarchism basin ↔ `'.'` (marginal, report as such)

Found between attractor 0 and attractor 1, at `|r| = 6.8e-3` (bisection only — it was
**not** put through the JFNK polish, so it is a *near*-saddle and its eigenvalues carry
the 20 % bar of §7.8.2).

```
m*        'ists' .301 | ' anarch' .289 | ',' .121 | '.' .119 | ' and' .035
H = 2.438 nats, ESS = 11.4            BELOW attractor 0 (2.499), far above attractor 1 (0.033)
MORSE INDEX      1
max Re lambda    +1.010                MARGINAL -- inside the ~5% Jacobian error bar
loop gain        0.996                 also marginal
tau_mix          6.07 tokens           worst tau_mix/L_ctx = 0.32  -> SEPARATION IS MARGINAL
untied growth    Re mu = +1.7e-4 per token -> escape time ~ 6000 tokens

unstable eigenvector
    '.'         -0.808
    ','         +0.572
    ' and'      +0.121
    ' anarch'   -0.049
    'ists'      -0.047
```

### A negative result

Between the two period-2 cycles (`' Ital'/'ic'` and `' Ze'/'us'`) the chord
`φ = m_A − m_B` gave **zero interior sign changes of `μ(c)`**. The method finds what lies
on the chord you pick; this says "no saddle on this line", not "no saddle".

## 8.5 Interpretation — what these states say about the model

**1. There is a universal sink, and it is `'.'`.** Attractor 1 (`'.'` at 0.995) is reached
from an unrelated article, and `'.'` is the dominant component of the unstable
eigenvector of *both* saddles (+0.716 in A, −0.808 in B). On this model the terminal
punctuation token is the gateway to a global absorbing basin: once the far field is
dominated by `'.'`, the mean field makes `'.'` the most likely continuation, and the loop
closes. Every saddle found is the boundary between *staying in a topic* and *falling into
the punctuation sink*. That is a mechanistic statement of neural text degeneration inside
this abstraction.

**2. The saddles are mixture states, and the unstable coordinate is the tilt.** Saddle A
is literally `0.78·(Lincoln basin) + 0.17·(period basin)`, its entropy is higher than
either neighbour, and its unstable eigenvector aligns with `m_A − m_B` to four decimals.
This is the Curie–Weiss picture: the transition state is the blend, and the order
parameter is how far you have tipped. It also means such a saddle is exactly the kind of
object the K-gram parameterization is *least* able to hold (cell 145: higher entropy
needs exponentially more explicit states).

**3. Saddle B is a punctuation decision.** Its unstable direction is `','` (+0.572)
against `'.'` (−0.808), with the content tokens `' anarch'/'ists'` barely participating.
Read as dynamics: inside the anarchism basin the model is balanced between *continuing
the clause* (comma, stay in the list construction) and *ending the sentence* (period, and
from there the sink). The saddle is a **syntactic branch point**, not a semantic one, and
it is marginally unstable — `max Re λ = 1.010`, escape time ≈ 6000 tokens. A near-neutral
direction like that is precisely what "the model dithers between continuing and stopping"
would look like as a dynamical statement.

**4. The barrier is asymmetric, and that predicts which way generation falls.** Saddle A
sits at `JSD = 0.037` from the `' Lincoln'` attractor but `0.446` from the `'.'`
attractor. The `' Lincoln'` basin is *shallow* — the transition state is almost on top of
it — while the `'.'` basin is deep and wide. With `1/L_ctx` as the temperature and escape
going as `exp(−ΔΦ·L_ctx)`, that asymmetry says: a repeated proper noun is a metastable
state that leaks quickly into terminal punctuation, and the punctuation basin does not
leak back. The measured untied escape rate, `+4.0e-3 … +9.0e-3` per token (≈ 110–250 tokens), is
the first quantitative version of that claim.

**5. Time-scale separation is good where nothing interesting happens and marginal where
it does.** `τ_mix/L_ctx` is 0.012 at the `'.'` attractor, 0.021 at saddle A, 0.22 at the
anarchism attractor and 0.32 at saddle B. The separation degrades exactly as the state
gets richer and as `Re λ` approaches 1 — critical slowing down. So the two-timescale
picture is *quantitatively* sound at the degenerate fixed points and only *qualitatively*
sound at the interesting ones. Any claim about a near-critical state has to carry that
ratio next to it.

**6. The single feedback loop is weak except near a saddle.** `loop_gain` is 0.17 at the
`'.'` attractor, 0.63 at the anarchism attractor, and 1.4–1.75 at saddle A. It has to
exceed 1 somewhere — a globally contracting map has a unique fixed point — and the only
places it does are the basin boundaries. This is why continuation from `λ = 0` (cell 148)
never finds a bifurcation: along that branch the gain peaks at ≈ 0.5–0.66 and then
*self-limits* as the state concentrates. The multiple basins are not created by a
bifurcation of the disordered branch; they are separate branches, and pairwise saddle
hunting rather than continuation is what reaches them.

**7. Caveat that colours all of the above.** This is a 1-layer, attention-only model, and
a unigram mean field with an i.i.d.-window ansatz. The `'.'` sink and the comma/period
branch point are real properties *of this abstraction*; whether they survive into the
concrete model is what cell 131's ablation table is for, and whether they survive a
richer window ansatz is open (§7.8.1).
