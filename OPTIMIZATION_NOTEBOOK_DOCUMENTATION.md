# `feedback_circuits_transformerlens_optimization_only.ipynb` — Developer Guide

> A `CLAUDE.md`-style reference for **one specific research notebook**:
> [`src/scripts/feedback_circuits_transformerlens_optimization_only.ipynb`](src/scripts/feedback_circuits_transformerlens_optimization_only.ipynb)
>
> 108 cells, ~350 KB of source. This file is the map you should read before editing or running any of it.

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
[88–98]  Wide semantic head optimization — one head per GPU over a real corpus, + result inspection
[99–103] APPROXIMATION QUALITY BENCHMARK — how close is the abstraction to the real model?
  ├─ [100] DEFINITIONS: estimators, approximate passes, the benchmark harness
  ├─ [102] Example run: partition-function abstraction on a Wikipedia corpus
  └─ [103] Result inspection: error histograms, error-vs-position, worst-case dump
[104–107] EXACT 1-LAYER / SINGLE-HEAD / POSITION-FREE PASS — the abstraction's zero point
  ├─ [105] `ExactSemanticHeadForwardPass`
  └─ [107] Example run + ablation table (measured JSD ≈ 1.6e-12)
```

Note that cells 88–98 (wide optimization) are only sketched here; cells 99–103 are
documented in full in §2.9 and §5, and cells 104–107 in §2.10.

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

### 1.7 Dependency tree — Pipeline D (approximation quality benchmark)

```
ApproximationQualityBenchmark(model, approximate_pass, ...)   [cell 100]  ← ENTRY POINT
└── run(corpus: List[str])
    ├── tokenize(text)                → [n_tokens]  (BOS-prepended, chopped to max_tokens)
    ├── sample_positions(n_tokens)    → List[int]   (seeded random.Random)
    └── evaluate_position(text, tokens, position)
        ├── real_probs(tokens, position)                         ← GROUND TRUTH
        │   └── model(tokens[:position+1])[0, -1]
        │       └── apply_sampling_transform(logits, T, top_p)
        ├── approximate_pass.predict(tokens, position)           ← UNDER TEST
        │   ├── PartitionFunctionApproximateForwardPass          [cell 100]
        │   │   ├── context_estimator.estimate(tokens[:position+1], model, S_init)
        │   │   │   ├── UnigramContextEstimator  → pi_from_context    [cell 9]
        │   │   │   └── KGramContextEstimator    → get_kgram_distribution_from_tokens  [cell 9]
        │   │   ├── _build_query_keys(tokens, position)  → [1, S_init]
        │   │   └── abstract_forward_pass(...)                   [cell 10]
        │   └── ExactSemanticHeadForwardPass                     [cell 105]   ← §2.10
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

summarize_benchmark(results)                                   [cell 100]  ← pretty printer
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

### 2.9 The approximation quality benchmark (cells 99–103)

Everything else in this notebook *assumes* the abstraction (`abstract_forward_pass`) is a
faithful stand-in for the concrete model. Cells 99–103 are the experiment that **measures
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
makes A/B comparisons meaningful; the commented-out sweep at the bottom of cell 103 relies
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
profiler output should be re-checked.** The fix is one line, and cell 102 shows it:

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

### 2.10 The *exact* 1-layer forward pass (cells 104–107)

`ExactSemanticHeadForwardPass` (cell 105) is not another approximation — it is the
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
  This is not checkable from inside the class; cell 107 wires it correctly, copy from there.

#### 2.10.6 How it patches global state

`abstract_forward_pass` reads `model.W_pos` and `model.W_E` directly, not through hooks, so
the class patches `.data` under two context managers (`_zero_pos_embeddings`,
`_rescaled_embeddings`) with `try/finally` restores. Consequences:

- `approx_hooks` must stay **empty** — the abstraction patches the weights itself.
- It is not thread-safe and must not run concurrently with anything else touching the model
  (relevant if you ever fold it into the multi-GPU lane of cells 88–92).
- `_W_E_buffer` is a persistent `[d_vocab, d_model]` scratch tensor (~100 MB at
  48k × 512 fp32) held for the lifetime of the object, alongside `W_E_unit` of the same
  size. Budget ~200 MB of VRAM per instance; cell 107 builds four of them for the ablation
  table, so delete them when done.

#### 2.10.7 Reference numbers

`attn-only-1l`, head 3, 9 positions ≥ 300 on a small synthetic corpus of repeated
sentences. Cell 107 as written runs on cell 102's Wikipedia `corpus`, so the exact digits
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
→ Cells 99–103. Build a `PartitionFunctionApproximateForwardPass` around your
`forward_pass_kwargs` and run `ApproximationQualityBenchmark` on a corpus. Check the mean
JSD **against the `jsd_baseline_context_unigram` control** — an abstraction that does not
beat the raw context histogram is not using the model. Do this *before* trusting a
fixed point, a QSD, or a PCCA+ macrostate, because every one of them is computed through
`abstract_forward_pass`.

**"My JSD is bad — is `single_layer_forward` wrong, or is my parameterization wrong?"**
→ Cells 104–107 (§2.10). `ExactSemanticHeadForwardPass` drives the *same*
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
Cell **100** is a fourth definitions-only cell (the approximation benchmark) and depends on
cells 9 and 10; it can be run at any point after them. Cell **105** is a fifth
(`ExactSemanticHeadForwardPass`, §2.10) and depends on 10 and 100. Cell 107 additionally
needs `zero_head_hook` / `remove_pos_embed_hook` from cell 7 and a `corpus` — cell 102's
will do.
Cell 15 loads the model. Everything after that assumes `model`, `device`, and the
definitions above are live. Beyond that the notebook is **not** linearly runnable —
sections 50–58, 59–60, 61–66, 67–70, 71–82, 88–98, 99–103 are alternative experiments
that each redefine overlapping globals (`losses`, `pi_vals`, `pi_keys`, `heads`, `K_list`,
`temperature`, `p`/`top_p`, `chunk_size`, `forward_pass_kwargs`). Pick one lane and run it
top to bottom. The benchmark lane is 100 (definitions) → 102 (run) → 103 (inspect); cell
102 rebuilds `forward_pass_kwargs` from scratch, so it will overwrite whatever cell 72 or
cell 80 left behind.

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

**⚠ `fold_ln=False` is mandatory — except for cells 104–107.** `extract_frozen_sigma`, the
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
`PartitionFunctionApproximateForwardPass` (cell 100) validates this and refuses to
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

---

## 5. Extending the Approximation Benchmark

The benchmark has exactly **two** extension points, and they are independent. Adding a new
hypothesis means writing one subclass; nothing else in cells 99–103 changes, and the
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
(cell 105, §2.10) is the reference for this pattern: nothing says `forward_pass_kwargs` has
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

The commented-out sweep at the bottom of cell 103 is the template: rebind
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
