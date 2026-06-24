# mlxrl Design

`mlxrl` is a small, single-process RL post-training library for Apple Silicon.
The design goal is narrow: make on-policy, critic-free RL for local MLX LLMs
fast enough that rollout is the main problem and the loss stays thin.

## Critic-Free By Design

The library is intentionally limited to rollout-based policy-gradient methods:
GRPO, Dr. GRPO, DAPO, GSPO, and RLOO. PPO is out of scope because it needs a
critic/value model path, a different forward pass, and a different memory
profile. DPO and ORPO are also out of scope because they are offline preference
objectives rather than on-policy rollout algorithms.

That boundary matters. `mlxrl` keeps one policy model object in memory, attaches
QLoRA adapters to it, and computes reference logprobs by disabling those
adapters for a second pass. There is no inference server and no second reference
model copy.

## Algorithm Protocol

The engine does not know which algorithm it is training. Concrete algorithms
implement the `Algorithm` protocol:

- compute per-completion advantages;
- optionally filter a prepared batch;
- compute loss and diagnostics from policy, old-policy, and reference logprobs.

`rollout/`, `policy/`, and `train/` must not import from `algo/`. The import
direction test enforces this. The payoff is that rollout and logprob code stay
stable while algorithms change. DAPO's dynamic sampling is the proof that the
interface is more general than "GRPO with renamed constants": it drops
zero-advantage groups through `filter_batch` without special trainer branches.

## Correctness Gates

Speed changes are allowed only behind equivalence gates. The original Phase 1
path is simple and readable; optimized rollout variants must match it
token-for-token at fixed seed and match loss within tolerance. The protocol
refactor was checked against the pre-refactor GRPO and Dr. GRPO losses with
zero loss and adapter-gradient difference.

Old-policy changes use a stronger gate than naive equality. At 4-bit, the
rollout-time cached decode realization and the full-forward realization are not
numerically identical. Stored rollout logprobs are attractive because they are
the behavior policy, but the actual gate is importance-ratio stability: on a
freshly sampled batch, `exp(logpi_current - logpi_old)` must stay near 1.0 and
short training runs must not show KL or gradient spikes. Until that gate justifies
a semantics change, `mlxrl` recomputes old-policy logprobs with a full forward.

## The 4-Bit KV Boundary

Quantized KV cache decode is not a drop-in numerical replacement for a full
forward over prompt+completion. That is fine for sampling, but it is not fine
for gradient-bearing or importance-weighting quantities. Anything used in the
loss denominator, KL, or adapter gradient must come from the full-forward path
unless a dedicated stability gate proves otherwise.

## Memory As A First-Class Constraint

Apple Silicon's unified memory is the deployment target, not an afterthought.
The code supports per-layer MLX-LM gradient checkpointing because DeltaNet and
linear-attention models otherwise keep O(sequence length) recurrent state live
through backward. The 9B fitting anchor is Qwen3.5-9B 4-bit at G=2, total
sequence length 609, with per-layer checkpointing at about 25.9 GB peak. G=4 at
the same shape is about 45.9 GB and tight on a 48 GB machine.

The memory estimator is deliberately conservative. It interpolates near measured
anchors and labels long-sequence uncheckpointed hybrid configs as OOM-risk
estimates, not measurements. Its purpose is to nudge users toward the knob that
will most likely make a run fit: enable checkpointing, reduce G, then reduce
completion length.
