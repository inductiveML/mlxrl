# mlxrl Design

`mlxrl` is a small, single-process RL post-training library for Apple Silicon.
The design goal is narrow: make on-policy, critic-free RL for local MLX LLMs
fast enough that rollout is the main problem and the loss stays thin.

## Critic-Free By Design

The library is intentionally limited to rollout-based policy-gradient methods:
GRPO, Dr. GRPO, DAPO, GSPO, RLOO, and GiGPO. PPO is out of scope because it
needs a critic/value model path, a different forward pass, and a different
memory profile. DPO and ORPO are also out of scope because they are offline
preference objectives rather than on-policy rollout algorithms.

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

Multi-turn algorithms use the sibling `TrajectoryAlgorithm` protocol rather
than widening the single-turn `Algorithm` contract. GiGPO is the first
implementation: the rollout engine records env-provided state ids, while the
algorithm owns anchor-state grouping and credit assignment.

## Agentic Environment Seam

Agentic rollout is built around `EnvFactory(task, seed, group_index) ->
Environment`. An environment exposes `reset()`, `step(action)`, `max_turns`,
and `state_id(observation)`. The state id is deliberately supplied by the env:
omega gym, WebShop, ALFWorld, or a toy env each decides what counts as the same
anchor state.

The engine records observations, actions, rewards, state ids, full token
history, and action-token spans. It does not compute anchor groups. This keeps
rollout reusable for future trajectory algorithms while GiGPO performs the
group-in-group advantage calculation offline from the same trajectories.

## Correctness Gates

Speed changes are allowed only behind equivalence gates. The original Phase 1
path is simple and readable; optimized rollout variants must match it
token-for-token at fixed seed and match loss within tolerance. The protocol
refactor was checked against the pre-refactor GRPO and Dr. GRPO losses with
zero loss and adapter-gradient difference.

GiGPO adds two reduction gates: `omega=0` must match episode-level GRPO over the
same trajectories, and single-turn one-step trajectories with `omega=0` must
match the v0.1 GRPO path through loss and adapter gradient. Toy multi-turn
cases also carry hand-computed advantage, loss, and gradient checks.

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

For multi-turn trajectories, this means policy, old-policy, and reference
logprobs are gathered from a full forward over the complete token history, then
masked down to action-token spans only. Rollout-time cached logprobs remain
diagnostic data, not training data.

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

Multi-turn configs add trajectory length as `prompt + turns * action_tokens +
(turns - 1) * observation_tokens`. Parallel-per-turn rollout keeps G
per-trajectory cache/state objects resident; sequential mode is the lower-memory
fallback. Until more measured anchors exist, multi-turn estimates are labeled as
OOM-risk estimates rather than confident measurements.
