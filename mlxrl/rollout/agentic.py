"""Multi-turn agentic rollout over Gym-style text environments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from mlxrl.env import (
    ActionParser,
    BatchEnvironment,
    EnvFactory,
    Environment,
    coerce_step_result,
    default_action_parser,
)
from mlxrl.rollout.naive import SamplingConfig, decode_completion, sampled_token_logprobs
from mlxrl.trajectory import ActionSpan, Trajectory, TrajectoryStep

RolloutMode = Literal["parallel_per_turn", "sequential"]


@dataclass(frozen=True)
class GeneratedAction:
    """One sampled model action."""

    tokens: tuple[int, ...]
    old_policy_logprobs: tuple[float, ...]
    text: str

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("Generated actions must contain at least one token.")
        if len(self.tokens) != len(self.old_policy_logprobs):
            raise ValueError("Generated action tokens/logprobs must align.")


ActionGenerator = Callable[[int, int, int, Sequence[int], str], GeneratedAction]


@dataclass
class _RolloutState:
    task_index: int
    group_index: int
    task: str
    env: Environment
    observation: str
    initial_observation: str
    full_tokens: list[int]
    pending_context_tokens: list[int]
    cache: list[Any] | None
    steps: list[TrajectoryStep]
    spans: list[ActionSpan]
    finished: bool = False
    done: bool = False
    truncated: bool = False

    @property
    def turn_index(self) -> int:
        return len(self.steps)


def generate_agentic_trajectories(
    *,
    model: nn.Module | None,
    tokenizer: Any,
    env_factory: EnvFactory,
    tasks: Sequence[Any],
    group_size: int,
    sampling: SamplingConfig,
    seed: int | None = None,
    rollout_mode: RolloutMode = "parallel_per_turn",
    parser: ActionParser | None = None,
    action_generator: ActionGenerator | None = None,
) -> tuple[Trajectory, ...]:
    """Generate G multi-turn trajectories per task.

    ``parallel_per_turn`` keeps all group caches resident and advances active
    trajectories turn-by-turn. ``sequential`` completes one trajectory at a time
    to reduce peak cache/state memory.
    """

    if group_size < 1:
        raise ValueError("group_size must be at least 1.")
    if rollout_mode not in {"parallel_per_turn", "sequential"}:
        raise ValueError("rollout_mode must be 'parallel_per_turn' or 'sequential'.")
    if seed is not None:
        mx.random.seed(seed)

    parser_impl = parser.parse_action if parser is not None else default_action_parser
    trajectories: list[Trajectory] = []
    for task_index, task in enumerate(tasks):
        states = [
            _initial_state(
                model=model,
                tokenizer=tokenizer,
                env=env_factory(task, int(seed or 0), group_index),
                task=str(task),
                task_index=task_index,
                group_index=group_index,
                use_cache=action_generator is None,
            )
            for group_index in range(group_size)
        ]
        if rollout_mode == "sequential":
            for state in states:
                while not state.finished:
                    _advance_states(
                        [state],
                        model,
                        tokenizer,
                        sampling,
                        parser_impl,
                        action_generator,
                    )
        else:
            while any(not state.finished for state in states):
                active = [state for state in states if not state.finished]
                _advance_states(
                    active,
                    model,
                    tokenizer,
                    sampling,
                    parser_impl,
                    action_generator,
                )
        trajectories.extend(_finalize_state(state) for state in states)
    return tuple(trajectories)


def cache_carry_logit_error(
    model: nn.Module,
    token_segments: Sequence[Sequence[int]],
) -> float:
    """Compare segmented cache carry with token-by-token incremental cache."""

    if not token_segments:
        raise ValueError("At least one token segment is required.")
    if any(not segment for segment in token_segments):
        raise ValueError("Token segments must be non-empty.")
    segmented_cache = make_prompt_cache(model)
    segmented_logits = None
    for segment in token_segments:
        for token in segment:
            token_ids = mx.array([int(token)], dtype=mx.int32)
            segmented_logits = model(token_ids[None], cache=segmented_cache)
    if segmented_logits is None:
        raise RuntimeError("Internal error: no segmented logits were computed.")

    full_tokens: list[int] = []
    for segment in token_segments:
        full_tokens.extend(int(token) for token in segment)
    incremental_cache = make_prompt_cache(model)
    incremental_logits = None
    for token in full_tokens:
        token_ids = mx.array([int(token)], dtype=mx.int32)
        incremental_logits = model(token_ids[None], cache=incremental_cache)
    if incremental_logits is None:
        raise RuntimeError("Internal error: no incremental logits were computed.")
    error = mx.max(mx.abs(segmented_logits[:, -1, :] - incremental_logits[:, -1, :]))
    mx.eval(  # Hybrid carry sync: materialize cache-carry equivalence scalar.
        error
    )
    return float(error.item())


def _initial_state(
    *,
    model: nn.Module | None,
    tokenizer: Any,
    env: Environment,
    task: str,
    task_index: int,
    group_index: int,
    use_cache: bool,
) -> _RolloutState:
    observation = env.reset()
    context_tokens = _encode_text(tokenizer, _initial_context(task, observation))
    return _RolloutState(
        task_index=task_index,
        group_index=group_index,
        task=task,
        env=env,
        observation=observation,
        initial_observation=observation,
        full_tokens=list(context_tokens),
        pending_context_tokens=list(context_tokens),
        cache=make_prompt_cache(model) if use_cache and model is not None else None,
        steps=[],
        spans=[],
    )


def _advance_states(
    states: Sequence[_RolloutState],
    model: nn.Module | None,
    tokenizer: Any,
    sampling: SamplingConfig,
    parser: Callable[[str], str],
    action_generator: ActionGenerator | None,
) -> None:
    generated = [
        (
            state,
            _generate_action(
                state,
                model,
                tokenizer,
                sampling,
                action_generator,
            ),
        )
        for state in states
    ]
    actions = [parser(action.text) for _, action in generated]
    if (
        len(states) > 1
        and isinstance(states[0].env, BatchEnvironment)
        and all(state.env is states[0].env for state in states)
    ):
        raw_results = states[0].env.step_batch(actions)
        if len(raw_results) != len(states):
            raise ValueError("BatchEnvironment.step_batch returned the wrong row count.")
        results = [coerce_step_result(result) for result in raw_results]
    else:
        results = [
            coerce_step_result(state.env.step(action))
            for (state, _), action in zip(generated, actions, strict=True)
        ]

    for (state, action), parsed_action, result in zip(
        generated,
        actions,
        results,
        strict=True,
    ):
        _record_step(state, tokenizer, action, parsed_action, result)


def _generate_action(
    state: _RolloutState,
    model: nn.Module | None,
    tokenizer: Any,
    sampling: SamplingConfig,
    action_generator: ActionGenerator | None,
) -> GeneratedAction:
    if action_generator is not None:
        return action_generator(
            state.task_index,
            state.group_index,
            state.turn_index,
            tuple(state.full_tokens),
            state.observation,
        )
    if model is None or state.cache is None:
        raise ValueError("model is required when action_generator is not provided.")
    return _generate_model_action(model, tokenizer, state, sampling)


def _generate_model_action(
    model: nn.Module,
    tokenizer: Any,
    state: _RolloutState,
    sampling: SamplingConfig,
) -> GeneratedAction:
    if sampling.max_tokens < 1:
        raise ValueError("sampling.max_tokens must be at least 1.")
    if not state.pending_context_tokens:
        raise ValueError("No pending context tokens available for action generation.")
    if state.cache is None:
        raise ValueError("No cache available for model action generation.")

    sampler = make_sampler(
        temp=sampling.temperature,
        top_p=sampling.top_p,
        min_p=sampling.min_p,
        top_k=sampling.top_k,
    )
    logprobs = _incremental_prefill_logprobs(
        model,
        state.cache,
        state.pending_context_tokens,
    )
    state.pending_context_tokens.clear()
    tokens: list[int] = []
    old_logprobs: list[float] = []
    current: mx.array | None = None

    for index in range(sampling.max_tokens):
        if index > 0:
            if current is None:
                raise RuntimeError("Internal error: missing previous decode token.")
            logits = model(current, cache=state.cache)
            logprobs = logits[:, -1, :] - mx.logsumexp(logits[:, -1, :], axis=-1, keepdims=True)
        next_token = sampler(logprobs)
        sampled_logprob = sampled_token_logprobs(logprobs, next_token)
        mx.eval(  # Rollout sync: materialize sampled action token/logprob for env step.
            next_token,
            sampled_logprob,
        )
        token_id = int(next_token.item())
        tokens.append(token_id)
        old_logprobs.append(float(sampled_logprob.item()))
        current = next_token

    return GeneratedAction(
        tokens=tuple(tokens),
        old_policy_logprobs=tuple(old_logprobs),
        text=decode_completion(tokenizer, tokens),
    )


def _incremental_prefill_logprobs(
    model: nn.Module,
    cache: list[Any],
    tokens: Sequence[int],
) -> mx.array:
    logits = None
    for token in tokens:
        token_ids = mx.array([int(token)], dtype=mx.int32)
        logits = model(token_ids[None], cache=cache)
    if logits is None:
        raise ValueError("No tokens supplied for incremental prefill.")
    return logits[:, -1, :] - mx.logsumexp(logits[:, -1, :], axis=-1, keepdims=True)


def _record_step(
    state: _RolloutState,
    tokenizer: Any,
    action: GeneratedAction,
    parsed_action: str,
    result: Any,
) -> None:
    start = len(state.full_tokens)
    state.full_tokens.extend(action.tokens)
    end = len(state.full_tokens)
    span = ActionSpan(step_index=state.turn_index, start=start, end=end)
    step = TrajectoryStep(
        observation=state.observation,
        state_id=state.env.state_id(state.observation),
        action_text=parsed_action,
        action_tokens=action.tokens,
        old_policy_logprobs=action.old_policy_logprobs,
        reward=float(result.reward),
        done=bool(result.done),
        info=result.info,
        next_observation=result.observation,
    )
    state.spans.append(span)
    state.steps.append(step)

    reached_limit = len(state.steps) >= int(state.env.max_turns)
    state.done = bool(result.done)
    state.truncated = not state.done and reached_limit
    state.finished = state.done or state.truncated
    state.observation = result.observation
    if not state.finished:
        observation_tokens = _encode_text(tokenizer, _observation_context(result.observation))
        state.full_tokens.extend(observation_tokens)
        state.pending_context_tokens = [action.tokens[-1], *observation_tokens]
    else:
        state.pending_context_tokens = []


def _finalize_state(state: _RolloutState) -> Trajectory:
    return Trajectory(
        task_index=state.task_index,
        group_index=state.group_index,
        task=state.task,
        initial_observation=state.initial_observation,
        full_token_ids=tuple(state.full_tokens),
        action_spans=tuple(state.spans),
        steps=tuple(state.steps),
        done=state.done,
        truncated=state.truncated,
    )


def _initial_context(task: str, observation: str) -> str:
    return f"Task:\n{task}\nObservation:\n{observation}\nAction:\n"


def _observation_context(observation: str) -> str:
    return f"\nObservation:\n{observation}\nAction:\n"


def _encode_text(tokenizer: Any, text: str) -> tuple[int, ...]:
    tokens = tuple(int(token) for token in tokenizer.encode(text))
    if not tokens:
        raise ValueError("Tokenizer produced no context tokens.")
    return tokens
