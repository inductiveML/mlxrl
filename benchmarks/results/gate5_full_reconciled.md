# Phase 4 Benchmark Summary

Rollout tok/s definition: generated completion tokens only, excluding prompt/prefix tokens and warmup steps, divided by synchronized generation phase wall-clock seconds.

Sync guarantee: every timed rollout, optimizer, and full-run boundary calls `mx.eval` or `mx.synchronize` on the relevant MLX outputs/state before stopping the timer. Synced gradient steps under 10 ms are rejected.

Comparison labels: `apples-to-apples` is reserved for mlxrl's own GRPO semantics/config; `package-speed only` means the package fast path is useful but training or sampling semantics differ; `gen-only` has no gradient phase.

| target | pass | comparison | status | rollout tok/s | gen toks | prompt toks | tok/s denom | grad s/step | samples/s | it/s | peak GB | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlxrl | 1 | apples-to-apples | ok | 443.957855 | 97280 | 7378 | 97280 | 1.406149 | 1.077378 | 0.269345 | 6.245587 |  |
| mlx-lm | 1 | gen-only | ok | 360.847105 | 24320 | 7378 | 24320 | - | 1.409099 | - | 0.500843 | generation-only baseline; sequential G=1; gradient fields are not applicable |
| mlx-lm-g4 | 1 | gen-only | ok | 366.291108 | 97280 | 29512 | 97280 | - | 1.430710 | - | 0.500843 | generation-only baseline; sequential G=4; gradient fields are not applicable |
| mlx-tune | 1 | package-speed only | ok | 141.249407 | 97280 | 7378 | 97280 | 0.506961 | 0.515544 | 0.128886 | 6.157581 | mlx-tune native trainer timed via package hooks |
| mlx-lm-lora | 1 | package-speed only | ok | 560.139887 | 97275 | 8170 | 97275 | 0.590363 | 1.653997 | 0.413499 | 5.320306 | gradient phase is measured as step remainder after package rollout hook |
| mlxrl | 2 | apples-to-apples | ok | 464.255706 | 97280 | 7378 | 97280 | 1.158277 | 1.189057 | 0.297264 | 6.245521 |  |
| mlx-lm | 2 | gen-only | ok | 333.196434 | 24320 | 7378 | 24320 | - | 1.301128 | - | 0.515245 | generation-only baseline; sequential G=1; gradient fields are not applicable |
| mlx-lm-g4 | 2 | gen-only | ok | 333.148981 | 97280 | 29512 | 97280 | - | 1.301259 | - | 0.515245 | generation-only baseline; sequential G=4; gradient fields are not applicable |
| mlx-tune | 2 | package-speed only | ok | 143.188061 | 97280 | 7378 | 97280 | 0.497069 | 0.522823 | 0.130706 | 6.157583 | mlx-tune native trainer timed via package hooks |
| mlx-lm-lora | 2 | package-speed only | ok | 555.729150 | 97275 | 8170 | 97275 | 0.592947 | 1.642389 | 0.410597 | 5.320317 | gradient phase is measured as step remainder after package rollout hook |

## Two-Pass Reproducibility

- mlxrl: throughput relative delta 0.093922
- mlx-lm: throughput relative delta 0.076624
- mlx-lm-g4: throughput relative delta 0.090480
- mlx-tune: throughput relative delta 0.013923
- mlx-lm-lora: throughput relative delta 0.007018