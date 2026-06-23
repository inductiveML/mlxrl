# Phase 4 Benchmark Summary

| target | pass | status | rollout tok/s | gen toks | prompt toks | tok/s denom | grad s/step | samples/s | it/s | peak GB | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlxrl | 1 | ok | 432.449599 | 64 | 152 | 64 | 0.203310 | 14.421718 | 3.605430 | 1.884294 |  |
| mlx-lm | 1 | ok | 91.260630 | 16 | 152 | 16 | - | 11.378277 | - | 0.498287 | generation-only baseline; sequential G=1; gradient fields are not applicable |
| mlx-lm-g4 | 1 | ok | 91.951821 | 64 | 608 | 64 | - | 11.487040 | - | 0.498287 | generation-only baseline; sequential G=4; gradient fields are not applicable |
| mlx-tune | 1 | ok | 65.337149 | 64 | 152 | 64 | 0.027018 | 5.924495 | 1.481124 | 2.343948 | mlx-tune native trainer timed via package hooks |
| mlx-lm-lora | 1 | ok | 258.974190 | 64 | 172 | 64 | 0.102236 | 17.714790 | 4.428698 | 1.279746 | gradient phase is measured as step remainder after package rollout hook |

## Two-Pass Reproducibility

- mlxrl: only 1 successful pass(es).
- mlx-lm: only 1 successful pass(es).
- mlx-lm-g4: only 1 successful pass(es).
- mlx-tune: only 1 successful pass(es).
- mlx-lm-lora: only 1 successful pass(es).