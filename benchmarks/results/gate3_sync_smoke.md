# Phase 4 Benchmark Summary

| target | pass | status | rollout tok/s | gen toks | prompt toks | tok/s denom | grad s/step | samples/s | it/s | peak GB | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlxrl | 1 | ok | 406.855945 | 64 | 152 | 64 | 0.212575 | 13.732117 | 3.433029 | 1.884294 |  |
| mlx-lm | 1 | ok | 92.941347 | 16 | 152 | 16 | - | 11.586831 | - | 0.473465 | generation-only baseline; sequential G=1; gradient fields are not applicable |
| mlx-lm-g4 | 1 | ok | 85.901255 | 64 | 608 | 64 | - | 10.730883 | - | 0.485753 | generation-only baseline; sequential G=4; gradient fields are not applicable |
| mlx-tune | 1 | ok | 60.370019 | 64 | 152 | 64 | 0.186691 | 5.562490 | 1.390622 | 2.347517 | mlx-tune native trainer timed via package hooks |
| mlx-lm-lora | 1 | ok | 217.413767 | 64 | 172 | 64 | 0.099807 | 16.194839 | 4.048710 | 1.279730 | gradient phase is measured as step remainder after package rollout hook |

## Two-Pass Reproducibility

- mlxrl: only 1 successful pass(es).
- mlx-lm: only 1 successful pass(es).
- mlx-lm-g4: only 1 successful pass(es).
- mlx-tune: only 1 successful pass(es).
- mlx-lm-lora: only 1 successful pass(es).