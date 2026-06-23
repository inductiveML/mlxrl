# Phase 4 Benchmark Summary

| target | pass | status | rollout tok/s | grad s/step | samples/s | it/s | peak GB | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlx-lm | 1 | ok | 30.779382 | - | 7.693060 | - | 0.473875 | generation-only baseline; sequential G=1; gradient fields are not applicable |
| mlx-lm-g4 | 1 | ok | 48.465266 | - | 12.114592 | - | 0.473465 | generation-only baseline; sequential G=4; gradient fields are not applicable |

## Two-Pass Reproducibility

- mlx-lm: only 1 successful pass(es).
- mlx-lm-g4: only 1 successful pass(es).