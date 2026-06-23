# Phase 4 Benchmark Summary

| target | pass | status | rollout tok/s | grad s/step | samples/s | it/s | peak GB | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlxrl | 1 | ok | 389.748369 | 1.995388 | 0.865283 | 0.216321 | 6.245718 |  |
| mlx-lm | 1 | ok | 314.017700 | - | 1.226623 | - | 0.496256 | generation-only baseline; gradient fields are not applicable |
| mlxrl | 2 | ok | 476.303930 | 1.353725 | 1.141665 | 0.285416 | 6.245750 |  |
| mlx-lm | 2 | ok | 398.752526 | - | 1.557617 | - | 0.500843 | generation-only baseline; gradient fields are not applicable |

## Two-Pass Reproducibility

- mlxrl: throughput relative delta 0.242087
- mlx-lm: throughput relative delta 0.212500