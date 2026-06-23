# Phase 4 Benchmark Summary

| target | pass | status | rollout tok/s | grad s/step | samples/s | it/s | peak GB | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlxrl | 1 | ok | 498.278098 | 1.371051 | 1.167486 | 0.291871 | 6.245718 |  |
| mlx-lm | 1 | ok | 415.525316 | - | 1.623133 | - | 0.508494 | generation-only baseline; gradient fields are not applicable |
| mlx-tune | 1 | ok | 149.751613 | 0.002425 | 0.546223 | 0.136556 | 6.197945 | mlx-tune native trainer timed via package hooks |
| mlx-lm-lora | 1 | ok | 603.977253 | 0.532907 | 1.795048 | 0.448762 | 5.361539 | gradient phase is measured as step remainder after package rollout hook |
| mlxrl | 2 | ok | 471.440287 | 1.464240 | 1.100006 | 0.275001 | 6.245734 |  |
| mlx-lm | 2 | ok | 356.850114 | - | 1.393936 | - | 0.500843 | generation-only baseline; gradient fields are not applicable |
| mlx-tune | 2 | ok | 141.398462 | 0.002486 | 0.516523 | 0.129131 | 6.196242 | mlx-tune native trainer timed via package hooks |
| mlx-lm-lora | 2 | ok | 580.406387 | 0.557123 | 1.723081 | 0.430770 | 5.361542 | gradient phase is measured as step remainder after package rollout hook |

## Two-Pass Reproducibility

- mlxrl: throughput relative delta 0.057799
- mlx-lm: throughput relative delta 0.141207
- mlx-tune: throughput relative delta 0.054373
- mlx-lm-lora: throughput relative delta 0.040092