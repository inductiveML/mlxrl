# Phase 4 Benchmark Summary

| target | pass | status | rollout tok/s | grad s/step | samples/s | it/s | peak GB | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlx-tune | 1 | ok | 36.255262 | 0.058942 | 1.929863 | 0.964931 | 1.402884 | mlx-tune native trainer timed via package hooks |
| mlx-lm-lora | 1 | ok | 90.133779 | 0.094146 | 10.934780 | 5.467390 | 1.022731 | gradient phase is measured as step remainder after package rollout hook |

## Two-Pass Reproducibility

- mlx-tune: only 1 successful pass(es).
- mlx-lm-lora: only 1 successful pass(es).