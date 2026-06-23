# Phase 4 Benchmark Summary

| target | pass | status | rollout tok/s | gen toks | prompt toks | tok/s denom | grad s/step | samples/s | it/s | peak GB | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlx-tune | 1 | ok | 145.365112 | 5120 | 387 | 5120 | 0.490815 | 0.530686 | 0.132671 | 6.158254 | mlx-tune native trainer timed via package hooks |

## Two-Pass Reproducibility

- mlx-tune: only 1 successful pass(es).