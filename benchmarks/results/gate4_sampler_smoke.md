# Phase 4 Benchmark Summary

| target | pass | status | rollout tok/s | gen toks | prompt toks | tok/s denom | grad s/step | samples/s | it/s | peak GB | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlxrl | 1 | ok | 402.140771 | 64 | 152 | 64 | 0.216344 | 13.515018 | 3.378755 | 1.882852 |  |
| mlx-lm | 1 | ok | 87.428648 | 16 | 152 | 16 | - | 10.898698 | - | 0.476726 | generation-only baseline; sequential G=1; gradient fields are not applicable |
| mlx-lm-g4 | 1 | ok | 84.090863 | 64 | 608 | 64 | - | 10.505105 | - | 0.497878 | generation-only baseline; sequential G=4; gradient fields are not applicable |
| mlx-tune | 1 | ok | 60.349186 | 64 | 152 | 64 | 0.186617 | 5.562895 | 1.390724 | 2.344356 | mlx-tune native trainer timed via package hooks |
| mlx-lm-lora | 1 | ok | 226.190006 | 64 | 172 | 64 | 0.102847 | 16.371887 | 4.092972 | 1.279732 | gradient phase is measured as step remainder after package rollout hook |

## Two-Pass Reproducibility

- mlxrl: only 1 successful pass(es).
- mlx-lm: only 1 successful pass(es).
- mlx-lm-g4: only 1 successful pass(es).
- mlx-tune: only 1 successful pass(es).
- mlx-lm-lora: only 1 successful pass(es).