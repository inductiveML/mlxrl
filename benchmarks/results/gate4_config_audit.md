# Gate 4 Config Audit

| target | status | version | model | 4bit | batch | G | ctx | max comp | temp | top_p | top_k | min_p | beta | LoRA | fast path | caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlxrl | match | 0.1.0 | mlx-community/Qwen3-0.6B-4bit | yes | 1 | 4 | 4096 | 256 | 0.7 | 0.95 | 0 | 0.0 | 0.04 | all transformer layers | prefix-cached grouped rollout, compiled decode | wired_limit_mb=32768 |
| mlx-lm | gen-only | 0.31.3 | mlx-community/Qwen3-0.6B-4bit | yes | 1 | 1 | 4096 | 256 | 0.7 | 0.95 | 0 | 0.0 | n/a | n/a | mlx-lm stream_generate | wired_limit_mb=32768; generation-only baseline |
| mlx-lm-g4 | gen-only | 0.31.3 | mlx-community/Qwen3-0.6B-4bit | yes | 1 | 4 | 4096 | 256 | 0.7 | 0.95 | 0 | 0.0 | n/a | n/a | mlx-lm stream_generate | wired_limit_mb=32768; generation-only baseline |
| mlx-tune | package-speed only | 0.5.1 | mlx-community/Qwen3-0.6B-4bit | yes | 1 | 4 | 4096 | 256 | 0.7 | 0.95 (configured, unused by generation) | 0 (configured, unused by generation) | 0.0 (configured, unused by generation) | 0.04 | all target modules via get_peft_model | prefix cache across group + stream_generate | wired_limit_mb=32768; installed mlx-tune generation consumes temperature only; top_p/top_k/min_p are not used |
| mlx-lm-lora | package-speed only | 2.1.0 | mlx-community/Qwen3-0.6B-4bit | yes | 1 | 4 | 4096 | 256 | 0.7 | 0.95 | 0 | 0.0 | 0.04 | num_layers=-1 | mlx-lm batch_generate with sampler knobs | wired_limit_mb=32768; uses package GRPO/reference semantics, not mlxrl single-model adapter-disable reference |
