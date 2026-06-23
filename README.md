# mlxrl

`mlxrl` is a small, single-process RL post-training library for LLMs on Apple
Silicon. The first target is GRPO with QLoRA on 0.6B-1.7B models using MLX.

Phase 0 only scaffolds the package, loads a 4-bit Qwen3 model, injects LoRA into
attention and MLP projections, verifies that only adapter parameters are
trainable, and runs a single forward pass.

```bash
UV_CACHE_DIR=.uv-cache uv run mlxrl phase0-smoke \
  --model mlx-community/Qwen3-0.6B-4bit \
  --prompt "What is 2+2?"
```

