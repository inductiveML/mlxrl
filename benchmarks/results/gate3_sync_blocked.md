# Phase 4 Benchmark Summary

| target | pass | status | rollout tok/s | gen toks | prompt toks | tok/s denom | grad s/step | samples/s | it/s | peak GB | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlxrl | 1 | missing | - | 0 | 0 | 0 | - | - | - | - |   @partial(mx.compile, shapeless=True)      ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^ RuntimeError: [metal::load_device] No Metal device available. This typically occurs in headless, sandboxed, or virtualized macOS sessions where the GPU is not a |
| mlx-lm | 1 | missing | - | 0 | 0 | 0 | - | - | - | - |   @partial(mx.compile, shapeless=True)      ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^ RuntimeError: [metal::load_device] No Metal device available. This typically occurs in headless, sandboxed, or virtualized macOS sessions where the GPU is not a |
| mlx-lm-g4 | 1 | missing | - | 0 | 0 | 0 | - | - | - | - |   @partial(mx.compile, shapeless=True)      ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^ RuntimeError: [metal::load_device] No Metal device available. This typically occurs in headless, sandboxed, or virtualized macOS sessions where the GPU is not a |
| mlx-tune | 1 | missing | - | 0 | 0 | 0 | - | - | - | - |   @partial(mx.compile, shapeless=True)      ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^ RuntimeError: [metal::load_device] No Metal device available. This typically occurs in headless, sandboxed, or virtualized macOS sessions where the GPU is not a |
| mlx-lm-lora | 1 | missing | - | 0 | 0 | 0 | - | - | - | - |   @partial(mx.compile, shapeless=True)      ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^ RuntimeError: [metal::load_device] No Metal device available. This typically occurs in headless, sandboxed, or virtualized macOS sessions where the GPU is not a |

## Two-Pass Reproducibility
