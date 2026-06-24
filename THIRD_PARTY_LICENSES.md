# Third-Party Notices

`mlxrl` reuses and adapts APIs, cache semantics, and sampling utilities from
upstream MLX projects. The repository keeps local attribution headers on files
that adapt those patterns.

## mlx-lm

- Source: https://github.com/ml-explore/mlx-lm
- License: MIT
- Used for: model loading, LoRA utilities, gradient checkpoint helper,
  prompt/KV-cache construction, cache state conventions, and sampling filters.

Upstream notice:

```text
MIT License Copyright © 2023 Apple Inc.
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of the
Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## mlx-tune

- Source: https://github.com/ARahim3/mlx-tune
- License: Apache-2.0 per installed package metadata.
- Used for: benchmark configuration inspection and external baseline adapter
  patterns only. No `mlx-tune` implementation code is vendored in `mlxrl`.

## mlx-lm-lora

- Source: https://github.com/Goekdeniz-Guelmez/mlx-lm-lora
- License: MIT per installed package metadata.
- Used for: benchmark configuration inspection and external baseline adapter
  patterns only. No `mlx-lm-lora` implementation code is vendored in `mlxrl`.
