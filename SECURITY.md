# Security policy

Security fixes target the latest released version. Use inductive-mlxrl 0.2.4 or
later and MLX-LM 0.32.0 or later. Earlier supported MLX-LM versions could execute
Python from a model's `model_file` configuration without an explicit trust flag.
See the [upstream fix](https://github.com/ml-explore/mlx-lm/pull/1385).

Report vulnerabilities through the repository's
[private vulnerability reporting form](https://github.com/inductiveML/mlxrl/security/advisories/new).
Include the affected version, a minimal reproduction, and the expected impact.
Keep credentials and private training data out of reports and public issues.

## Trust boundaries

Model completions and action text are untrusted data. Built-in tag parsers scan
fixed delimiters without regex backtracking. Model loading explicitly disables
custom remote Python code.

Training configurations, custom Python reward functions, environment callbacks,
benchmark command adapters, model repositories, and adapter files must come from
sources you trust. The library runs in the caller's process and does not provide
a sandbox for these extensions or for native tensor parsers. CLI output paths
must resolve inside the working directory; this check does not isolate a
process from another user who can concurrently change its files or symlinks.

## Release controls

Release tags must match the package version, identify a commit already on main,
and have successful main push CI. CI audits every locked dependency version,
including packages for other platforms and Python versions. Release actions
use verified commit pins and build tools come from the frozen lockfile.

GitHub's `pypi` environment restricts publication to protected release tags and
requires a maintainer's approval. PyPI publication uses OIDC Trusted Publishing
and signed attestations. Local Metal tests remain a release requirement.
