# Contributing to GSEGUtils

## Dev install

- Editable install: `pip install -e .[dev]`
- Pre-commit hooks: `pre-commit install`

## Test authoring

### Seed discipline (TEST-08)

Every randomness source in tests MUST be seeded explicitly.

| Pattern | Status |
|---------|--------|
| `rng = np.random.default_rng(0)` then `rng.random(...)` etc. | REQUIRED |
| `np.random.default_rng()` (no seed) | FORBIDDEN — CI grep blocks |
| `np.random.{rand,choice,randint,uniform,normal,seed,...}` | FORBIDDEN — ruff NPY002 blocks |
| `import random; random.choice(...)` | DISCOURAGED — prefer numpy Generator |

Canonical exemplar: `tests/filters/test_downsample.py:17-29` (`pcd_all` fixture).

### Running tests

- All tests: `pytest -m "not benchmark"`
- Benchmarks: `pytest -m benchmark`

## Trust boundary (CI workflows)

- `pull_request_target` is FORBIDDEN in any `.github/workflows/*.yml`.
- Self-hosted GPU runner accepts only `push` (to `develop/gsd`/`main`/tags) + `workflow_dispatch`.

## See also

- `.planning/DESIGN-DECISIONS.md` — workspace-level architectural landmarks
