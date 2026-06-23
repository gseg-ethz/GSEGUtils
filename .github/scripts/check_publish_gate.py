#!/usr/bin/env python3
# .github/scripts/check_publish_gate.py
# Source: Phase 11 RESEARCH.md Pattern 3 — PyYAML-based workflow publish gate
"""Assert publish steps appear ONLY in allowed files under allowed environments.

Called from the Lint (pre-commit) CI job. Exits 0 = clean; exits 1 = violation.
"""
import pathlib
import re
import sys

import yaml  # available on ubuntu-latest runner by default

ALLOWED: dict[str, str] = {
    "publish-pypi.yml": "pypi",
    "publish-testpypi.yml": "testpypi",
}

PUBLISH_STEP_PATTERNS = [
    r"pypa/gh-action-pypi-publish",  # covers any SHA/tag/branch ref
    r"twine\s+upload",  # covers `run: twine upload dist/*`
]


def get_env_name(env_field: object) -> str | None:
    """Normalize environment: 'pypi' and environment: {name: pypi} both."""
    if isinstance(env_field, str):
        return env_field
    if isinstance(env_field, dict):
        return env_field.get("name")
    return None


def is_publish_step(step: dict[str, object]) -> bool:
    """Return True if the step contains a publish action or twine upload."""
    uses = step.get("uses", "") or ""
    run = step.get("run", "") or ""
    return (
        any(re.search(p, str(uses)) for p in PUBLISH_STEP_PATTERNS)
        or any(re.search(p, str(run)) for p in PUBLISH_STEP_PATTERNS)
    )


violations: list[str] = []
workflows_dir = pathlib.Path(".github/workflows")
for wf_path in sorted(workflows_dir.glob("*.yml")):
    with wf_path.open() as f:
        wf = yaml.safe_load(f)
    if not isinstance(wf, dict) or "jobs" not in wf:
        continue
    for job_id, job in (wf.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        env_name = get_env_name(job.get("environment"))
        for step in job.get("steps") or []:
            if not isinstance(step, dict):
                continue
            if is_publish_step(step):
                allowed_env = ALLOWED.get(wf_path.name)
                if allowed_env is None:
                    violations.append(
                        f"{wf_path.name} job={job_id}: publish step found in non-allowed file"
                    )
                elif env_name != allowed_env:
                    violations.append(
                        f"{wf_path.name} job={job_id}: publish step requires"
                        f" environment={allowed_env!r}, got {env_name!r}"
                    )

if violations:
    for v in violations:
        print(f"::error::{v}")
    sys.exit(1)
print("check_publish_gate: OK — publish steps found only in allowed files + environments")
