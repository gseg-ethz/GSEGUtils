#!/usr/bin/env python3
# .github/scripts/test_assert_no_skip.py
# Source: Phase 13 plan 13-16 Task 1 — unit tests for the base-ref anti-skip assertion
"""Unit tests for the three-rule anti-skip assertion.

Every case builds a **hybrid tree** in a ``tmp_path`` fixture: a
``.github/rulesets/`` directory standing in for the base checkout's committed
payloads, and a ``.github/workflows/`` directory standing in for the pull
request head's workflow set. That split is the mechanism under test, not an
implementation detail -- the rules come from the base side and the jobs come
from the head side, so an edit to the head cannot change what judges it.

A **clean control** sits beside the negatives, deliberately: a checker that
raised on everything would satisfy every negative test on its own, so
``test_a_clean_hybrid_tree_produces_no_violations`` is what distinguishes a
working assertion from a broken one.

Run with::

    python -m pytest .github/scripts/test_assert_no_skip.py -q

Note that ruff's ``per-file-ignores`` relax docstring rules under ``tests/**``
only, which does not cover ``.github/scripts/``, so every test here carries a
numpy-convention docstring by design rather than by accident.
"""

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import assert_no_skip as ans  # noqa: E402  — deliberately after the sys.path insertion above

DEFAULT_CONTEXTS = ("Lint (pre-commit)", "Tests (pytest)")

# A minimal head workflow producing both default required contexts, with a clean
# trigger block written in the BARE `on:` form -- the form live in both repos
# today, and therefore the form a naive parser would silently read as job-less.
CLEAN_CI_YML = """\
name: CI

on:
  pull_request:
  push:
    branches: [main]

jobs:
  lint:
    name: Lint (pre-commit)
    runs-on: ubuntu-latest
    steps:
      - run: echo lint
  tests:
    name: Tests (pytest)
    runs-on: ubuntu-latest
    steps:
      - run: echo test
"""

UNNAMED_JOB_YML = """\
name: Canary

on:
  workflow_dispatch:

jobs:
  canary:
    runs-on: ubuntu-latest
    steps:
      - run: echo canary
"""

NO_TRIGGER_YML = """\
name: Broken

jobs:
  stub:
    runs-on: ubuntu-latest
    steps:
      - run: echo stub
"""


def build_hybrid_tree(
    root: pathlib.Path,
    workflows: dict[str, str],
    *,
    contexts: tuple[str, ...] = DEFAULT_CONTEXTS,
    payloads: dict[str, tuple[str, ...]] | None = None,
    make_workflow_dir: bool = True,
) -> pathlib.Path:
    """Write a hybrid tree: base-side rulesets plus head-side workflows.

    Parameters
    ----------
    root
        Directory to populate; normally a ``tmp_path`` fixture.
    workflows
        Mapping of workflow filename to file text, written to the head side.
    contexts
        Required status-check contexts written into a single ``main.json``
        payload on the base side. Ignored when `payloads` is given.
    payloads
        Mapping of payload filename to its required contexts, when more than one
        committed payload is needed.
    make_workflow_dir
        When false the workflows directory is not created at all, which is the
        failed-fetch shape.

    Returns
    -------
    pathlib.Path
        The populated root, for chaining.
    """
    if make_workflow_dir:
        workflows_dir = root / ans.WORKFLOWS_DIR
        workflows_dir.mkdir(parents=True, exist_ok=True)
        for name, text in workflows.items():
            (workflows_dir / name).write_text(text)

    rulesets_dir = root / ans.RULESETS_DIR
    rulesets_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload_contexts in (payloads or {"main.json": contexts}).items():
        (rulesets_dir / filename).write_text(
            json.dumps(
                {
                    "name": "protect-main",
                    "target": "branch",
                    "conditions": {"ref_name": {"include": ["refs/heads/main"], "exclude": []}},
                    "bypass_actors": [],
                    "rules": [
                        {
                            "type": "required_status_checks",
                            "parameters": {
                                "required_status_checks": [
                                    {"context": c, "integration_id": None} for c in payload_contexts
                                ]
                            },
                        }
                    ],
                }
            )
        )
    return root


def with_job_condition(text: str, job_name_line: str) -> str:
    """Return `text` with a job-level condition inserted under `job_name_line`.

    Parameters
    ----------
    text
        The workflow text.
    job_name_line
        The exact ``    name: ...`` line to insert beneath.

    Returns
    -------
    str
        The rewritten workflow text.
    """
    return text.replace(job_name_line, f"{job_name_line}    if: false\n", 1)


# --------------------------------------------------------------------------
# The control — a checker that raised on everything would fail here
# --------------------------------------------------------------------------


def test_a_clean_hybrid_tree_produces_no_violations(tmp_path: pathlib.Path) -> None:
    """Base-side rulesets plus a clean head workflow set produce nothing at all."""
    root = build_hybrid_tree(tmp_path, {"ci.yml": CLEAN_CI_YML})
    assert ans.run_all(root) == []


def test_the_exit_status_is_zero_only_when_there_are_no_violations(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A clean tree exits 0 and prints no error annotation."""
    root = build_hybrid_tree(tmp_path, {"ci.yml": CLEAN_CI_YML})
    status = ans.main([str(root)])
    captured = capsys.readouterr()
    assert status == 0
    assert "::error::" not in captured.out


# --------------------------------------------------------------------------
# Rule 1 — a job-level condition on a required-context job (Phase 12 D-10)
# --------------------------------------------------------------------------


def test_a_job_level_condition_on_a_required_context_job_is_one_violation(tmp_path: pathlib.Path) -> None:
    """A job-level condition on a required-context job is exactly one violation, named."""
    root = build_hybrid_tree(tmp_path, {"ci.yml": with_job_condition(CLEAN_CI_YML, "    name: Lint (pre-commit)\n")})
    violations = ans.run_all(root)
    assert len(violations) == 1, violations
    assert "lint" in violations[0]
    assert "Lint (pre-commit)" in violations[0]


# --------------------------------------------------------------------------
# Rule 2 — the propagation shape: a dependency edge onto a conditional job
# --------------------------------------------------------------------------


def test_a_required_context_job_depending_on_a_conditional_job_is_one_violation(tmp_path: pathlib.Path) -> None:
    """A required-context job needing a conditional job is exactly one violation, naming the edge."""
    workflow = CLEAN_CI_YML.replace(
        "  lint:\n    name: Lint (pre-commit)\n",
        "  lint:\n    name: lint-helper\n    if: false\n",
        1,
    ).replace("    name: Tests (pytest)\n", "    name: Tests (pytest)\n    needs: [lint]\n", 1)
    # `Lint (pre-commit)` is no longer produced by any job, so this fixture needs
    # a payload that requires only the context under test.
    root = build_hybrid_tree(tmp_path, {"ci.yml": workflow}, contexts=("Tests (pytest)",))
    violations = ans.run_all(root)
    assert len(violations) == 1, violations
    assert "Tests (pytest)" in violations[0]
    assert "lint" in violations[0]


def test_both_conditionality_shapes_at_once_produce_both_violations(tmp_path: pathlib.Path) -> None:
    """Both fail-open shapes present together are reported as two violations, not one."""
    workflow = with_job_condition(CLEAN_CI_YML, "    name: Lint (pre-commit)\n").replace(
        "    name: Tests (pytest)\n", "    name: Tests (pytest)\n    needs: [lint]\n", 1
    )
    root = build_hybrid_tree(tmp_path, {"ci.yml": workflow})
    violations = ans.run_all(root)
    assert len(violations) == 2, violations
    assert any("Lint (pre-commit)" in v and "job-level condition" in v for v in violations)
    assert any("Tests (pytest)" in v and "depends" in v for v in violations)


def test_the_two_conditionality_rules_are_the_self_tests_own_assertions() -> None:
    """Rules 1 and 2 are invoked from the self-test module, never reimplemented here.

    A second parser for one rule is how a branch gate and a build-time check stop
    agreeing about what a violation is. Pinned as a test rather than left to a
    grep in a plan document, so it survives the plan.
    """
    assert ans.check_ci_config.check_job_level_conditions is not None
    assert ans.check_ci_config.check_conditional_dependencies is not None
    source = pathlib.Path(ans.__file__).read_text(encoding="utf-8")
    assert "def check_job_level_conditions" not in source
    assert "def check_conditional_dependencies" not in source


# --------------------------------------------------------------------------
# Rule 3 — a required context no head job produces
# --------------------------------------------------------------------------


def test_a_required_context_no_head_job_produces_is_a_violation(tmp_path: pathlib.Path) -> None:
    """A context nothing in the head declares is a violation naming it and its payload."""
    root = build_hybrid_tree(
        tmp_path,
        {"ci.yml": CLEAN_CI_YML},
        contexts=(*DEFAULT_CONTEXTS, "Integrity (base-ref)"),
    )
    violations = ans.run_all(root)
    assert len(violations) == 1, violations
    assert "Integrity (base-ref)" in violations[0]
    assert "main.json" in violations[0]


def test_deleting_the_producing_workflow_in_the_head_is_caught_by_rule_three(tmp_path: pathlib.Path) -> None:
    """Removing the job that produces a required context is the missing-producer shape.

    Cut at the job body, not at the job id: renaming ``tests:`` alone leaves the
    declared ``name: Tests (pytest)`` in place, and GitHub reports the declared
    name — so the context would still be produced and the fixture would prove
    nothing.
    """
    gutted = CLEAN_CI_YML.split("  tests:\n")[0]
    root = build_hybrid_tree(tmp_path, {"ci.yml": gutted})
    violations = ans.run_all(root)
    assert len(violations) == 1, violations
    assert "Tests (pytest)" in violations[0]


def test_every_payload_requiring_an_unproduced_context_is_named(tmp_path: pathlib.Path) -> None:
    """The violation names every committed payload that requires the missing context."""
    root = build_hybrid_tree(
        tmp_path,
        {"ci.yml": CLEAN_CI_YML},
        payloads={
            "main.json": (*DEFAULT_CONTEXTS, "Integrity (base-ref)"),
            "main.no-gpu.json": (*DEFAULT_CONTEXTS, "Integrity (base-ref)"),
            "develop.json": DEFAULT_CONTEXTS,
        },
    )
    violations = ans.run_all(root)
    assert len(violations) == 1, violations
    assert "main.json" in violations[0]
    assert "main.no-gpu.json" in violations[0]
    assert "develop.json" not in violations[0]


def test_a_required_context_produced_by_an_unnamed_job_matches_on_the_job_id(tmp_path: pathlib.Path) -> None:
    """An unnamed job matches on its bare job id, which is the string GitHub reports."""
    root = build_hybrid_tree(
        tmp_path,
        {"ci.yml": CLEAN_CI_YML, "canary.yml": UNNAMED_JOB_YML},
        contexts=(*DEFAULT_CONTEXTS, "canary"),
    )
    assert ans.run_all(root) == []


# --------------------------------------------------------------------------
# The trigger-key trap, inherited rather than re-derived
# --------------------------------------------------------------------------


def test_both_trigger_key_forms_are_read_and_neither_is_flagged(tmp_path: pathlib.Path) -> None:
    """A bare `on:` and a quoted `"on":` are both resolved; neither form is flagged."""
    quoted = CLEAN_CI_YML.replace("\non:", '\n"on":', 1).replace(
        "  lint:\n    name: Lint (pre-commit)\n", "  quoted:\n    name: Docs (sphinx -W)\n", 1
    )
    root = build_hybrid_tree(
        tmp_path,
        {"ci.yml": CLEAN_CI_YML, "docs.yml": quoted},
        contexts=(*DEFAULT_CONTEXTS, "Docs (sphinx -W)"),
    )
    assert ans.run_all(root) == []


def test_a_head_workflow_with_neither_trigger_key_form_is_a_violation(tmp_path: pathlib.Path) -> None:
    """A workflow whose trigger block resolves under neither key form is a violation, not a skip."""
    root = build_hybrid_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, "broken.yml": NO_TRIGGER_YML})
    violations = ans.run_all(root)
    assert len(violations) == 1, violations
    assert "broken.yml" in violations[0]
    assert "trigger" in violations[0].lower()


def test_an_unreadable_workflow_does_not_silently_contribute_an_empty_job_set(tmp_path: pathlib.Path) -> None:
    """A workflow whose triggers cannot be read is reported; its jobs are not assumed absent.

    The distinguishing case: the file that cannot be read is the ONLY producer of
    a required context. A checker that skipped it would emit two violations -- the
    unreadable file AND a spurious missing-producer -- or, if it also swallowed the
    read failure, would emit the missing-producer alone. Exactly one violation, the
    unreadable one, is the correct reading.
    """
    root = build_hybrid_tree(
        tmp_path,
        {"ci.yml": NO_TRIGGER_YML.replace("    name: ", "    name: ", 1)},
        contexts=("stub",),
    )
    violations = ans.run_all(root)
    assert len(violations) == 1, violations
    assert "ci.yml" in violations[0]
    assert "trigger" in violations[0].lower()


def test_a_workflow_that_parses_to_a_non_mapping_is_a_violation(tmp_path: pathlib.Path) -> None:
    """A workflow document that is not a mapping has not been checked, so it is a violation."""
    root = build_hybrid_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, "list.yml": "- one\n- two\n"})
    violations = ans.run_all(root)
    assert len(violations) == 1, violations
    assert "list.yml" in violations[0]


def test_a_workflow_that_is_not_valid_yaml_is_a_violation(tmp_path: pathlib.Path) -> None:
    """An unparseable workflow is a hard violation with a named reason, never a skip."""
    root = build_hybrid_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, "bad.yml": "name: [unclosed\n"})
    violations = ans.run_all(root)
    assert len(violations) == 1, violations
    assert "bad.yml" in violations[0]


def test_a_yaml_extension_workflow_is_reported_as_outside_the_shared_assertions_reach(
    tmp_path: pathlib.Path,
) -> None:
    """A `.yaml` head workflow is a violation: rules 1 and 2 glob `*.yml` and would not see it.

    GitHub accepts either extension, so renaming a required-check workflow to
    `.yaml` in the head and adding a job-level condition to it would slip past the
    two shared assertions while still producing the context. An unchecked
    workflow is not a clean one.
    """
    root = build_hybrid_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, "extra.yaml": UNNAMED_JOB_YML})
    violations = ans.run_all(root)
    assert len(violations) == 1, violations
    assert "extra.yaml" in violations[0]


# --------------------------------------------------------------------------
# A failed fetch is a hard failure with a named reason, never an empty set
# --------------------------------------------------------------------------


def test_a_missing_head_workflow_directory_is_a_violation(tmp_path: pathlib.Path) -> None:
    """An absent workflow directory means nothing was fetched, which is a violation."""
    root = build_hybrid_tree(tmp_path, {}, make_workflow_dir=False)
    violations = ans.run_all(root)
    assert violations, violations
    assert any("does not exist" in v for v in violations)


def test_an_empty_head_workflow_directory_is_a_violation(tmp_path: pathlib.Path) -> None:
    """A workflow directory holding no workflow files is a violation, not a clean result."""
    root = build_hybrid_tree(tmp_path, {})
    violations = ans.run_all(root)
    assert violations, violations
    assert any("no workflow files" in v for v in violations)


# --------------------------------------------------------------------------
# Driver behaviour
# --------------------------------------------------------------------------


def test_the_exit_status_is_non_zero_when_any_violation_is_found(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A violation exits non-zero and prints one error annotation per violation."""
    root = build_hybrid_tree(tmp_path, {"ci.yml": with_job_condition(CLEAN_CI_YML, "    name: Lint (pre-commit)\n")})
    status = ans.main([str(root)])
    captured = capsys.readouterr()
    assert status == 1
    assert captured.out.count("::error::") == 1


def test_violations_accumulate_rather_than_exiting_on_the_first(tmp_path: pathlib.Path) -> None:
    """Three distinct problems produce three violations in one run."""
    workflow = with_job_condition(CLEAN_CI_YML, "    name: Lint (pre-commit)\n").replace(
        "    name: Tests (pytest)\n", "    name: Tests (pytest)\n    needs: [lint]\n", 1
    )
    root = build_hybrid_tree(
        tmp_path,
        {"ci.yml": workflow, "broken.yml": NO_TRIGGER_YML},
        contexts=DEFAULT_CONTEXTS,
    )
    violations = ans.run_all(root)
    assert len(violations) == 3, violations


def test_the_wrong_argument_count_is_refused(tmp_path: pathlib.Path) -> None:
    """The one-argument invocation contract is enforced, in both directions."""
    assert ans.main([]) == 1
    assert ans.main([str(tmp_path), str(tmp_path)]) == 1
