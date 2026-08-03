#!/usr/bin/env python3
# .github/scripts/test_check_ci_config.py
# Source: Phase 13 plan 13-03 Task 2 — unit tests pinning CI-12 assertions A1 through A7
"""Unit tests for the CI-12 self-test's seven assertions.

Every assertion helper takes a root path, so each test writes a synthetic tree of
workflow, ruleset, composite-action and release-please files into a ``tmp_path``
fixture and asserts against that tree rather than against the repository. Three
tests deliberately run against the real repository root instead, because "passes
on today's tree" is itself a property worth pinning: A1, A2 and A5 all have a
current-tree true-negative that a synthetic fixture cannot demonstrate.

The A1 pair is the one that matters most. ``test_a1_passes_on_the_real_ci_yml``
and ``test_a1_fails_on_a_paths_filter_under_a_pull_request_trigger`` exist
together so that the day-one false positive described in
``check_ci_config``'s module docstring is demonstrably resolved by *narrowing*
the rule rather than by muting the check.

Run with::

    python -m pytest .github/scripts/test_check_ci_config.py -q

Note that ruff's ``per-file-ignores`` relax docstring rules under ``tests/**``
only, which does not cover ``.github/scripts/``, so every test here carries a
numpy-convention docstring by design rather than by accident.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import check_ci_config as cc  # noqa: E402  — deliberately after the sys.path insertion above

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

DEFAULT_CONTEXTS = ("Lint (pre-commit)", "Tests (pytest)")
DEFAULT_ALLOWLIST = ("CHANGELOG.md", ".release-please-manifest.json", "docs/source/conf.py")

# A minimal workflow producing both default required contexts, with a clean
# trigger block. Individual tests override one aspect of it at a time.
CLEAN_CI_YML = """\
name: CI

on:
  pull_request:
  push:
    branches: [main, develop/gsd]

jobs:
  lint:
    name: Lint (pre-commit)
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - run: echo lint
  tests:
    name: Tests (pytest)
    runs-on: ubuntu-latest
    steps:
      - run: echo test
"""


def build_tree(
    root: pathlib.Path,
    workflows: dict[str, str],
    *,
    contexts: tuple[str, ...] = DEFAULT_CONTEXTS,
    allowlist: tuple[str, ...] = DEFAULT_ALLOWLIST,
    changelog_path: str = "CHANGELOG.md",
    extra_files: tuple[str, ...] = ("docs/source/conf.py",),
) -> pathlib.Path:
    """Write a synthetic repository tree the assertion helpers can run against.

    Parameters
    ----------
    root
        Directory to populate; normally a ``tmp_path`` fixture.
    workflows
        Mapping of workflow filename to file text.
    contexts
        Required status-check contexts written into the committed ruleset.
    allowlist
        Paths written into the composite action's allowlist variable.
    changelog_path
        The release-please changelog path.
    extra_files
        The release-please extra files.

    Returns
    -------
    pathlib.Path
        The populated root, for chaining.
    """
    workflows_dir = root / cc.WORKFLOWS_DIR
    workflows_dir.mkdir(parents=True, exist_ok=True)
    for name, text in workflows.items():
        (workflows_dir / name).write_text(text)

    rulesets_dir = root / cc.RULESETS_DIR
    rulesets_dir.mkdir(parents=True, exist_ok=True)
    (rulesets_dir / "main.json").write_text(
        json.dumps(
            {
                "name": "protect-main",
                "rules": [
                    {
                        "type": "required_status_checks",
                        "parameters": {
                            "required_status_checks": [{"context": c, "integration_id": None} for c in contexts]
                        },
                    }
                ],
            }
        )
    )

    action_path = root / cc.COMPOSITE_ACTION_PATH
    action_path.parent.mkdir(parents=True, exist_ok=True)
    action_path.write_text(
        f"runs:\n  using: composite\n  steps:\n    - run: |\n        ALLOW='{' '.join(allowlist)}'\n"
    )

    (root / cc.RELEASE_PLEASE_CONFIG_PATH).write_text(
        json.dumps({"packages": {".": {"changelog-path": changelog_path, "extra-files": list(extra_files)}}})
    )
    return root


def with_trigger(trigger_block: str) -> str:
    """Return the clean CI workflow with its trigger block replaced.

    Parameters
    ----------
    trigger_block
        Replacement text for everything between the name line and the jobs key.

    Returns
    -------
    str
        The rewritten workflow text.
    """
    _, jobs = CLEAN_CI_YML.split("jobs:", 1)
    return f"name: CI\n\n{trigger_block}\njobs:{jobs}"


# --------------------------------------------------------------------------
# A1 — filter keys under the pull-request-side events
# --------------------------------------------------------------------------


def test_a1_passes_on_the_real_ci_yml() -> None:
    """A1 is clean on today's real workflow, whose push trigger already carries a branch filter."""
    assert cc.check_trigger_filters(REPO_ROOT) == []


def test_a1_fails_on_a_paths_filter_under_a_pull_request_trigger(tmp_path: pathlib.Path) -> None:
    """A paths filter under the pull-request trigger is flagged, naming file, key, event and contexts."""
    root = build_tree(tmp_path, {"ci.yml": with_trigger("on:\n  pull_request:\n    paths: ['src/**']\n")})
    violations = cc.check_trigger_filters(root)
    assert len(violations) == 1, violations
    message = violations[0]
    assert "ci.yml" in message
    assert "paths" in message
    assert "pull_request" in message
    assert "Lint (pre-commit)" in message and "Tests (pytest)" in message


def test_a1_ignores_a_branch_filter_under_the_push_trigger(tmp_path: pathlib.Path) -> None:
    """A push-side branch filter is not inspected — it cannot leave a pull-request check pending."""
    root = build_tree(tmp_path, {"ci.yml": with_trigger("on:\n  pull_request:\n  push:\n    branches: [main]\n")})
    assert cc.check_trigger_filters(root) == []
    assert "push" not in cc.A1_INSPECTED_EVENTS


def test_a1_skips_a_workflow_that_produces_no_required_context(tmp_path: pathlib.Path) -> None:
    """A workflow whose jobs produce no required context is not inspected at all."""
    unrelated = "name: Canary\n\non:\n  pull_request:\n    paths: ['src/**']\n\njobs:\n  canary:\n    steps: []\n"
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, "canary.yml": unrelated})
    assert cc.check_trigger_filters(root) == []


def test_a1_skips_a_reviewed_exception_and_every_entry_states_a_reason(tmp_path: pathlib.Path) -> None:
    """A file in the reviewed-exceptions table is skipped, and each table entry carries a stated reason."""
    filtered = with_trigger("on:\n  pull_request:\n    branches: [main]\n")
    root = build_tree(tmp_path, {"ci.yml": filtered})
    assert cc.check_trigger_filters(root) != []
    assert cc.check_trigger_filters(root, exceptions={"ci.yml": "reviewed for this test"}) == []
    for filename, reason in cc.A1_REVIEWED_EXCEPTIONS.items():
        assert len(reason) > 40, f"{filename} exception must state why, not merely exist"


def test_a1_reports_a_workflow_whose_trigger_block_cannot_be_resolved(tmp_path: pathlib.Path) -> None:
    """A required-check workflow with no resolvable trigger block is a violation, not a pass."""
    root = build_tree(tmp_path, {"ci.yml": "name: CI\n\njobs:" + CLEAN_CI_YML.split("jobs:", 1)[1]})
    violations = cc.check_trigger_filters(root)
    assert len(violations) == 1, violations
    assert "trigger" in violations[0].lower()


def test_a1_tolerates_a_bare_list_trigger_block(tmp_path: pathlib.Path) -> None:
    """A trigger block written as a bare list carries no filters and must not raise or be flagged."""
    root = build_tree(tmp_path, {"ci.yml": with_trigger("on: [push, pull_request]\n")})
    assert cc.check_trigger_filters(root) == []


def test_a1_tolerates_a_workflow_document_with_no_jobs_key(tmp_path: pathlib.Path) -> None:
    """A workflow document with no jobs key is handled without raising and without being flagged."""
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, "stub.yml": "name: Stub\n\non:\n  pull_request:\n"})
    assert cc.check_trigger_filters(root) == []


# --------------------------------------------------------------------------
# A2 — the base-repo-context pull-request trigger
# --------------------------------------------------------------------------


def test_a2_passes_on_the_real_workflows_directory() -> None:
    """The one excepted workflow still meets both asserted conditions on today's tree.

    This is not a vacuous pass: ``integrity.yml`` carries the banned trigger, so
    this call exercises the exception AND its two condition assertions against
    the real file rather than against a fixture.
    """
    assert cc.check_base_repo_context_trigger(REPO_ROOT) == []


def test_a2_fails_when_the_token_appears_only_inside_a_comment(tmp_path: pathlib.Path) -> None:
    """The scan is over raw text, so an occurrence in a comment is caught."""
    commented = f"name: CI\n\n# never use {cc.BASE_REPO_CONTEXT_TRIGGER} here\non:\n  pull_request:\n\njobs: {{}}\n"
    root = build_tree(tmp_path, {"ci.yml": commented})
    violations = cc.check_base_repo_context_trigger(root)
    assert len(violations) == 1, violations
    assert "ci.yml" in violations[0]


# --------------------------------------------------------------------------
# A2's one reviewed exception, and the two conditions asserted behind it
# --------------------------------------------------------------------------

# The real excepted workflow, read from the tree it ships in, so these fixtures
# drift with it rather than describing a file that no longer exists.
EXCEPTED_WORKFLOW = "integrity.yml"


def excepted_workflow_text() -> str:
    """Return the real excepted workflow's text.

    Returns
    -------
    str
        The contents of the repository's own excepted workflow file.
    """
    return (REPO_ROOT / cc.WORKFLOWS_DIR / EXCEPTED_WORKFLOW).read_text()


def test_a2_honours_the_exception_for_the_real_conforming_workflow(tmp_path: pathlib.Path) -> None:
    """The real excepted workflow passes A2: the trigger is allowed and both conditions hold."""
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, EXCEPTED_WORKFLOW: excepted_workflow_text()})
    assert cc.check_base_repo_context_trigger(root) == []


def test_a2_fails_when_the_excepted_workflow_grants_a_write_scope(tmp_path: pathlib.Path) -> None:
    """Widening the excepted workflow's permissions block fails condition (iii)."""
    widened = excepted_workflow_text().replace(
        "permissions:\n  contents: read\n", "permissions:\n  contents: read\n  pull-requests: write\n", 1
    )
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, EXCEPTED_WORKFLOW: widened})
    violations = cc.check_base_repo_context_trigger(root)
    assert len(violations) == 1, violations
    assert EXCEPTED_WORKFLOW in violations[0]
    assert "permissions" in violations[0]


def test_a2_fails_when_a_job_in_the_excepted_workflow_widens_the_permissions(tmp_path: pathlib.Path) -> None:
    """A job-level block REPLACES the workflow-level one, so it is asserted too.

    Without this, the workflow-level block could read clean while the job that
    actually runs held a write scope -- the whole exception defeated by one
    correctly-indented key.
    """
    widened = excepted_workflow_text().replace(
        "    timeout-minutes: 10\n",
        "    timeout-minutes: 10\n    permissions:\n      contents: write\n",
        1,
    )
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, EXCEPTED_WORKFLOW: widened})
    violations = cc.check_base_repo_context_trigger(root)
    assert len(violations) == 1, violations
    assert "job `integrity`" in violations[0]


def test_a2_fails_when_the_excepted_workflows_checkout_carries_a_ref_input(tmp_path: pathlib.Path) -> None:
    """A checkout with an explicit reference fails condition (i) -- base-ref checkout only."""
    with_ref = excepted_workflow_text().replace(
        "        uses: actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd  # v5.0.1\n",
        "        uses: actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd  # v5.0.1\n"
        "        with:\n          ref: ${{ github.event.pull_request.head.sha }}\n",
        1,
    )
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, EXCEPTED_WORKFLOW: with_ref})
    violations = cc.check_base_repo_context_trigger(root)
    assert len(violations) == 1, violations
    assert EXCEPTED_WORKFLOW in violations[0]
    assert cc.CHECKOUT_REF_INPUT in violations[0]


def test_a2_still_fails_on_the_banned_trigger_in_a_different_file(tmp_path: pathlib.Path) -> None:
    """The exception is keyed on a file path, not a global mute.

    The same trigger in any other workflow still fails the build, which is what
    distinguishes a named exception from having turned the assertion off.
    """
    offender = f"name: Other\n\non:\n  {cc.BASE_REPO_CONTEXT_TRIGGER}:\n    branches: [main]\n\njobs: {{}}\n"
    root = build_tree(
        tmp_path,
        {"ci.yml": CLEAN_CI_YML, EXCEPTED_WORKFLOW: excepted_workflow_text(), "other.yml": offender},
    )
    violations = cc.check_base_repo_context_trigger(root)
    assert len(violations) == 1, violations
    assert "other.yml" in violations[0]
    assert EXCEPTED_WORKFLOW not in violations[0]


def test_a2_exception_table_is_injectable_in_both_directions(tmp_path: pathlib.Path) -> None:
    """An empty table restores the blanket ban; a table naming a file excepts only that file."""
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, EXCEPTED_WORKFLOW: excepted_workflow_text()})
    assert cc.check_base_repo_context_trigger(root, exceptions={}) != []
    assert cc.check_base_repo_context_trigger(root, exceptions={EXCEPTED_WORKFLOW: "reviewed here"}) == []


def test_every_a2_exception_entry_states_its_reason_and_names_its_three_conditions() -> None:
    """Each entry says WHY, and says which conditions are asserted and which reviewed."""
    assert cc.A2_REVIEWED_EXCEPTIONS, "the table must never be silently emptied"
    for filename, reason in cc.A2_REVIEWED_EXCEPTIONS.items():
        assert len(reason) > 40, f"{filename} exception must state why, not merely exist"
        assert reason.count("ASSERTED") == 2, f"{filename} must name exactly the two asserted conditions"
        assert "REVIEWED" in reason, f"{filename} must say which condition is reviewed rather than enforced"


# --------------------------------------------------------------------------
# A3 — the pinned workflow name
# --------------------------------------------------------------------------


def test_a3_passes_when_the_name_matches_the_pinned_literal(tmp_path: pathlib.Path) -> None:
    """The pinned literal matches today's declared workflow name byte for byte."""
    assert cc.check_workflow_name_pin(build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML})) == []
    assert cc.check_workflow_name_pin(REPO_ROOT) == []


def test_a3_fails_on_a_case_changed_name(tmp_path: pathlib.Path) -> None:
    """The comparison does not case-fold, so a case change is drift."""
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML.replace("name: CI", "name: Ci", 1)})
    violations = cc.check_workflow_name_pin(root)
    assert violations and "Ci" in violations[0], violations


def test_a3_fails_on_a_whitespace_padded_name(tmp_path: pathlib.Path) -> None:
    """The comparison does not strip, so a padded name is drift."""
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML.replace("name: CI", 'name: "CI "', 1)})
    assert cc.check_workflow_name_pin(root) != []


def test_a3_fails_when_a_workflow_run_trigger_names_an_undeclared_workflow(tmp_path: pathlib.Path) -> None:
    """A workflow-run trigger listing a name no workflow declares is a silent-breakage risk."""
    consumer = "name: Release\n\non:\n  workflow_run:\n    workflows: [Continuous Integration]\n\njobs: {}\n"
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, "release-please.yml": consumer})
    violations = cc.check_workflow_name_pin(root)
    assert violations and "Continuous Integration" in violations[0], violations


# --------------------------------------------------------------------------
# A4 — allowlist reconciliation against release-please's own configuration
# --------------------------------------------------------------------------


def test_a4_passes_when_the_allowlist_set_equals_the_release_configuration(tmp_path: pathlib.Path) -> None:
    """Changelog path plus extra files plus the manifest path is exactly the allowlist."""
    assert cc.check_fast_path_allowlist(build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML})) == []
    assert cc.check_fast_path_allowlist(REPO_ROOT) == []


def test_a4_is_decided_by_set_equality_not_by_written_order(tmp_path: pathlib.Path) -> None:
    """Reordering the allowlist does not change the verdict."""
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML}, allowlist=tuple(reversed(DEFAULT_ALLOWLIST)))
    assert cc.check_fast_path_allowlist(root) == []


def test_a4_fails_when_a_fourth_extra_file_is_added(tmp_path: pathlib.Path) -> None:
    """A fourth extra file fails the build with both sets printed and a named remedy."""
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML}, extra_files=("docs/source/conf.py", "docs/source/api.rst"))
    violations = cc.check_fast_path_allowlist(root)
    assert len(violations) == 1, violations
    assert "docs/source/api.rst" in violations[0]
    assert "fast path" in violations[0]


# --------------------------------------------------------------------------
# A5 / A6 — job-level conditions on, and reachable from, a required context
# --------------------------------------------------------------------------


def test_a5_passes_on_the_real_ci_yml() -> None:
    """No job producing a required context carries a job-level condition today."""
    assert cc.check_job_level_conditions(REPO_ROOT) == []


def test_a5_fails_when_a_required_context_job_carries_a_job_level_condition(tmp_path: pathlib.Path) -> None:
    """A skipped job satisfies its required context, so a job-level condition on one is a violation."""
    conditioned = CLEAN_CI_YML.replace(
        "    name: Lint (pre-commit)\n", "    name: Lint (pre-commit)\n    if: false\n", 1
    )
    root = build_tree(tmp_path, {"ci.yml": conditioned})
    violations = cc.check_job_level_conditions(root)
    assert len(violations) == 1, violations
    assert "Lint (pre-commit)" in violations[0]


def test_a6_fails_when_a_required_context_job_depends_on_a_conditional_job(tmp_path: pathlib.Path) -> None:
    """A skipped dependency skips the dependent, which also counts as satisfying."""
    workflow = CLEAN_CI_YML.replace(
        "    name: Lint (pre-commit)\n", "    name: Lint (pre-commit)\n    if: false\n", 1
    ).replace("    name: Tests (pytest)\n", "    name: Tests (pytest)\n    needs: [lint]\n", 1)
    root = build_tree(tmp_path, {"ci.yml": workflow})
    violations = cc.check_conditional_dependencies(root)
    assert len(violations) == 1, violations
    assert "Tests (pytest)" in violations[0] and "lint" in violations[0]


def test_a6_passes_when_the_dependency_carries_no_condition(tmp_path: pathlib.Path) -> None:
    """Depending on an unconditional job is fine and must not be flagged."""
    workflow = CLEAN_CI_YML.replace("    name: Tests (pytest)\n", "    name: Tests (pytest)\n    needs: [lint]\n", 1)
    assert cc.check_conditional_dependencies(build_tree(tmp_path, {"ci.yml": workflow})) == []


# --------------------------------------------------------------------------
# A7 — least privilege on the lint job
# --------------------------------------------------------------------------


def test_a7_fails_when_the_lint_job_grants_a_write_scope(tmp_path: pathlib.Path) -> None:
    """Any write scope in the lint job's permissions block fails the build."""
    workflow = CLEAN_CI_YML.replace("      contents: read\n", "      contents: write\n      pull-requests: write\n", 1)
    root = build_tree(tmp_path, {"ci.yml": workflow})
    violations = cc.check_lint_least_privilege(root, pending={})
    assert len(violations) == 2, violations
    assert any("contents" in v for v in violations) and any("pull-requests" in v for v in violations)


def test_a7_passes_when_the_lint_job_is_narrowed_to_read(tmp_path: pathlib.Path) -> None:
    """A read-only permissions block on the lint job is clean."""
    assert cc.check_lint_least_privilege(build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML}), pending={}) == []


def test_a7_fails_when_the_lint_job_declares_no_permissions_block(tmp_path: pathlib.Path) -> None:
    """An absent permissions block inherits the repository default and is a violation."""
    workflow = CLEAN_CI_YML.replace("    permissions:\n      contents: read\n", "", 1)
    violations = cc.check_lint_least_privilege(build_tree(tmp_path, {"ci.yml": workflow}), pending={})
    assert len(violations) == 1, violations
    assert "permissions" in violations[0]


def test_a7_pending_register_suppresses_a_scheduled_narrowing_and_names_its_removal(tmp_path: pathlib.Path) -> None:
    """A pending entry suppresses A7 for one context only, and every entry names the plan that removes it."""
    workflow = CLEAN_CI_YML.replace("      contents: read\n", "      contents: write\n", 1)
    root = build_tree(tmp_path, {"ci.yml": workflow})
    assert cc.check_lint_least_privilege(root, pending={cc.LINT_CONTEXT: "scheduled"}) == []
    assert cc.check_lint_least_privilege(root, pending={"Some Other Context": "scheduled"}) != []
    for context, reason in cc.A7_PENDING_NARROWINGS.items():
        assert "CI-16" in reason, context
        assert "13-04" in reason and "13-07" in reason, context


# --------------------------------------------------------------------------
# Driver behaviour
# --------------------------------------------------------------------------


def test_violations_accumulate_rather_than_exiting_on_the_first(tmp_path: pathlib.Path) -> None:
    """A file with three problems produces three annotations and one exit."""
    broken = with_trigger("on:\n  pull_request:\n    paths: ['src/**']\n    branches: [main]\n").replace(
        "    name: Lint (pre-commit)\n", "    name: Lint (pre-commit)\n    if: false\n", 1
    )
    root = build_tree(tmp_path, {"ci.yml": broken})
    violations, _ = cc.run_all(root, pending={})
    assert len(violations) >= 3, violations


def test_run_all_is_clean_on_the_real_repository_and_counts_its_workflows() -> None:
    """The self-test passes on the tree it ships in, and reports how many workflows it inspected."""
    violations, inspected = cc.run_all(REPO_ROOT)
    assert violations == [], violations
    assert inspected == len(sorted((REPO_ROOT / cc.WORKFLOWS_DIR).glob("*.yml")))
