#!/usr/bin/env python3
# .github/scripts/test_classify_changes.py
# Source: Phase 16 plan 16-11 — the standing regression test for review finding CR-02
"""Behavioural tests for the ``classify-changes`` composite action's shell body.

These tests run the **real** step body. It is extracted from
``.github/actions/classify-changes/action.yml`` with a YAML parse and executed under
``bash`` with the ``gh`` call replaced by a stub, rather than being restated here as a
Python reimplementation. A reimplementation would drift from the shipped shell and then
pass while the shipped shell failed, which is the exact failure shape this file exists
to prevent: nothing automated covered ``.github/actions/`` before it, so the allowlist's
matching rule had no standing enforcement at all and shipped with a required-gate bypass
in it.

The bypass, for the record. Membership used to be containment against a space-padded
string::

    case " $ALLOW " in
      *" $changed "*) ;;

which is not set membership. A single changed path whose own *name* contained a space --
``CHANGELOG.md .release-please-manifest.json`` -- padded to a literal substring of
``" $ALLOW "`` and was therefore accepted, so one pull request adding one such file
emitted ``release-artifacts-only=true`` and greened every required status check having
executed nothing. ``test_a_single_path_containing_a_space_is_rejected`` pins that path to
``false`` and fails against the pre-fix shell.

Run with::

    python -m pytest .github/scripts/test_classify_changes.py -q

Note that ruff's ``per-file-ignores`` relax docstring rules under ``tests/**`` only, which
does not cover ``.github/scripts/``, so every test here carries a numpy-convention
docstring by design rather than by accident.
"""

import os
import pathlib
import re
import subprocess

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ACTION_PATH = REPO_ROOT / ".github/actions/classify-changes/action.yml"

#: The two paths a generated release pull request actually touches. Kept as a literal
#: rather than parsed out of ``ALLOW`` so that a test asserting "these are allowed"
#: cannot be satisfied by an allowlist that has silently changed underneath it --
#: assertion A4 in ``check_ci_config.py`` separately reconciles ``ALLOW`` against
#: release-please's own configuration.
RELEASE_ARTIFACTS = ("CHANGELOG.md", ".release-please-manifest.json")


def _step_body() -> str:
    """Return the composite's classify step as a runnable bash script.

    Returns
    -------
    str
        The step's ``run:`` body with GitHub's expression contexts replaced by
        environment-variable references the harness supplies.
    """
    document = yaml.safe_load(ACTION_PATH.read_text())
    (step,) = document["runs"]["steps"]
    body = step["run"]
    body = body.replace("${{ github.event_name }}", "${EVENT_NAME}")
    body = body.replace("${{ github.repository }}", "${REPO}")
    body = body.replace("${{ github.event.pull_request.number }}", "${PRNUM}")
    return re.sub(r"\$\{\{[^}]*\}\}", "", body)


def classify(tmp_path: pathlib.Path, changed: list[str], event_name: str = "pull_request") -> str:
    """Run the shipped step against a fixed changed-path list and return its verdict.

    Parameters
    ----------
    tmp_path
        Per-test temporary directory, used for the stub `gh`, the fixture file list and
        the emulated ``$GITHUB_OUTPUT``.
    changed
        The changed paths the stubbed changed-path API should return, one per line.
    event_name
        The value ``github.event_name`` resolves to for this run.

    Returns
    -------
    str
        ``'true'`` or ``'false'`` -- the value written to ``$GITHUB_OUTPUT`` -- or the
        empty string when the step emitted nothing at all.
    """
    script = tmp_path / "step.sh"
    script.write_text(_step_body())

    fixture = tmp_path / "files.txt"
    fixture.write_text("".join(f"{path}\n" for path in changed))

    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "gh"
    stub.write_text(f'#!/usr/bin/env bash\ncat "{fixture}"\nexit 0\n')
    stub.chmod(0o755)

    output = tmp_path / "github_output"
    output.write_text("")

    environment = dict(os.environ)
    environment.update(
        PATH=f"{bindir}{os.pathsep}{environment['PATH']}",
        EVENT_NAME=event_name,
        REPO="owner/repo",
        PRNUM="1",
        RUNNER_TEMP=str(tmp_path),
        RUNNER_NAME="pytest",
        GITHUB_OUTPUT=str(output),
    )
    subprocess.run(["bash", str(script)], env=environment, capture_output=True, check=False)

    emitted = re.findall(r"release-artifacts-only=(true|false)", output.read_text())
    return emitted[-1] if emitted else ""


@pytest.mark.parametrize("path", RELEASE_ARTIFACTS)
def test_each_release_artifact_alone_takes_the_fast_path(tmp_path: pathlib.Path, path: str) -> None:
    """Each allowlisted path, changed on its own, still engages the fast path.

    This is the true-positive half. Without it a fix that rejected everything would
    satisfy every other test in this file.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    path
        The single allowlisted path under test.
    """
    assert classify(tmp_path, [path]) == "true"


@pytest.mark.parametrize(
    "changed",
    [
        list(RELEASE_ARTIFACTS),
        list(reversed(RELEASE_ARTIFACTS)),
    ],
    ids=["changelog-first", "manifest-first"],
)
def test_the_verdict_does_not_depend_on_the_order_the_api_returns_files(
    tmp_path: pathlib.Path, changed: list[str]
) -> None:
    """Both allowlisted paths together take the fast path in either return order.

    The changed-path API makes no ordering guarantee, so a matching rule that was
    order-sensitive would engage the fast path intermittently.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    changed
        The changed-path list, in one of its two permutations.
    """
    assert classify(tmp_path, changed) == "true"


def test_a_single_path_containing_a_space_is_rejected(tmp_path: pathlib.Path) -> None:
    """CR-02 -- one file whose NAME is the two allowlist entries joined by a space.

    Under the padded-containment rule this shipped with, this single path was ALLOWED:
    its padded form is literally a substring of the padded allowlist. A pull request
    adding only this file therefore greened ``Lint``, ``Tests`` and ``Docs`` with every
    guarded step skipped and honest ``success`` conclusions, on a branch requiring zero
    approving reviews. It must be rejected: it is one path, and it is not an allowlisted
    one.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    """
    assert classify(tmp_path, [" ".join(RELEASE_ARTIFACTS)]) == "false"


@pytest.mark.parametrize(
    "path",
    [
        " ".join(reversed(RELEASE_ARTIFACTS)),
        "\t".join(RELEASE_ARTIFACTS),
        " CHANGELOG.md",
        "CHANGELOG.md ",
        "my notes.md",
    ],
    ids=["reversed-join", "tab-join", "leading-space", "trailing-space", "ordinary-space"],
)
def test_no_whitespace_bearing_path_can_satisfy_the_allowlist(tmp_path: pathlib.Path, path: str) -> None:
    """No path containing whitespace of any kind reaches a true verdict.

    The allowlist is a space-delimited string, so it cannot represent a path containing
    whitespace at all. Rejecting the whole class by name is cheaper than reasoning about
    which members of it happen to be exploitable under whichever matching rule is in
    force.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    path
        A whitespace-bearing changed path.
    """
    assert classify(tmp_path, [path]) == "false"


@pytest.mark.parametrize(
    "path",
    [
        "CHANGELOG.md.bak",
        "docs/CHANGELOG.md",
        "HANGELOG.md",
        "*",
        "CHANGELOG.?d",
        "src/evil.py",
        ".github/workflows/ci.yml",
    ],
)
def test_a_path_that_is_not_an_exact_allowlist_entry_runs_everything(tmp_path: pathlib.Path, path: str) -> None:
    """Superstrings, substrings, path prefixes and glob metacharacters all run everything.

    Membership is exact and entry-by-entry. Nothing that merely resembles an allowlist
    entry, and no pattern that could match one, may satisfy it.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    path
        A changed path that is not an allowlist entry.
    """
    assert classify(tmp_path, [path]) == "false"


def test_one_unrecognised_path_disqualifies_an_otherwise_allowlisted_set(tmp_path: pathlib.Path) -> None:
    """A source file alongside the changelog runs everything.

    The fast path is all-or-nothing by design: it engages only when *every* changed path
    is a release artifact.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    """
    assert classify(tmp_path, ["CHANGELOG.md", "src/evil.py"]) == "false"


def test_a_non_pull_request_event_runs_everything(tmp_path: pathlib.Path) -> None:
    """The push and schedule triggers have no pull request to classify.

    A fail-safe branch, pinned here so a future edit to the matching rule cannot remove it
    as collateral.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    """
    assert classify(tmp_path, ["CHANGELOG.md"], event_name="push") == "false"


def test_an_empty_changed_path_list_runs_everything(tmp_path: pathlib.Path) -> None:
    """An empty list is anomalous rather than a licence to skip.

    A fail-safe branch, pinned for the same reason as the one above.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    """
    assert classify(tmp_path, []) == "false"
