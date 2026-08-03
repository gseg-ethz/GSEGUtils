#!/usr/bin/env python3
# .github/scripts/assert_no_skip.py
# Source: Phase 13 CONTEXT.md D-19/D-23 + 13-RESEARCH.md Probe 1 (CONFIRMED) — the anti-skip assertion
"""Report the pull-request edit shapes GitHub's own required-check gate fails OPEN on.

Invoked from ``integrity.yml`` as ``python3 .github/scripts/assert_no_skip.py
HYBRID_TREE_ROOT``. One positional argument, deliberately: the workflow step is a
single call and there is nothing to get wrong at the call site. Every violation is
accumulated and emitted as a GitHub ``::error::`` annotation before the single
exit, so one run reports the whole set rather than the first item. Exits non-zero
when any violation is found, and zero only when none is.

**What a hybrid tree is, and why the shape matters.** The argument names a
directory whose ``.github/rulesets/`` came from the **base checkout** -- the
protected branch's committed payloads -- and whose ``.github/workflows/`` came
from the **pull request's head**, fetched as data through the contents API. The
rules and the required contexts therefore come from a side the pull request
cannot reach, while the jobs under judgement come from the side it can. That
split is the entire mechanism: an edit to the pull request cannot change what
judges it.

**Why this exists at all.** GitHub's own required-status-check gate fails CLOSED
on a workflow-level filter, on a deleted workflow and on a renamed job -- all
three leave the context perpetually pending, which blocks the merge. It fails
**OPEN** on exactly two shapes, because a ``skipped`` conclusion counts among the
satisfying ones (13-RESEARCH.md R5):

  1. a job-level condition on a job whose declared name is a required context; and
  2. a dependency edge from such a job onto a job that itself carries one.

The build-time self-test cannot close that hole, because it runs inside the job
the attack skips. This program is run from the protected branch's checkout by a
job triggered on the base-repository pull-request event, whose check run attaches
to the pull request's **head** SHA and therefore satisfies a required context
while the runner sees the **base** commit. Measured, not inferred: 13-RESEARCH.md
``## Probe 1 (D-19)``.

**Rules 1 and 2 are INVOKED, not written.** They are
:func:`check_ci_config.check_job_level_conditions` and
:func:`check_ci_config.check_conditional_dependencies`, called with the hybrid
tree as their root. Reuse is the point, not a convenience: the branch gate and
the build-time self-test must agree on what a violation *is*, and two
implementations of one rule is precisely how they stop agreeing. Those functions
already read the required contexts from the tree's ruleset files and the jobs
from the tree's workflow files, which is exactly the base/head split this
mechanism needs.

**Rule 3 is the new one.** A required context that no job in the head's workflow
set declares is a violation. That is what deleting or gutting the producing
workflow in the head looks like from here -- a shape GitHub does block, but only
by leaving the pull request stuck, which is a worse signal than a named failure.

**Nothing here executes head content.** The head's workflow YAML is parsed as
data by this program, which comes from the protected branch. There is no
checkout of the head, no install, no ``run:`` of anything the pull request
supplied. See ``integrity.yml``'s header for the three conditions CI-12's
exception carries and which two of them are asserted statically.

Report on stdout, so the annotations land in the run log where a reviewer reads
them.
"""

import pathlib
import sys
from typing import Any

import check_ci_config
import ruleset_lib
import yaml  # available on ubuntu-latest runner by default

USAGE = "usage: assert_no_skip.py HYBRID_TREE_ROOT"

# Re-exported from the self-test rather than restated, so the two can never name
# different directories.
WORKFLOWS_DIR = check_ci_config.WORKFLOWS_DIR
RULESETS_DIR = check_ci_config.RULESETS_DIR

# GitHub accepts either extension for a workflow file, so both are globbed when
# building the head's producer set -- a workflow this program cannot see would
# make a legitimate context look unproduced, which is a false violation.
WORKFLOW_GLOBS: tuple[str, ...] = ("*.yml", "*.yaml")

# The extension the SHARED assertions glob (`check_ci_config.load_workflows`).
# A head workflow outside this reach is reported rather than trusted -- see
# :func:`check_shared_assertion_coverage`.
SHARED_ASSERTION_GLOB = "*.yml"


def head_workflow_paths(root: pathlib.Path) -> list[pathlib.Path]:
    """Return every head workflow file in the hybrid tree, sorted.

    Parameters
    ----------
    root
        The hybrid tree root.

    Returns
    -------
    list of pathlib.Path
        One path per workflow file, under either accepted extension.
    """
    workflows_dir = root / WORKFLOWS_DIR
    if not workflows_dir.is_dir():
        return []
    paths: list[pathlib.Path] = []
    for pattern in WORKFLOW_GLOBS:
        paths.extend(workflows_dir.glob(pattern))
    return sorted(paths)


def head_job_names(root: pathlib.Path) -> tuple[set[str], list[str], bool]:
    """Build the check-run names the head's workflow set can produce.

    A workflow that cannot be read is recorded as a violation rather than
    skipped, in the wording the apply preflight already uses: an unreadable
    workflow has not been checked, so it cannot be assumed to contribute
    nothing. The trigger block is resolved through :func:`ruleset_lib.trigger_block`,
    which looks for the quoted ``"on":`` key first and the bare boolean key
    second -- most workflows in these repositories use the bare form today, so a
    parser reading only the quoted key would treat every required-check workflow
    as job-less and pass a head that had gutted them.

    Job names come from :func:`ruleset_lib.job_names`, which falls back to the
    bare job id for a job declaring no ``name:`` -- the string GitHub actually
    reports as that job's check-run name.

    Parameters
    ----------
    root
        The hybrid tree root.

    Returns
    -------
    tuple
        The producible name set, the violations found while building it, and
        whether any file failed to parse at all -- which additionally makes the
        two shared assertions unrunnable, because
        ``check_ci_config.load_workflows`` calls ``yaml.safe_load`` unguarded and
        would raise a traceback instead of an annotation.
    """
    violations: list[str] = []
    unparseable = False
    workflows_dir = root / WORKFLOWS_DIR

    if not workflows_dir.is_dir():
        violations.append(
            f"the head workflow directory `{WORKFLOWS_DIR}` does not exist in the assembled tree — "
            f"nothing was fetched from the pull request's head, so no cross-reference is possible. "
            f"A failed fetch is a hard failure here, never an empty set that reads as no violations."
        )
        return set(), violations, unparseable

    paths = head_workflow_paths(root)
    if not paths:
        violations.append(
            f"the head workflow directory `{WORKFLOWS_DIR}` holds no workflow files — nothing was "
            f"fetched from the pull request's head, so no cross-reference is possible. An empty "
            f"directory is a failed fetch, not a clean result."
        )
        return set(), violations, unparseable

    names: set[str] = set()
    for path in paths:
        try:
            with path.open(encoding="utf-8") as handle:
                document = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError) as error:
            unparseable = True
            violations.append(
                f"{path.name}: could not be read as YAML — {error}. This workflow has not been "
                f"checked, so it cannot be assumed clean, and the two shared conditionality rules "
                f"cannot be run over a tree containing an unparseable workflow at all — so nothing "
                f"in this pull request has been checked."
            )
            continue
        if not isinstance(document, dict):
            violations.append(
                f"{path.name}: parsed to {type(document).__name__}, not a mapping — this workflow "
                f"has not been checked, so it cannot be assumed clean."
            )
            continue
        _, form = ruleset_lib.trigger_block(document)
        if form is None:
            violations.append(
                f"{path.name}: no resolvable trigger block under either the quoted or the bare "
                f"`on:` key — this workflow has not been checked, so it cannot be assumed clean, "
                f"and it must not be read as contributing no jobs."
            )
            continue
        names.update(ruleset_lib.job_names(document))
    return names, violations, unparseable


def check_shared_assertion_coverage(root: pathlib.Path) -> list[str]:
    """Flag a head workflow the two shared conditionality assertions would not read.

    ``check_ci_config.load_workflows`` globs :data:`SHARED_ASSERTION_GLOB` only,
    while GitHub accepts either extension. So a required-check workflow renamed
    to ``.yaml`` in the head would still produce its context -- satisfying rule 3
    -- while sitting outside rules 1 and 2's reach entirely, which is a
    one-rename bypass of the whole mechanism. An unchecked workflow is not a
    clean one, so this is a violation rather than a note.

    Parameters
    ----------
    root
        The hybrid tree root.

    Returns
    -------
    list of str
        One entry per head workflow outside the shared assertions' glob.
    """
    covered = (
        {path.name for path in (root / WORKFLOWS_DIR).glob(SHARED_ASSERTION_GLOB)}
        if (root / WORKFLOWS_DIR).is_dir()
        else set()
    )
    return [
        f"{path.name}: the two shared conditionality assertions read `{SHARED_ASSERTION_GLOB}` only, "
        f"so this workflow is NOT covered by them — a required-check workflow renamed to this "
        f"extension in the head would still report its context while escaping both rules. Rename it "
        f"to `.yml` in this pull request, or widen the glob in the protected branch's copy of "
        f"`check_ci_config.load_workflows` first."
        for path in head_workflow_paths(root)
        if path.name not in covered
    ]


def payloads_requiring(root: pathlib.Path, context: str) -> list[str]:
    """Return the committed payload filenames that require `context`.

    Named rather than counted so the violation carries its payload-side origin:
    a reviewer reading the failure sees which committed file on the protected
    branch is asking for the context, and therefore which one would have to
    change if the context is genuinely gone for good.

    Parameters
    ----------
    root
        The hybrid tree root.
    context
        A required status-check context string.

    Returns
    -------
    list of str
        Payload filenames, sorted.
    """
    return sorted(
        path.name
        for path in (root / RULESETS_DIR).glob("*.json")
        if context in ruleset_lib.required_contexts(_load_json(path))
    )


def check_unproduced_contexts(root: pathlib.Path, producible: set[str]) -> list[str]:
    """Rule 3 -- flag a required context no job in the head's workflow set declares.

    Parameters
    ----------
    root
        The hybrid tree root.
    producible
        The check-run names the head's workflow set can produce.

    Returns
    -------
    list of str
        One entry per required context nothing in the head produces.
    """
    violations: list[str] = []
    for context in sorted(check_ci_config.committed_required_contexts(root)):
        if context in producible:
            continue
        origin = ", ".join(f"`{name}`" for name in payloads_requiring(root, context)) or "<no payload>"
        violations.append(
            f"`{context}` is required by the protected branch's committed payload(s) {origin}, but no "
            f"job in this pull request's workflow set declares that name — nothing here would report "
            f"it. Deleting or gutting the producing workflow in the head looks exactly like this. The "
            f"payload comes from the base checkout and this pull request cannot edit it, so the fix "
            f"is to restore the producing job."
        )
    return violations


def run_all(root: pathlib.Path) -> list[str]:
    """Run all three rules against a hybrid tree and return the accumulated violations.

    Parameters
    ----------
    root
        The hybrid tree root: base-side ``.github/rulesets/``, head-side
        ``.github/workflows/``.

    **Why a read failure suppresses rule 3 rather than adding to it.** When a head
    workflow could not be read, the producible-name set is incomplete by
    definition, so every context that file might have produced would ALSO be
    reported as unproduced. That is one cause wearing two violations, and the
    second one names the wrong problem -- it would tell a reviewer to restore a
    job that may well already be there. The read failure is reported and rule 3
    is held back; the run still exits non-zero, so nothing is let through.

    **Why an unparseable file suppresses rules 1 and 2 as well.**
    ``check_ci_config.load_workflows`` calls ``yaml.safe_load`` without a guard,
    so a head workflow that is not valid YAML makes both shared assertions raise.
    The exception would still fail the job -- fail-closed, which is the right
    direction -- but as a traceback rather than a named annotation. The read
    violation carries that reason in its own message instead, so the count stays
    one violation per cause.

    Returns
    -------
    list of str
        Every violation found, in rule order.
    """
    producible, read_violations, unparseable = head_job_names(root)
    coverage = check_shared_assertion_coverage(root)

    if unparseable:
        return [*read_violations, *coverage]

    conditionality = [
        # Rules 1 and 2, INVOKED from the self-test. Never reimplemented here --
        # see the module docstring for why that is the point rather than a
        # shortcut.
        *check_ci_config.check_job_level_conditions(root),
        *check_ci_config.check_conditional_dependencies(root),
    ]
    if read_violations:
        return [*conditionality, *read_violations, *coverage]
    return [*conditionality, *coverage, *check_unproduced_contexts(root, producible)]


def main(argv: list[str]) -> int:
    """Assert the head's workflow set against the base's committed payloads.

    Parameters
    ----------
    argv
        Positional arguments, excluding the program name.

    Returns
    -------
    int
        0 when no violation was found, 1 otherwise.
    """
    if len(argv) != 1:
        print(f"::error::{USAGE}")
        return 1

    root = pathlib.Path(argv[0])
    if not root.is_dir():
        print(
            f"::error::`{root}` is not a directory — the hybrid tree was never assembled, so nothing "
            f"was checked. This is a hard failure, not an empty result."
        )
        return 1

    violations = run_all(root)
    if violations:
        for violation in violations:
            print(f"::error::{violation}")
        return 1
    return 0


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    """Parse a committed ruleset payload, returning an empty mapping when it is not one."""
    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
