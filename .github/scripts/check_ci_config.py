#!/usr/bin/env python3
# .github/scripts/check_ci_config.py
# Source: Phase 13 CONTEXT.md D-05/D-08/D-11 + 13-RESEARCH.md R5/R6 — the CI-12 self-test
"""Fail the build on the CI-configuration regressions that would break the gate.

Invoked from the Lint (pre-commit) CI job as ``python
.github/scripts/check_ci_config.py``. It takes NO arguments, accumulates every
violation, emits each as a GitHub ``::error::`` annotation and exits 1 once;
exits 0 and prints one trailing OK line when the tree is clean.

Seven assertions, each named by the identifier later plans and the adversarial
suite refer to:

  A1  No filter key under a pull-request-side trigger on a workflow producing a
      required status-check context.
  A2  The base-repo-context pull-request trigger appears nowhere under the
      workflows directory, except in the ONE reviewed file named in
      ``A2_REVIEWED_EXCEPTIONS`` -- and there only while the two statically
      checkable conditions of that exception still hold.
  A3  The CI workflow's declared name still equals the pinned literal, and every
      workflow-run trigger names a workflow some file actually declares.
  A4  The fast-path allowlist still set-equals release-please's own changelog
      path, extra files and manifest path.
  A5  No job whose declared name is a required context carries a job-level
      condition.
  A6  No such job depends on a job that itself carries a job-level condition.
  A7  The lint job declares a permissions block granting no write scope.

**Why A1 inspects the pull-request-side events and NOT push.** A required status
check comes from the event that produced it. A workflow skipped by a filter never
reports, so the pull request waits on a check that will never arrive -- but a
*push*-side branch filter cannot strand a *pull-request* check, because the
pull-request run is a different run with a different trigger. Both repos'
``ci.yml`` already carries ``branches:`` under ``push:``, deliberately and
correctly, so including push in A1's inspected set would fail this assertion on
day one against a configuration nobody wants changed. That is a false positive,
and the resolution is to narrow the rule with the reason written down -- here,
and pinned in both directions by ``test_check_ci_config.py`` -- rather than to
mute the check. Stating the converse plainly for the same reason: a schedule is
not a filter and a dispatch cannot suppress a pull-request check, so neither
event is ever inspected either.

**What this script does NOT defend against, stated so it is not assumed.** A5
and A6 make a job-level condition on a required context a build failure. They do
not make such a condition un-suppressible: this script runs from inside the lint
job, from the pull request's own checkout, so a head-ref edit reaches it -- an
``if:`` on ``lint`` skips the very job that would have caught it, and a skipped
conclusion satisfies the gate.

What closes that circularity is ``.github/workflows/integrity.yml`` (Phase 13
D-19 / D-23, DESIGN-DECISIONS entry 46): a required-context job triggered on the
base-repository pull-request event, whose definition and checker both come from
the protected branch while its check run attaches to the pull request's head sha.
It calls :func:`check_job_level_conditions` and
:func:`check_conditional_dependencies` -- these functions, not a second copy of
them -- against a tree whose rulesets come from the base checkout and whose
workflows come from the head. A5 and A6 remain the fast build-time signal; the
un-skippable evaluation of the same two rules lives there.

**Superseded, and named so a reader does not act on it:** this paragraph
previously credited the ruleset ``workflows`` rule (Phase 13 D-15) with pinning
``ci.yml``'s definition to the protected branch. That rule requires GitHub
Enterprise Cloud and is org/enterprise-scoped, so it is unavailable to these
repositories at any payload shape -- proved empirically and recorded as
DESIGN-DECISIONS entry 45. Entry 46 replaces it.

Assertion helpers each take a root path so they are testable against synthetic
trees; the module-level driver below calls them against this repository. The
driver executes at import, matching ``check_publish_gate.py``'s straight-line
shape and keeping the invocation contract argument-free -- see plan 13-03's
deferred-review note for the consequence when the tree is dirty.
"""

import json
import pathlib
import re
import sys
from typing import Any

import ruleset_lib
import yaml  # available on ubuntu-latest runner by default

# Resolved from this file rather than from the process working directory, so the
# self-test always inspects the repository it ships in. The invocation contract
# is unchanged: still no arguments.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

WORKFLOWS_DIR = ".github/workflows"
RULESETS_DIR = ".github/rulesets"
COMPOSITE_ACTION_PATH = ".github/actions/classify-changes/action.yml"
SELF_PATH = ".github/scripts/check_ci_config.py"
RELEASE_PLEASE_CONFIG_PATH = "release-please-config.json"
# `release-type: simple` with no version file in either repo, so the manifest is
# the only member of the allowlist release-please does not name in its config.
RELEASE_PLEASE_MANIFEST_PATH = ".release-please-manifest.json"

# Sub-keys of a triggering event that suppress a run. Presence of any of these
# under an inspected event means some pushes/pull requests produce no run at all.
FILTER_KEYS: frozenset[str] = frozenset({"paths", "paths-ignore", "branches", "branches-ignore", "tags", "tags-ignore"})

# The events A1 inspects: the pull-request event and its base-repo-context
# sibling. See the module docstring for why `push`, `schedule` and
# `workflow_dispatch` are all deliberately absent.
A1_INSPECTED_EVENTS: tuple[str, ...] = ("pull_request", "pull_request_target")

# Deliberate, reviewed exceptions to A1, keyed by workflow filename. Each value
# states WHY, so removing one is a decision rather than an oversight.
A1_REVIEWED_EXCEPTIONS: dict[str, str] = {
    "gpu.yml": (
        "Phase 13 D-11 scopes `Tests (pytest, GPU)` to a required context on the protected branch "
        "ONLY, mirroring how D-06 scopes the docs context. A `branches:` filter under its "
        "pull-request trigger is therefore the intended scope rather than drift: the context is not "
        "required on any branch the filter excludes, so no check is left pending."
    ),
    "integrity.yml": (
        "Phase 13 D-23 scopes `Integrity (base-ref)` to a required context on the protected branch "
        "ONLY, exactly as D-11 scopes the GPU context and D-06 the docs one. A `branches:` filter "
        "under its trigger is therefore the intended scope rather than drift: the context is not "
        "required on any branch the filter excludes, so no check is left pending. The deferred "
        "`develop/gsd` extension is a recorded cost — a skip that reaches the integration branch is "
        "caught by the promotion pull request into `main`."
    ),
}

# The base-repo-context pull-request trigger. Never legitimate in either repo
# EXCEPT for the one file named in A2_REVIEWED_EXCEPTIONS below: it runs with the
# base repository's secrets against a fork's code, and one of these repos
# attaches a self-hosted lab runner.
BASE_REPO_CONTEXT_TRIGGER = "pull_request_target"

# The ONE reviewed exception to A2, keyed by workflow filename. CI-12 as amended
# by Phase 13 D-19 permits this trigger for a single file, and only while three
# conditions hold. Two of them are STATICALLY ASSERTED below and fail the build;
# the third is a property of what the steps do rather than of a key that can be
# looked up, so it is carried here as a reviewed reason. An excepted file is
# therefore never simply skipped -- a workflow that drifts out of the asserted
# conditions fails the build rather than quietly keeping its exemption.
#
# This file is byte-identical across both repos by design (D-04), so adding or
# removing an entry is a paired cross-repo change.
A2_REVIEWED_EXCEPTIONS: dict[str, str] = {
    "integrity.yml": (
        "Phase 13 D-19, recorded as DESIGN-DECISIONS entry 46 and licensed by CI-12 as amended by "
        "plan 13-14. This is the anti-skip mechanism itself: the ONLY control on this account that "
        "can catch a job-level condition added to a required-context job, because the build-time "
        "self-test lives inside the job such an edit skips. Its check run attaches to the pull "
        "request's HEAD sha while the runner sees the BASE commit, measured by the D-21 probe. "
        "THREE conditions make it safe. (i) BASE-REF CHECKOUT ONLY -- ASSERTED here: no checkout "
        "step in that workflow may carry a `ref:` input. (ii) IT NEVER CHECKS OUT OR EXECUTES "
        "HEAD-REF CODE, with reading the head's workflow YAML as DATA through the contents API "
        "explicitly permitted, because that is the entire point of the job and its parser and rules "
        "both come from the protected branch -- this condition is REVIEWED, not asserted, since it "
        "is a property of what the steps do rather than of a lookupable key. (iii) THE WORKFLOW "
        "GRANTS EXACTLY A READ CONTENT SCOPE -- ASSERTED here, at workflow level and at job level "
        "both, because a job-level block REPLACES rather than narrows the workflow-level one."
    ),
}

# The exact permissions mapping conditions (iii) allows. Compared as a whole
# mapping rather than key-by-key: "grants nothing beyond a read content scope"
# has to mean the block carries nothing else at all, or an added key slips
# through as long as the contents key still reads `read`.
READ_CONTENT_ONLY_PERMISSIONS: dict[str, str] = {"contents": "read"}

CHECKOUT_ACTION = "actions/checkout"
CHECKOUT_REF_INPUT = "ref"

CI_WORKFLOW_FILENAME = "ci.yml"
CI_WORKFLOW_NAME = "CI"

LINT_CONTEXT = "Lint (pre-commit)"

# A7's pending-narrowing register, keyed by required context. An entry suppresses
# A7 for that one context and MUST state which requirement narrows it and which
# plan lands the narrowing. It is a narrowing with a stated reason, not a mute:
# the capability it suppresses is pinned in both directions by
# `test_check_ci_config.py`, which passes an empty register.
#
# EMPTY, and that is the intended steady state — A7 now evaluates every
# required-context lint job for real rather than skipping one. The single entry
# this register ever held covered CI-16's narrowing while the self-test ran one
# wave ahead of it (Phase 13 plan 13-03); it was deleted by plan 13-07 once BOTH
# repos' lint jobs were narrowed (GSEGUtils by 13-04, pchandler by 13-07), in one
# paired change. Adding an entry is likewise a paired cross-repo change: this
# file is byte-identical across both repos by design (D-04).
A7_PENDING_NARROWINGS: dict[str, str] = {}


def load_workflows(root: pathlib.Path) -> list[tuple[pathlib.Path, dict[str, Any]]]:
    """Parse every workflow document under the workflows directory.

    Parameters
    ----------
    root
        Repository root to inspect.

    Returns
    -------
    list of tuple
        One ``(path, document)`` pair per workflow that parses to a mapping,
        in sorted filename order.
    """
    out: list[tuple[pathlib.Path, dict[str, Any]]] = []
    for path in sorted((root / WORKFLOWS_DIR).glob("*.yml")):
        with path.open() as handle:
            document = yaml.safe_load(handle)
        if isinstance(document, dict):
            out.append((path, document))
    return out


def committed_required_contexts(root: pathlib.Path) -> set[str]:
    """Return the union of required status-check contexts across the committed rulesets.

    Parameters
    ----------
    root
        Repository root to inspect.

    Returns
    -------
    set of str
        Every context string any committed payload requires.
    """
    contexts: set[str] = set()
    for path in sorted((root / RULESETS_DIR).glob("*.json")):
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            contexts.update(ruleset_lib.required_contexts(payload))
    return contexts


def _declared_jobs(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the workflow's jobs mapping, defensively, as job id to job body."""
    jobs = document.get("jobs")
    if not isinstance(jobs, dict):
        return {}
    return {str(job_id): job for job_id, job in jobs.items() if isinstance(job, dict)}


def _job_display_name(job_id: str, job: dict[str, Any]) -> str:
    """Return the string GitHub reports as the check-run name for this job."""
    name = job.get("name")
    return name if isinstance(name, str) else job_id


def check_trigger_filters(
    root: pathlib.Path,
    exceptions: dict[str, str] | None = None,
) -> list[str]:
    """A1 -- flag filter keys under a pull-request-side trigger on a required-check workflow.

    Parameters
    ----------
    root
        Repository root to inspect.
    exceptions
        Reviewed-exceptions table keyed by workflow filename; defaults to
        :data:`A1_REVIEWED_EXCEPTIONS`.

    Returns
    -------
    list of str
        One entry per offending filter key.
    """
    table = A1_REVIEWED_EXCEPTIONS if exceptions is None else exceptions
    contexts = committed_required_contexts(root)
    violations: list[str] = []
    for path, document in load_workflows(root):
        produced = sorted(set(ruleset_lib.job_names(document)) & contexts)
        if not produced or path.name in table:
            continue
        block, form = ruleset_lib.trigger_block(document)
        if form is None or block is None:
            # A workflow whose triggers cannot be read has not been checked, and
            # an unchecked required-check workflow is a violation, not a pass.
            violations.append(
                f"{path.name}: no resolvable trigger block under either the quoted or the bare "
                f"`on:` key, on a workflow producing required context(s) {produced} — this "
                f"workflow cannot be checked, so it cannot be assumed clean."
            )
            continue
        if not isinstance(block, dict):
            continue  # a bare list trigger block carries no filter sub-keys
        for event in A1_INSPECTED_EVENTS:
            config = block.get(event)
            if not isinstance(config, dict):
                continue
            for key in sorted(FILTER_KEYS & set(config)):
                violations.append(
                    f"{path.name}: `{key}:` under `{event}:` on a workflow producing required "
                    f"context(s) {produced} — a filtered-out run never reports, so the pull request "
                    f"waits on that check forever and can never merge. Conditionality belongs on "
                    f"STEPS, not on the trigger."
                )
    return violations


def check_base_repo_context_trigger(
    root: pathlib.Path,
    exceptions: dict[str, str] | None = None,
) -> list[str]:
    """A2 -- flag the base-repo-context pull-request trigger, except in the one reviewed file.

    The scan is over raw file text rather than parsed keys, so an occurrence in a
    comment or nested under a reusable-workflow call is caught too. It is scoped
    to the workflows directory so this script's own search literal cannot match
    itself.

    **An excepted file is not simply skipped.** Before the exception is honoured,
    the two statically checkable conditions of :data:`A2_REVIEWED_EXCEPTIONS` are
    asserted and either failure is a violation:

      (i)   no checkout step anywhere in that workflow carries a ``ref:`` input --
            a checkout with an explicit reference is how the job would start
            looking at head code, and it is the single edit that would turn the
            exception into the privilege-escalation vector the general ban exists
            to stop;
      (iii) the workflow grants exactly a read content scope and nothing else, at
            workflow level AND in every job that declares its own block, because
            a job-level ``permissions:`` REPLACES the workflow-level one rather
            than narrowing it.

    The third condition -- that no head-supplied code is executed, with API reads
    of head workflow YAML permitted -- stays a *reviewed* condition stated in the
    exception's reason string, because it is a property of what the steps do
    rather than of a key that can be looked up. It is not enforced here and must
    not be read as though it were.

    Parameters
    ----------
    root
        Repository root to inspect.
    exceptions
        Reviewed-exceptions table keyed by workflow filename; defaults to
        :data:`A2_REVIEWED_EXCEPTIONS`. Injectable so the tests can pin both
        directions, mirroring :func:`check_trigger_filters`'s signature.

    Returns
    -------
    list of str
        One entry per offending workflow file, or per unmet condition on an
        excepted one.
    """
    table = A2_REVIEWED_EXCEPTIONS if exceptions is None else exceptions
    violations: list[str] = []
    for path in sorted((root / WORKFLOWS_DIR).glob("*.yml")):
        lines = [
            str(number)
            for number, line in enumerate(path.read_text().splitlines(), start=1)
            if BASE_REPO_CONTEXT_TRIGGER in line
        ]
        if not lines:
            continue
        if path.name in table:
            violations.extend(_check_exception_conditions(path))
            continue
        violations.append(
            f"{path.name}: `{BASE_REPO_CONTEXT_TRIGGER}` found on line(s) {', '.join(lines)} — "
            f"that trigger runs with the base repository's secrets against a fork's code, and "
            f"one of these repos attaches a self-hosted lab runner. It is never legitimate here. "
            f"Exactly one file is excepted, by path, in A2_REVIEWED_EXCEPTIONS, and this is not it."
        )
    return violations


def _check_exception_conditions(path: pathlib.Path) -> list[str]:
    """Assert conditions (i) and (iii) on the one workflow excepted from A2.

    Parameters
    ----------
    path
        The excepted workflow file.

    Returns
    -------
    list of str
        One entry per unmet condition; empty when the exception still holds.
    """
    try:
        with path.open() as handle:
            document = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        return [
            f"{path.name}: is excepted from A2 but could not be parsed — {error}. Its exception "
            f"conditions cannot be asserted, so the exception cannot be honoured."
        ]
    if not isinstance(document, dict):
        return [
            f"{path.name}: is excepted from A2 but does not parse to a mapping, so its exception "
            f"conditions cannot be asserted and the exception cannot be honoured."
        ]

    violations: list[str] = []

    # Condition (iii), workflow level. Compared as a whole mapping: an added key
    # must fail even while `contents: read` is still present.
    permissions = document.get("permissions")
    if permissions != READ_CONTENT_ONLY_PERMISSIONS:
        violations.append(
            f"{path.name}: is excepted from the `{BASE_REPO_CONTEXT_TRIGGER}` ban only while its "
            f"workflow-level `permissions:` block is exactly {READ_CONTENT_ONLY_PERMISSIONS}, and it "
            f"is {permissions!r} — condition (iii) of that exception. A base-repo-context job with "
            f"anything beyond a read content scope is the privilege escalation the general ban "
            f"exists to stop."
        )

    for job_id, job in _declared_jobs(document).items():
        # Condition (iii), job level. A job-level block REPLACES the
        # workflow-level one rather than narrowing it, so a job could otherwise
        # widen the scope while the workflow-level block still read clean.
        if "permissions" in job and job.get("permissions") != READ_CONTENT_ONLY_PERMISSIONS:
            violations.append(
                f"{path.name}: job `{job_id}` declares `permissions: {job.get('permissions')!r}`, "
                f"which is not exactly {READ_CONTENT_ONLY_PERMISSIONS} — condition (iii) of this "
                f"file's A2 exception. A job-level block REPLACES the workflow-level one, so this "
                f"widens the scope rather than narrowing it."
            )
        # Condition (i).
        steps = job.get("steps")
        for index, step in enumerate(steps if isinstance(steps, list) else []):
            if not isinstance(step, dict) or CHECKOUT_ACTION not in str(step.get("uses") or ""):
                continue
            inputs = step.get("with")
            if isinstance(inputs, dict) and CHECKOUT_REF_INPUT in inputs:
                violations.append(
                    f"{path.name}: job `{job_id}` step {index} checks out with a "
                    f"`{CHECKOUT_REF_INPUT}:` input — condition (i) of this file's A2 exception is "
                    f"base-ref checkout ONLY. A checkout carrying an explicit reference is how this "
                    f"job would start looking at head code, and it is the single edit that turns the "
                    f"exception into the vector the ban exists to stop."
                )
    return violations


def check_workflow_name_pin(root: pathlib.Path) -> list[str]:
    """A3 -- assert the CI workflow name is byte-exact and every workflow-run reference resolves.

    Parameters
    ----------
    root
        Repository root to inspect.

    Returns
    -------
    list of str
        One entry per name mismatch or unresolvable workflow-run reference.
    """
    workflows = load_workflows(root)
    violations: list[str] = []

    declared = {document.get("name") for _, document in workflows if isinstance(document.get("name"), str)}
    ci = [document for path, document in workflows if path.name == CI_WORKFLOW_FILENAME]
    if not ci:
        violations.append(f"{CI_WORKFLOW_FILENAME}: not found under {WORKFLOWS_DIR} — the pinned CI workflow is gone.")
    else:
        # Byte-exact on purpose: no case folding, no stripping. Any consumer
        # matching on this string matches it literally.
        actual = ci[0].get("name")
        if actual != CI_WORKFLOW_NAME:
            violations.append(
                f"{CI_WORKFLOW_FILENAME}: declared name is {actual!r}, not {CI_WORKFLOW_NAME!r} — the "
                f"run-counting rule in the workspace audit script "
                f"(.planning/scripts/audit_release_flow.py, CI_SUBTOTAL_WORKFLOW) keys on that exact "
                f"string, and any workflow keying on another workflow's name breaks silently."
            )

    for path, document in workflows:
        block, _ = ruleset_lib.trigger_block(document)
        if not isinstance(block, dict):
            continue
        config = block.get("workflow_run")
        if not isinstance(config, dict):
            continue
        for referenced in config.get("workflows") or []:
            if referenced not in declared:
                violations.append(
                    f"{path.name}: `workflow_run` lists workflow name {referenced!r}, which no "
                    f"workflow in this repository declares — this trigger will never fire, silently."
                )
    return violations


def _allowlist_from_composite(root: pathlib.Path) -> set[str] | None:
    """Extract the fast-path allowlist from the composite action's step body."""
    path = root / COMPOSITE_ACTION_PATH
    if not path.is_file():
        return None
    match = re.search(r"ALLOW='([^']*)'", path.read_text())
    if match is None:
        return None
    return set(match.group(1).split())


def check_fast_path_allowlist(root: pathlib.Path) -> list[str]:
    """A4 -- reconcile the fast-path allowlist against release-please's own configuration.

    Parameters
    ----------
    root
        Repository root to inspect.

    Returns
    -------
    list of str
        A single entry when the two sets differ, or when either side is unreadable.
    """
    allowlist = _allowlist_from_composite(root)
    if allowlist is None:
        return [
            f"{COMPOSITE_ACTION_PATH}: no `ALLOW='...'` assignment found — the fast-path allowlist "
            f"cannot be reconciled against {RELEASE_PLEASE_CONFIG_PATH}."
        ]

    config_path = root / RELEASE_PLEASE_CONFIG_PATH
    if not config_path.is_file():
        return [f"{RELEASE_PLEASE_CONFIG_PATH}: not found — the fast-path allowlist cannot be reconciled."]

    expected = {RELEASE_PLEASE_MANIFEST_PATH}
    for package in (json.loads(config_path.read_text()).get("packages") or {}).values():
        if not isinstance(package, dict):
            continue
        expected.add(str(package.get("changelog-path") or "CHANGELOG.md"))
        expected.update(str(extra) for extra in package.get("extra-files") or [])

    if allowlist == expected:
        return []
    return [
        f"{COMPOSITE_ACTION_PATH}: the fast-path allowlist {sorted(allowlist)} no longer set-equals "
        f"the release artifacts {sorted(expected)} derived from {RELEASE_PLEASE_CONFIG_PATH} "
        f"(changelog path + extra files) plus {RELEASE_PLEASE_MANIFEST_PATH}. A new extra file must "
        f"be added to the allowlist, or the fast path silently stops engaging on release pull "
        f"requests and the economy it exists to buy is lost without any failure."
    ]


def check_job_level_conditions(root: pathlib.Path) -> list[str]:
    """A5 -- flag a job-level condition on any job whose declared name is a required context.

    Parameters
    ----------
    root
        Repository root to inspect.

    Returns
    -------
    list of str
        One entry per offending job.
    """
    contexts = committed_required_contexts(root)
    violations: list[str] = []
    for path, document in load_workflows(root):
        for job_id, job in _declared_jobs(document).items():
            name = _job_display_name(job_id, job)
            if name in contexts and "if" in job:
                violations.append(
                    f"{path.name}: job `{job_id}` produces required context {name!r} and carries a "
                    f"job-level condition — a skipped job's `skipped` conclusion counts among the "
                    f"satisfying ones, so this greens the gate having executed nothing. "
                    f"Conditionality belongs on STEPS."
                )
    return violations


def check_conditional_dependencies(root: pathlib.Path) -> list[str]:
    """A6 -- flag a required-context job that depends on a job carrying a job-level condition.

    Parameters
    ----------
    root
        Repository root to inspect.

    Returns
    -------
    list of str
        One entry per offending dependency edge.
    """
    contexts = committed_required_contexts(root)
    violations: list[str] = []
    for path, document in load_workflows(root):
        jobs = _declared_jobs(document)
        for job_id, job in jobs.items():
            name = _job_display_name(job_id, job)
            if name not in contexts:
                continue
            needs = job.get("needs")
            needs = [needs] if isinstance(needs, str) else needs
            for dependency in needs or []:
                upstream = jobs.get(str(dependency))
                if isinstance(upstream, dict) and "if" in upstream:
                    violations.append(
                        f"{path.name}: job `{job_id}` produces required context {name!r} and depends "
                        f"on `{dependency}`, which carries a job-level condition — a skipped "
                        f"dependency skips the dependent, and that also counts as satisfying."
                    )
    return violations


def check_lint_least_privilege(
    root: pathlib.Path,
    pending: dict[str, str] | None = None,
) -> list[str]:
    """A7 -- assert the lint job declares a permissions block granting no write scope.

    Parameters
    ----------
    root
        Repository root to inspect.
    pending
        Pending-narrowing register keyed by required context; defaults to
        :data:`A7_PENDING_NARROWINGS`. A context listed here is exempt, and a
        listed context that is already narrowed emits a removal notice.

    Returns
    -------
    list of str
        One entry per write scope, or one entry when no permissions block exists.
    """
    register = A7_PENDING_NARROWINGS if pending is None else pending
    violations: list[str] = []
    found = False
    for path, document in load_workflows(root):
        for job_id, job in _declared_jobs(document).items():
            if _job_display_name(job_id, job) != LINT_CONTEXT:
                continue
            found = True
            granted = _write_scopes(job.get("permissions"))
            if granted is None:
                violations.append(
                    f"{path.name}: job `{job_id}` produces required context {LINT_CONTEXT!r} and "
                    f"declares no `permissions:` block, so it inherits the repository default "
                    f"instead of stating its least privilege explicitly (CI-16)."
                )
                continue
            if LINT_CONTEXT in register:
                if not granted:
                    print(
                        f"::notice::check_ci_config: A7's pending-narrowing entry for "
                        f"{LINT_CONTEXT!r} is no longer warranted — `{job_id}` in {path.name} grants "
                        f"no write scope. Delete the entry from A7_PENDING_NARROWINGS in "
                        f"{SELF_PATH}, in BOTH repos in the same paired change — this file is "
                        f"byte-identical across them by design (D-04)."
                    )
                continue
            for scope in granted:
                violations.append(
                    f"{path.name}: job `{job_id}` produces required context {LINT_CONTEXT!r} and "
                    f"grants `{scope}: write` — CI-16 narrows this job to read scopes only, and a "
                    f"write scope here is a token the pre-commit run does not need."
                )
    if not found:
        violations.append(
            f"No job under {WORKFLOWS_DIR} declares the name {LINT_CONTEXT!r}, yet it is a required "
            f"status-check context — nothing produces it, so the gate can never be satisfied."
        )
    return violations


def _write_scopes(permissions: Any) -> list[str] | None:
    """Return the permission keys granted at write, or None when no block is declared."""
    if permissions is None:
        return None
    if isinstance(permissions, str):
        # The shorthand forms: `permissions: read-all` / `write-all`.
        return [] if permissions == "read-all" else [permissions]
    if not isinstance(permissions, dict):
        return None
    return [str(key) for key, value in permissions.items() if str(value) == "write"]


def run_all(
    root: pathlib.Path,
    exceptions: dict[str, str] | None = None,
    pending: dict[str, str] | None = None,
) -> tuple[list[str], int]:
    """Run every assertion against a tree and return the violations plus the workflow count.

    Parameters
    ----------
    root
        Repository root to inspect.
    exceptions
        A1's reviewed-exceptions table; defaults to the module constant.
    pending
        A7's pending-narrowing register; defaults to the module constant.

    Returns
    -------
    tuple
        The accumulated violations and the number of workflows inspected.
    """
    violations = [
        *check_trigger_filters(root, exceptions),
        *check_base_repo_context_trigger(root),  # its own table; see A2_REVIEWED_EXCEPTIONS
        *check_workflow_name_pin(root),
        *check_fast_path_allowlist(root),
        *check_job_level_conditions(root),
        *check_conditional_dependencies(root),
        *check_lint_least_privilege(root, pending),
    ]
    return violations, len(load_workflows(root))


# The driver runs under `python .github/scripts/check_ci_config.py`, which is the
# ONLY invocation contract this file has and is unchanged by the guard below.
#
# The guard exists because this module is now IMPORTED as a library, by
# `assert_no_skip.py` (which calls A5 and A6 directly rather than reimplementing
# them) and by `test_check_ci_config.py`. Without it, importing this module runs
# the whole self-test against THIS repository as an import side effect — and on a
# tree that failed any assertion it would `sys.exit(1)` inside the importer,
# before the importer's own rules had run. For `assert_no_skip.py` that is a
# wrong-verdict bug in a branch-protection gate: the integrity context would go
# red reporting the BASE tree's self-test findings, having never looked at the
# pull request at all. The straight-line shape plan 13-03 chose is preserved;
# only the import side effect is removed.
if __name__ == "__main__":
    findings, inspected_count = run_all(REPO_ROOT)
    if findings:
        for finding in findings:
            print(f"::error::{finding}")
        sys.exit(1)
    print(f"check_ci_config: OK — assertions A1 through A7 clean across {inspected_count} workflow(s)")
