#!/usr/bin/env python3
# .github/scripts/preflight_ruleset_apply.py
# Source: Phase 13 CONTEXT.md D-03 (job-name cross-reference gate) + D-01 (committed is authoritative)
r"""Gate a ruleset apply on the target branch actually producing every required context.

Invocation, three positional arguments::

    python .github/scripts/preflight_ruleset_apply.py \\
        .github/rulesets/main.json /tmp/target-workflows /tmp/send-payload.json

Called from ``ruleset-apply.yml`` between the step that fetches the TARGET
branch tip's workflow files and the step that sends the update. Reading the
workflows from the target tip rather than from the checked-out ref is the whole
point: dispatching from a stale branch would otherwise cross-reference stale job
names, pass, and break the live gate.

Exits 0 = the payload may be applied, and the send body has been written to the
third argument; exits 1 = refused, and NO send payload exists. Every violation is
accumulated and annotated before the single exit, so one run reports the whole
set rather than the first item.

The check is a containment test -- the payload's required contexts must all be
job names on the target tip -- which is why it refuses in **both** directions
with one assertion: a payload that adds a context nothing produces, and a target
tip from which a job was removed while a payload still requires it. That is what
makes CI-10's same-commit constraint enforceable rather than merely documented.
Under an empty bypass-actor list (the v1.1 security floor, PROT-03) a branch
requiring a check nothing produces is unmergeable by anyone, including an admin;
this gate is what stands between a dispatch and that state.

Deliberately NOT inherited: the workspace audit script's read-only argv guard and
the frozen set of write-capable ``gh`` flags it rejects. That guard makes the
audit provably read-only and is correct there; this script exists precisely to
feed a write, so a guard forbidding write verbs cannot serve it. The read-only
property that matters lives elsewhere -- at the drift job's token mint scope,
which is the only place it can actually bind -- and the integrity properties that
matter *here* are this preflight plus the committed-paths-only restriction below.
The rejected constant is named descriptively rather than literally so that a grep
for it over this file returns zero, which is the phase's own check that the guard
was not copied.

Report on stdout, diagnostics on stderr.
"""

import json
import os
import pathlib
import sys
from typing import Any

import ruleset_lib
import yaml  # available on ubuntu-latest runner by default

USAGE = "usage: preflight_ruleset_apply.py COMMITTED_PAYLOAD.json TARGET_WORKFLOW_DIR SEND_PAYLOAD.json"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# The ONLY directory a payload may be applied from. This script runs inside a job
# holding a protection-rewriting credential, so the set of objects it can apply
# must be the set of objects committed and reviewed -- not whatever path a
# dispatcher can name. The apply workflow additionally selects the filename from
# a fixed `choice` input, so the two restrictions compose.
RULESETS_DIR = REPO_ROOT / ".github/rulesets"

# GitHub accepts either extension for a workflow file. Both are globbed: a
# workflow this script cannot see contributes no job names, which makes a
# legitimate context look unproduced -- a false REFUSAL. That direction is safe,
# but it is still wrong, and the fix costs one glob pattern.
WORKFLOW_GLOBS: tuple[str, ...] = ("*.yml", "*.yaml")


class PreflightError(Exception):
    """Raised when the preflight cannot proceed at all, as distinct from a violation.

    A violation says the payload and the target branch disagree and is
    accumulated alongside its peers. This says the run could not be set up --
    a payload outside the committed rulesets directory, a bad argument count --
    so there is nothing to accumulate.
    """


def resolve_committed_payload(candidate: str, rulesets_dir: pathlib.Path) -> pathlib.Path:
    """Resolve `candidate` and assert it lives inside the committed rulesets directory.

    Symlinks are resolved on both sides before the containment test, so a link
    planted inside the rulesets directory cannot smuggle in an outside file. The
    file is NOT opened here: an outside path must be refused without being read.

    Parameters
    ----------
    candidate
        The payload path as supplied on the command line.
    rulesets_dir
        The directory committed payloads live in.

    Returns
    -------
    pathlib.Path
        The resolved payload path.

    Raises
    ------
    PreflightError
        When the resolved path is not a file directly inside `rulesets_dir`.
    """
    root = rulesets_dir.resolve()
    resolved = pathlib.Path(candidate).resolve()
    if resolved.parent != root:
        raise PreflightError(
            f"refusing `{candidate}`: it resolves to `{resolved}`, which is not inside this "
            f"repository's committed rulesets directory `{root}`. This job holds a "
            f"protection-rewriting credential and applies only committed, reviewed payloads — "
            f"never a caller-supplied path."
        )
    if not resolved.is_file():
        raise PreflightError(f"refusing `{candidate}`: `{resolved}` is not a regular file.")
    return resolved


def target_branch(payload: dict[str, Any]) -> str:
    """Return the single ref name the payload's conditions include.

    Taken from the payload itself rather than from a fourth argument, so the
    branch named in every violation message is the branch the payload would
    actually protect -- not one a dispatcher asserted.

    Parameters
    ----------
    payload
        A committed ruleset payload.

    Returns
    -------
    str
        The included ref name, or ``"<no included ref>"`` when the payload
        declares none.
    """
    conditions = payload.get("conditions")
    if isinstance(conditions, dict):
        ref_name = conditions.get("ref_name")
        if isinstance(ref_name, dict):
            include = ref_name.get("include")
            if isinstance(include, list) and include:
                return ", ".join(str(ref) for ref in include)
    return "<no included ref>"


def matchable_job_names(workflow_dir: pathlib.Path) -> tuple[set[str], list[str]]:
    """Build the set of check-run names the target tip's workflows can produce.

    The set is the union of every job's declared ``name:``, falling back to the
    bare job id -- which is exactly what GitHub reports as the check-run name for
    an unnamed job, so the fallback is the honest reading rather than a
    convenience.

    A workflow whose trigger block resolves under neither the quoted nor the bare
    ``on:`` key is recorded as a violation rather than skipped: an unreadable
    workflow has not been checked, and an unchecked workflow cannot be assumed
    to contribute nothing.

    Parameters
    ----------
    workflow_dir
        A directory holding the target branch tip's workflow files.

    Returns
    -------
    tuple
        The matchable name set and the list of violations found while building it.
    """
    violations: list[str] = []
    if not workflow_dir.is_dir():
        violations.append(
            f"the target workflow directory `{workflow_dir}` does not exist — nothing was fetched "
            f"from the target branch tip, so no cross-reference is possible and this apply must not proceed."
        )
        return set(), violations

    paths: list[pathlib.Path] = []
    for pattern in WORKFLOW_GLOBS:
        paths.extend(workflow_dir.glob(pattern))

    names: set[str] = set()
    for path in sorted(paths):
        try:
            with path.open(encoding="utf-8") as handle:
                document = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError) as error:
            violations.append(f"{path.name}: could not be parsed as YAML — {error}")
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
                f"`on:` key — this workflow has not been checked, so it cannot be assumed clean."
            )
            continue
        names.update(ruleset_lib.job_names(document))

    if not paths:
        violations.append(
            f"the target workflow directory `{workflow_dir}` holds no workflow files — nothing was "
            f"fetched from the target branch tip, so no cross-reference is possible."
        )
    return names, violations


def cross_reference(
    contexts: list[str],
    matchable: set[str],
    payload_path: pathlib.Path,
    branch: str,
) -> list[str]:
    """Assert every required context is a job name on the target tip -- the D-03 gate.

    Parameters
    ----------
    contexts
        The payload's required status-check contexts.
    matchable
        Job names the target tip's workflows produce.
    payload_path
        The committed payload being applied, named in every violation.
    branch
        The ref the payload protects, named in every violation.

    Returns
    -------
    list of str
        One entry per context no job on the target tip produces.
    """
    return [
        f"`{context}` is required by `{payload_path.name}` but no job on `{branch}` produces it. "
        f"Applying this payload would leave that branch requiring a check nothing reports, and under "
        f"an empty bypass-actor list that makes the branch unmergeable by anyone, including an admin. "
        f"Land the producing job on `{branch}` first, or remove the context from the payload."
        for context in contexts
        if context not in matchable
    ]


def write_send_payload(payload: dict[str, Any], destination: pathlib.Path) -> None:
    """Serialise the committed payload into a request body and write it atomically.

    Atomic because the workflow's send step reads this path: a partial file is a
    partial request body against a protected branch. The temporary file is
    replaced onto the destination in one operation, so the destination is either
    absent or complete.

    Parameters
    ----------
    payload
        The committed payload.
    destination
        Where to write the send body.
    """
    body = ruleset_lib.to_payload(payload)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def main(argv: list[str], rulesets_dir: pathlib.Path | None = None) -> int:
    """Cross-reference the payload against the target tip and emit the send body on success.

    Parameters
    ----------
    argv
        Positional arguments, excluding the program name.
    rulesets_dir
        The committed rulesets directory; defaults to this repository's.

    Returns
    -------
    int
        0 when the payload may be applied, 1 otherwise.
    """
    if len(argv) != 3:
        print(f"::error::{USAGE}", file=sys.stderr)
        return 1

    payload_argument, workflow_argument, output_argument = argv
    output_path = pathlib.Path(output_argument)
    # Removed UP FRONT, before anything can fail. A refusal must never leave a
    # send payload behind for the workflow's send step to pick up -- a stale body
    # from an earlier run is exactly the shape that turns a refusal into a write.
    output_path.unlink(missing_ok=True)

    try:
        payload_path = resolve_committed_payload(payload_argument, rulesets_dir or RULESETS_DIR)
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise PreflightError(f"`{payload_path}` parsed to {type(payload).__name__}, expected a JSON object.")
    except PreflightError as error:
        print(f"::error::{error}")
        return 1
    except (OSError, ValueError) as error:
        print(f"::error::`{payload_argument}` could not be read as JSON — {error}")
        return 1

    branch = target_branch(payload)
    contexts = ruleset_lib.required_contexts(payload)
    matchable, violations = matchable_job_names(pathlib.Path(workflow_argument))
    for name in sorted(matchable):
        print(f"{payload_path.name}: target tip produces check-run name `{name}`", file=sys.stderr)
    violations.extend(cross_reference(contexts, matchable, payload_path, branch))

    if violations:
        for violation in violations:
            print(f"::error::{violation}")
        return 1

    write_send_payload(payload, output_path)
    print(
        f"preflight_ruleset_apply: OK — `{payload_path.name}` vs `{branch}`, "
        f"{len(contexts)} required context(s) all matched a job name on the target tip; "
        f"send payload written to {output_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
