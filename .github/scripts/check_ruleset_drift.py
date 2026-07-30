#!/usr/bin/env python3
# .github/scripts/check_ruleset_drift.py
# Source: Phase 13 CONTEXT.md D-04/D-08 — the PROT-02 live-vs-committed drift comparator
r"""Compare live ruleset reads against their committed JSON and fail on surviving drift.

Invocation, one or more positional arguments, each shaped
``name:live.json:committed.json``::

    python .github/scripts/check_ruleset_drift.py \\
        "protect-main:/tmp/live-main.json:.github/rulesets/main.json" \\
        "protect-develop-gsd:/tmp/live-develop.json:.github/rulesets/develop.json"

Called from the nightly ``scheduled-health.yml`` job, after a step has written
each live read to a file with ``includes_parents=false`` — a parent organization
ruleset is not drift in a committed file and would be reported as one.

Exits 0 = every pair compared clean; exits 1 = at least one surviving difference
or at least one unreadable payload. Every failure is accumulated and annotated
before the single exit, so one run reports the whole set rather than the first
item. An unreadable payload (``RulesetReadError``) is a failure in its own right:
a check must never report clean about a field it could not read.

Report on stdout, diagnostics on stderr.
"""

import json
import pathlib
import sys

import ruleset_lib

USAGE = "usage: check_ruleset_drift.py NAME:LIVE.json:COMMITTED.json [...]"


def parse_spec(spec: str) -> tuple[str, pathlib.Path, pathlib.Path]:
    """Split one ``name:live:committed`` argument into its three parts.

    Split from the right on exactly two colons, so a ruleset name containing a
    colon is still parseable while the two trailing paths are unambiguous.

    Parameters
    ----------
    spec
        One positional argument.

    Returns
    -------
    tuple
        The ruleset name, the live payload path and the committed payload path.

    Raises
    ------
    ValueError
        When the argument does not carry two colons.
    """
    parts = spec.rsplit(":", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"malformed pair `{spec}` — expected NAME:LIVE.json:COMMITTED.json")
    return parts[0], pathlib.Path(parts[1]), pathlib.Path(parts[2])


def load(path: pathlib.Path) -> dict[str, object]:
    """Read a JSON payload from `path` and assert it is an object.

    Parameters
    ----------
    path
        The file to read.

    Returns
    -------
    dict
        The parsed payload.

    Raises
    ------
    ruleset_lib.RulesetReadError
        When the file is missing, unparseable, or not a JSON object.
    """
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as error:
        raise ruleset_lib.RulesetReadError(f"{path}: could not be read as JSON — {error}") from error
    if not isinstance(payload, dict):
        raise ruleset_lib.RulesetReadError(f"{path}: parsed to {type(payload).__name__}, expected a JSON object")
    return payload


def main(argv: list[str]) -> int:
    """Compare every requested pair, annotate every failure, and exit once.

    Parameters
    ----------
    argv
        Positional arguments, excluding the program name.

    Returns
    -------
    int
        0 when every pair compared clean, 1 otherwise.
    """
    if not argv:
        print(f"::error::{USAGE}", file=sys.stderr)
        return 1

    failures: list[str] = []
    normalised_away = 0
    compared = 0

    for spec in argv:
        try:
            name, live_path, committed_path = parse_spec(spec)
        except ValueError as error:
            failures.append(str(error))
            continue

        try:
            live = load(live_path)
            committed = load(committed_path)
            norm_live, norm_committed, removed = ruleset_lib.normalize(live, committed)
        except ruleset_lib.RulesetReadError as error:
            failures.append(f"{name}: {error}")
            continue

        compared += 1
        normalised_away += len(removed)
        for record in removed:
            print(f"{name}: normalised away — {record}", file=sys.stderr)

        differences = ruleset_lib.diff(norm_live, norm_committed)
        failures.extend(f"{name}: drift survives normalisation — {difference}" for difference in differences)

    if failures:
        for failure in failures:
            print(f"::error::{failure}")
        return 1

    print(
        f"check_ruleset_drift: OK — {compared} ruleset(s) compared, "
        f"0 differences surviving, {normalised_away} normalised away"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
