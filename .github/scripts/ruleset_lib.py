#!/usr/bin/env python3
# .github/scripts/ruleset_lib.py
# Source: Phase 13 CONTEXT.md D-04 (byte-identical in both repos) + D-16 (absent-key guard)
"""Shared primitives for the ruleset drift check, the CI-12 self-test and the apply preflight.

A pure library: no ``main``, no argv parsing, no side effects at import. Importers
are ``check_ruleset_drift.py`` (PROT-02, read path), ``check_ci_config.py`` (CI-12)
and ``preflight_ruleset_apply.py`` (D-03, write path), all of which live in this
same directory, so a plain ``import ruleset_lib`` resolves through the script
directory Python prepends to ``sys.path``.

The comparison contract is inherited wholesale from ``12-AUDIT.md`` section 5 and
``.planning/scripts/audit_release_flow.py``; where this module departs from that
source, the departure is stated in the relevant docstring rather than left for a
reader to discover by diffing.

Deliberately NOT inherited: the audit's ``_assert_read_only`` argv guard and the
frozenset of write-capable ``gh`` flags it rejects. That guard is what makes the
audit script provably read-only and it is correct there, but this library is
shared by a read-only drift job and a write-capable apply job (Phase 13
D-01/D-03), and a library that forbids write verbs cannot serve the second.
Read-only enforcement for the drift job lives at the job level instead, in the
token mint scope, which is the only place it can actually bind. The constant is
named descriptively rather than literally here so that a grep for it over this
file returns zero, which is the phase's own check that the guard was not copied.
"""

import json
from typing import Any

# Live-only envelope keys with no committed counterpart at all -- rule (a).
# Copied verbatim from audit_release_flow.py:139-148.
LIVE_ONLY_ENVELOPE_KEYS: tuple[str, ...] = (
    "_links",
    "created_at",
    "current_user_can_bypass",
    "id",
    "node_id",
    "source",
    "source_type",
    "updated_at",
)
# Keys GitHub fills on read that the committed files never set -- rule (d).
# Copied verbatim from audit_release_flow.py:150-155.
PULL_REQUEST_READ_FILLED_KEYS: tuple[str, ...] = (
    "allowed_merge_methods",
    "dismissal_restriction",
    "required_reviewers",
)
STATUS_CHECKS_READ_FILLED_KEYS: tuple[str, ...] = ("do_not_enforce_on_create",)

# The complete PUT body schema for `PUT /repos/{owner}/{repo}/rulesets/{id}`, in
# the order the committed `.github/rulesets/*.json` files use -- which is fixed
# and is NOT alphabetical. The API declares no required fields and does not
# document whether omitted ones are preserved or reset, so the apply path always
# sends the complete object; the committed files already carry exactly these six
# keys and nothing else, which is what makes "committed is authoritative" true.
SENDABLE_KEYS: tuple[str, ...] = (
    "name",
    "target",
    "enforcement",
    "conditions",
    "bypass_actors",
    "rules",
)

BYPASS_ACTORS_KEY = "bypass_actors"

# The two keys a GitHub REST error body always carries, and the two a real ruleset
# always carries. An error envelope is discriminated structurally on those four --
# never by matching against message text, which GitHub is free to reword.
API_ERROR_ENVELOPE_KEYS: tuple[str, ...] = ("message", "status")
RULESET_IDENTITY_KEYS: tuple[str, ...] = ("id", "rules")

# The status the D-22 probe measured when the rulesets endpoint refuses a private
# repository on a plan without rulesets. GitHub returns it inside the error body as
# a JSON *string*, not an integer, so every comparison against it normalises to str.
RULESETS_UNAVAILABLE_STATUS = "403"


class RulesetReadError(Exception):
    """Raised when a live ruleset payload cannot be trusted enough to compare.

    Distinct from a drift finding. A drift finding says the live state differs
    from the committed state; this says the live state could not be fully read,
    so no verdict about it is available. Callers must surface it as a failure and
    must never fall through to a clean result.
    """


def assert_not_api_error(payload: dict[str, Any], source: str) -> None:
    """Raise when a parsed payload is a GitHub API error envelope rather than a ruleset.

    **Why this is a hard failure and not a drift finding.** A REST error body is a
    well-formed JSON object, so it parses cleanly and would be handed to
    :func:`normalize` and then to :func:`diff` like any other payload. Compared
    against a committed file it produces differences that describe the envelope
    rather than the protection state -- or, if the caller were ever to treat an
    unreadable read as "nothing to report", no differences at all. **A check that
    cannot read is indistinguishable from a passing one**, which is the exact
    fail-open shape PROT-02 exists to eliminate. This generalises D-16's rule that
    an absent ``bypass_actors`` key is a hard failure: an absent *whole payload* is
    one too, and it gets its own named reason so the two are told apart in the log.

    **Discrimination is structural, never textual.** An error envelope carries
    :data:`API_ERROR_ENVELOPE_KEYS` and carries neither of
    :data:`RULESET_IDENTITY_KEYS`; every real ruleset carries both of the latter.
    Matching on message text would break the first time GitHub rewords a string,
    and would be defeated by a ruleset that happened to be *named* something
    error-shaped.

    The raised message has two parts, deliberately: the status and message the API
    actually returned, quoted so the operator sees the real cause rather than this
    module's interpretation of it, followed by a named diagnosis. For the status the
    D-22 probe measured (:data:`RULESETS_UNAVAILABLE_STATUS`) that diagnosis names
    the plan-and-visibility cause, because a template adopter hitting this is on a
    private repository on a plan without rulesets and needs to be told what to do,
    not told that a field is missing.

    Parameters
    ----------
    payload
        A parsed JSON object, live or committed.
    source
        Where the payload came from -- a file path or a ruleset name -- quoted back
        in the failure so one line identifies which read refused.

    Raises
    ------
    RulesetReadError
        When `payload` is an API error envelope.
    """
    is_envelope = all(key in payload for key in API_ERROR_ENVELOPE_KEYS) and not any(
        key in payload for key in RULESET_IDENTITY_KEYS
    )
    if not is_envelope:
        return

    status = str(payload.get("status", "")).strip()
    message = str(payload.get("message", "")).strip()

    if status == RULESETS_UNAVAILABLE_STATUS:
        diagnosis = (
            "the rulesets endpoint is unavailable for this repository's plan and visibility — "
            "a private repository on a plan without rulesets has no ruleset surface at all, so "
            "there is no floor here to compare against and this is NOT an absence of drift. "
            "Make the repository public, or move it to a plan that has rulesets, before this "
            "check can mean anything (Phase 13 D-22)."
        )
    else:
        diagnosis = (
            f"the live ruleset read failed with status {status or '<none>'} and returned an API "
            f"error envelope instead of a ruleset, so no verdict about the live state is "
            f"available and this run must not report clean."
        )

    raise RulesetReadError(
        f'{source}: the read returned an API error envelope, not a ruleset — status "{status}", '
        f'message "{message}". {diagnosis}'
    )


def trigger_block(document: dict[str, Any]) -> tuple[Any, str | None]:
    """Return a workflow's trigger block and which `on:` key form carried it.

    **The trigger-key trap.** A bare `on:` key is resolved by PyYAML to the
    Python boolean ``True`` (YAML 1.1 treats `on` as a boolean literal), while a
    quoted `"on":` resolves to the string.  Both forms are live in these repos --
    the two `publish-*` workflows quote the key and every other workflow does
    not -- so the string key is looked up first and the boolean key is the
    fallback, and which form was found is recorded.  A silent ``None`` here
    would report a workflow as having no triggers at all.

    Returns ``(None, None)`` rather than raising when neither key form is
    present. This is a deliberate choice, not an omission: two callers share this
    function -- the CI-12 self-test and the D-03 apply preflight -- and a raise
    inside a shared library is not catchable at the call site in a way that lets
    either caller accumulate the failure alongside its others. Callers turn the
    empty result into a recorded failure, which is the shape
    ``audit_release_flow.py:471-477`` already uses.

    Parameters
    ----------
    document
        A workflow document as returned by ``yaml.safe_load``.

    Returns
    -------
    tuple
        The trigger block and one of ``"quoted"``, ``"bare"`` or ``None``.
    """
    if "on" in document:
        return document["on"], "quoted"
    if True in document:
        return document[True], "bare"
    return None, None


def job_names(document: dict[str, Any]) -> list[str]:
    """Return each job's `name:` value, falling back to the bare job id when unset.

    GitHub displays the job id when a job declares no `name:`, and the displayed
    string is what a ruleset context matches on -- so the fallback is the honest
    reading, not a convenience.

    Note the difference from ``audit_release_flow.py:429-443``, which appends the
    literal suffix ``" (job id)"`` to an unnamed job. That suffix is a rendering
    affordance for a human report. Here the value is cross-referenced against
    ruleset required-context strings, so the suffix must NOT be appended: the
    unnamed `canary` job reports to GitHub as the check run ``canary``, and a
    preflight comparing against ``canary (job id)`` would never match it.

    Parameters
    ----------
    document
        A workflow document as returned by ``yaml.safe_load``.

    Returns
    -------
    list of str
        One entry per job, in declaration order.
    """
    jobs = document.get("jobs")
    if not isinstance(jobs, dict):
        return []
    names: list[str] = []
    for job_id, job in jobs.items():
        name = job.get("name") if isinstance(job, dict) else None
        names.append(str(name) if isinstance(name, str) else str(job_id))
    return names


def required_contexts(payload: dict[str, Any]) -> list[str]:
    """Return the sorted `required_status_checks` context strings from a ruleset payload.

    Parameters
    ----------
    payload
        A ruleset mapping, live or committed.

    Returns
    -------
    list of str
        The context strings, sorted. Empty when the payload declares no
        ``required_status_checks`` rule.
    """
    for rule in payload.get("rules") or []:
        if not isinstance(rule, dict) or rule.get("type") != "required_status_checks":
            continue
        parameters = rule.get("parameters") or {}
        checks = parameters.get("required_status_checks") or []
        return sorted(str(c.get("context")) for c in checks if isinstance(c, dict) and c.get("context"))
    return []


def normalize(live: dict[str, Any], committed: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Reduce a live and a committed ruleset payload to a comparable shape.

    Inherits the enumerated contract of ``12-AUDIT.md`` section 5. Every field
    this function drops, re-keys or sorts is enumerated here **and** recorded in
    the returned removal list. A key that vanishes without a matching entry in
    that list is the failure mode this design exists to prevent.

    Drops (recorded):
      (a) Live-only envelope keys with no committed counterpart at all:
          ``id``, ``node_id``, ``source``, ``source_type``, ``created_at``,
          ``updated_at``, ``_links``, ``current_user_can_bypass``.
      (c) ``integration_id`` on every entry of the inner ``required_status_checks``
          list -- live returns the integer 15368 (the GitHub Actions app), the
          committed files carry JSON ``null``.
      (d) Keys GitHub fills on read that the committed side is silent about.

    Re-keys and sorts (shape-only, no information removed, so not recorded):
      (b) ``rules`` goes from a list of ``{type, parameters}`` objects to a
          mapping keyed on ``type``, because the two sides order the list
          differently and comparing by position would report a difference that
          does not exist.
      (c) The inner ``required_status_checks`` list is sorted on ``context``.
      (e) ``conditions.ref_name.include`` and ``.exclude`` are sorted.

    **Rule (d) is conditional here, unlike in the audit.** A read-filled key is
    dropped only when the committed side does not set it, and is compared when
    the committed side does. The audit drops unconditionally, which is safe for a
    report but not for a gate: ``allowed_merge_methods`` reads live as
    merge/squash/rebase and is load-bearing for DESIGN-DECISIONS entry 34's
    squash-only branch model, so an unconditional drop would make this check
    blind to exactly the key that model depends on. This closes the standing
    STATE.md "rule (d) fail-open" blocker.

    **The bypass-actors guard.** Before any comparison, the live payload is
    asserted to carry a ``bypass_actors`` key. GitHub returns that property only
    to a requester with write access to the ruleset, so a read-scoped token
    receives a payload with the key ABSENT -- and an absent key compared against
    a committed ``"bypass_actors": []`` would read as clean. That is a false
    negative on the single field PROT-02 exists to watch. An absent key is
    therefore a hard failure, never an empty list. Verified empirically on
    2026-07-29: a token minted at ``administration: read`` does NOT receive the
    key on either repo, and ``bypass_actors`` is the only key that differs
    between a read-scoped and a write-scoped read.

    Nothing here is tuned to make the surviving-difference list come out empty.

    Parameters
    ----------
    live
        The ruleset payload as read from the API.
    committed
        The payload parsed from ``.github/rulesets/*.json``.

    Returns
    -------
    tuple
        The normalised live payload, the normalised committed payload, and the
        list of human-readable removal records.

    Raises
    ------
    RulesetReadError
        When the live payload carries no ``bypass_actors`` key.
    """
    if BYPASS_ACTORS_KEY not in live:
        # The repo is named from the payload's own `source` field rather than from
        # a third parameter, so the two-argument signature every caller shares is
        # preserved. `source` is a live-only envelope key and survives the
        # read-scoped read that triggers this guard -- verified 2026-07-29:
        # `bypass_actors` is the ONLY key missing at administration:read.
        repo = live.get("source") or "<unknown repo>"
        name = live.get("name") or committed.get("name") or "<unnamed>"
        ruleset_id = live.get("id", "<no id>")
        raise RulesetReadError(
            f"{repo} ruleset `{name}` (id {ruleset_id}): the live payload has no `{BYPASS_ACTORS_KEY}` "
            f"key — the reading token lacks write access to the ruleset, so this check CANNOT "
            f"see bypass actors and must not report clean. Never substitute an empty list. "
            f"Mint the reading token with `permission-administration: write` (Phase 13 D-16 iii)."
        )

    removed: list[str] = []
    committed_rules = _rules_by_type(committed)
    norm_live = _normalize_side(live, "live", removed, committed_rules)
    norm_committed = _normalize_side(committed, "committed", removed, committed_rules)
    return norm_live, norm_committed, removed


def to_payload(committed: dict[str, Any]) -> dict[str, Any]:
    """Serialise a committed ruleset payload into a valid update request body.

    Distinct from :func:`normalize`, and deliberately so. ``normalize`` re-keys
    ``rules`` from a list into a mapping (rule (b)) purely to make two payloads
    comparable; that output is **not** a valid request body and must never be
    sent. This function is the send path and does the opposite: it preserves the
    list, preserves the committed key order, and removes nothing but the one
    field the API schema rejects.

    Two rules, in the ``normalize`` vocabulary:
      (s1) Emit only :data:`SENDABLE_KEYS`, in the committed order. The envelope
           keys rule (a) drops exist purely on the read side; a body carrying
           them is not what the schema accepts. Sending the complete six-key
           object -- rather than a partial one -- sidesteps the undocumented
           merge-versus-replace semantics for omitted fields.
      (s2) Drop null-valued ``integration_id`` entries from every
           ``required_status_checks`` entry. The API schema types that field as a
           non-nullable ``integer``, so the ``null`` the committed files carry
           yields a 422. The null is deliberate in the committed file and must
           not be "fixed" there -- stripping it here is what keeps the committed
           file readable as "any app may report this context".

    Parameters
    ----------
    committed
        The payload parsed from ``.github/rulesets/*.json``.

    Returns
    -------
    dict
        A request body carrying exactly the sendable keys present in the input.
    """
    payload: dict[str, Any] = {}
    for key in SENDABLE_KEYS:  # (s1)
        if key not in committed:
            continue
        if key != "rules":
            payload[key] = committed[key]
            continue
        rules: list[Any] = []
        for rule in committed["rules"] or []:
            if not isinstance(rule, dict):
                continue
            rule = dict(rule)
            parameters = rule.get("parameters")
            if rule.get("type") == "required_status_checks" and isinstance(parameters, dict):
                parameters = dict(parameters)
                checks = parameters.get("required_status_checks")
                if isinstance(checks, list):
                    parameters["required_status_checks"] = [
                        {k: v for k, v in entry.items() if v is not None}  # (s2)
                        for entry in checks
                        if isinstance(entry, dict)
                    ]
                rule["parameters"] = parameters
            rules.append(rule)
        payload["rules"] = rules
    return payload


def diff(live: Any, committed: Any, path: str = "$") -> list[str]:
    """Compare two normalised payloads recursively; return sorted difference strings.

    An empty list means no drift survives normalisation.

    Parameters
    ----------
    live
        The normalised live value.
    committed
        The normalised committed value.
    path
        JSON-path-ish prefix for the current node; callers pass the default.

    Returns
    -------
    list of str
        One entry per surviving difference, sorted.
    """
    diffs: list[str] = []
    if isinstance(live, dict) and isinstance(committed, dict):
        for key in sorted(set(live) | set(committed)):
            child = f"{path}.{key}"
            if key not in live:
                diffs.append(f"`{child}`: live=<absent> committed={json.dumps(committed[key], sort_keys=True)}")
            elif key not in committed:
                diffs.append(f"`{child}`: live={json.dumps(live[key], sort_keys=True)} committed=<absent>")
            else:
                diffs.extend(diff(live[key], committed[key], child))
    elif isinstance(live, list) and isinstance(committed, list):
        if len(live) != len(committed):
            diffs.append(f"`{path}`: list length live={len(live)} committed={len(committed)}")
        for index, (left, right) in enumerate(zip(live, committed, strict=False)):
            diffs.extend(diff(left, right, f"{path}[{index}]"))
    elif live != committed:
        diffs.append(
            f"`{path}`: live={json.dumps(live, sort_keys=True)} committed={json.dumps(committed, sort_keys=True)}"
        )
    return sorted(diffs)


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------


def _rules_by_type(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index a payload's `rules` list on rule type, for the rule (d) conditional lookup."""
    indexed: dict[str, dict[str, Any]] = {}
    for rule in payload.get("rules") or []:
        if not isinstance(rule, dict) or not isinstance(rule.get("type"), str):
            continue
        parameters = rule.get("parameters")
        indexed[rule["type"]] = dict(parameters) if isinstance(parameters, dict) else {}
    return indexed


def _normalize_side(
    payload: dict[str, Any],
    side: str,
    removed: list[str],
    committed_rules: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Normalise one side of the comparison, appending every removal to `removed`."""
    norm: dict[str, Any] = {}

    for key in sorted(payload):
        if key in LIVE_ONLY_ENVELOPE_KEYS:
            # Rule (a) is unconditional, unlike rule (d): these eight keys have no
            # committed counterpart at all -- they are read-side envelope, not
            # fields the committed file could have an opinion about.
            removed.append(f"[{side}] rule (a) live-only envelope key — dropped top-level `{key}`")
            continue
        if key in ("rules", "conditions"):
            continue
        norm[key] = payload[key]

    conditions = payload.get("conditions")
    if isinstance(conditions, dict):
        norm["conditions"] = _normalize_conditions(conditions)

    norm["rules"] = _normalize_rules(payload, side, removed, committed_rules)
    return norm


def _normalize_rules(
    payload: dict[str, Any],
    side: str,
    removed: list[str],
    committed_rules: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Re-key `rules` from a list onto rule type -- rule (b) -- normalising each rule's parameters.

    The two sides order the list differently (the committed files lead with
    ``pull_request``; the live read leads with ``deletion``), so comparing by
    list position would report a difference that does not exist. Re-keying
    removes no information, which is why it is not recorded as a removal.
    """
    rules: dict[str, Any] = {}
    for rule in payload.get("rules") or []:
        if not isinstance(rule, dict):
            continue
        rule_type = rule.get("type")
        if not isinstance(rule_type, str):
            continue
        parameters = rule.get("parameters")
        parameters = dict(parameters) if isinstance(parameters, dict) else {}
        committed_parameters = committed_rules.get(rule_type, {})

        if rule_type == "pull_request":
            parameters = _drop_keys(
                parameters,
                PULL_REQUEST_READ_FILLED_KEYS,
                committed_parameters,
                removed,
                side,
                "pull_request",
            )
        elif rule_type == "required_status_checks":
            parameters = _drop_keys(
                parameters,
                STATUS_CHECKS_READ_FILLED_KEYS,
                committed_parameters,
                removed,
                side,
                "required_status_checks",
            )
            parameters = _clean_status_checks(parameters, side, removed)
        rules[rule_type] = parameters

    return {key: rules[key] for key in sorted(rules)}


def _clean_status_checks(parameters: dict[str, Any], side: str, removed: list[str]) -> dict[str, Any]:
    """Pop `integration_id` from every status-check entry and sort on context -- rule (c).

    Live returns the integer 15368 (the GitHub Actions app) where the committed
    files carry JSON ``null``, so the field is dropped from both sides and every
    drop is recorded.
    """
    checks = parameters.get("required_status_checks")
    if not isinstance(checks, list):
        return parameters
    out = dict(parameters)
    cleaned: list[dict[str, Any]] = []
    for entry in checks:
        if not isinstance(entry, dict):
            continue
        entry = dict(entry)
        if "integration_id" in entry:
            entry.pop("integration_id")
            removed.append(
                f"[{side}] rule (c) integration_id differs live-vs-committed — "
                f"dropped `integration_id` from required_status_checks entry "
                f"`{entry.get('context')}`"
            )
        cleaned.append(entry)
    out["required_status_checks"] = sorted(cleaned, key=lambda e: str(e.get("context") or ""))
    return out


def _normalize_conditions(conditions: dict[str, Any]) -> dict[str, Any]:
    """Sort `conditions.ref_name.include` / `.exclude` -- normalisation rule (e)."""
    out = dict(conditions)
    ref_name = out.get("ref_name")
    if isinstance(ref_name, dict):
        ref_out = dict(ref_name)
        for key in ("include", "exclude"):
            value = ref_out.get(key)
            if isinstance(value, list):
                ref_out[key] = sorted(str(v) for v in value)
        out["ref_name"] = ref_out
    return out


def _drop_keys(
    parameters: dict[str, Any],
    keys: tuple[str, ...],
    committed_parameters: dict[str, Any],
    removed: list[str],
    side: str,
    where: str,
) -> dict[str, Any]:
    """Drop read-filled `keys`, but only where the committed side is silent — rule (d).

    A key the committed payload sets is a key the committed payload has an
    opinion about, so it stays in the comparison and any difference is reported.
    Every actual removal appends a record naming the rule that did it.
    """
    out = dict(parameters)
    for key in keys:
        if key not in out:
            continue
        if key in committed_parameters:
            continue  # the committed side sets it — compare, do not drop
        out.pop(key)
        removed.append(
            f"[{side}] rule (d) GitHub fills this on read and the committed payload is "
            f"silent about it — dropped `{key}` from {where} parameters"
        )
    return out
