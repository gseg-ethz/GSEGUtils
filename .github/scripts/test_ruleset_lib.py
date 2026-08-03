#!/usr/bin/env python3
# .github/scripts/test_ruleset_lib.py
# Source: Phase 13 CONTEXT.md D-16 — unit tests pinning ruleset_lib's three guards
"""Unit tests for `ruleset_lib`'s absent-key guard, trigger-key forms and conditional rule (d).

Plain pytest with inline dict fixtures: no network, no filesystem. Run with::

    python -m pytest .github/scripts/test_ruleset_lib.py -q

The module lives alongside the library it tests so that `import ruleset_lib`
resolves the same way it does for the two CLIs in this directory; the `sys.path`
insertion below makes that work when pytest is invoked from the repo root.

Note that ruff's `per-file-ignores` relax docstring rules under `tests/**` only,
which does not cover `.github/scripts/`, so every test here carries a
numpy-convention docstring by design rather than by accident.
"""

import json
import pathlib
import sys

import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import ruleset_lib  # noqa: E402  — deliberately after the sys.path insertion above

REPO = "gseg-ethz/GSEGUtils"
RULESET_NAME = "protect-main"
RULESET_ID = 17762781


def committed_payload(pull_request_parameters: dict | None = None) -> dict:
    """Build a minimal committed ruleset payload in the committed files' shape.

    Parameters
    ----------
    pull_request_parameters
        Extra keys to merge into the ``pull_request`` rule's parameters, used to
        make the committed side non-silent about a read-filled key.

    Returns
    -------
    dict
        A payload carrying exactly the six sendable keys.
    """
    parameters = {"required_approving_review_count": 0}
    parameters.update(pull_request_parameters or {})
    return {
        "name": RULESET_NAME,
        "target": "branch",
        "enforcement": "active",
        "conditions": {"ref_name": {"include": ["refs/heads/main"], "exclude": []}},
        "bypass_actors": [],
        "rules": [
            {"type": "pull_request", "parameters": parameters},
            {
                "type": "required_status_checks",
                "parameters": {
                    "strict_required_status_checks_policy": False,
                    "required_status_checks": [
                        {"context": "Tests (pytest)", "integration_id": None},
                        {"context": "Lint (pre-commit)", "integration_id": None},
                    ],
                },
            },
            {"type": "non_fast_forward"},
        ],
    }


def live_payload(*, with_bypass_actors: bool = True, allowed_merge_methods: list | None = None) -> dict:
    """Build a live ruleset payload in the shape the API actually returns.

    Parameters
    ----------
    with_bypass_actors
        When False, omit the key entirely — the shape a token lacking write
        access to the ruleset receives.
    allowed_merge_methods
        Value for the read-filled ``allowed_merge_methods`` key.

    Returns
    -------
    dict
        A payload carrying the live envelope keys and the read-filled inner keys.
    """
    payload = {
        "id": RULESET_ID,
        "name": RULESET_NAME,
        "target": "branch",
        "source_type": "Repository",
        "source": REPO,
        "enforcement": "active",
        "node_id": "RRS_lACkUmVwbw",
        "_links": {"self": {"href": "https://api.github.com/..."}},
        "created_at": "2026-01-01T00:00:00.000Z",
        "updated_at": "2026-01-01T00:00:00.000Z",
        "current_user_can_bypass": "always",
        "conditions": {"ref_name": {"include": ["refs/heads/main"], "exclude": []}},
        "bypass_actors": [],
        "rules": [
            {"type": "non_fast_forward"},
            {
                "type": "pull_request",
                "parameters": {
                    "required_approving_review_count": 0,
                    "allowed_merge_methods": (
                        allowed_merge_methods if allowed_merge_methods is not None else ["merge", "squash", "rebase"]
                    ),
                    "dismissal_restriction": {},
                    "required_reviewers": [],
                },
            },
            {
                "type": "required_status_checks",
                "parameters": {
                    "do_not_enforce_on_create": False,
                    "strict_required_status_checks_policy": False,
                    "required_status_checks": [
                        {"context": "Lint (pre-commit)", "integration_id": 15368},
                        {"context": "Tests (pytest)", "integration_id": 15368},
                    ],
                },
            },
        ],
    }
    if not with_bypass_actors:
        del payload["bypass_actors"]
    return payload


def api_error_envelope(*, status: str, message: str) -> dict:
    """Build a GitHub REST error body in the shape a refused ruleset read returns.

    Note that `status` is a JSON *string* in GitHub's error bodies, not an integer;
    the fixture keeps that as measured rather than tidying it into an int.

    Parameters
    ----------
    status
        The HTTP status GitHub reports inside the body.
    message
        The human-readable refusal text.

    Returns
    -------
    dict
        An envelope carrying no ``id`` and no ``rules``, which is what makes it
        structurally distinguishable from a ruleset.
    """
    return {
        "message": message,
        "documentation_url": "https://docs.github.com/rest/repos/rules#create-a-repository-ruleset",
        "status": status,
    }


# The refusal the D-22 probe measured verbatim against a private repository in a
# free-plan organisation. Recorded here as a fixture so the guard is tested against
# the body GitHub actually sends rather than a paraphrase of it.
MEASURED_REFUSAL_MESSAGE = "Upgrade to GitHub Pro or make this repository public to enable this feature."


def test_api_error_envelope_with_the_measured_status_names_the_plan_and_visibility_cause() -> None:
    """The refusal D-22 measured raises, quoting status and message and naming the real cause."""
    envelope = api_error_envelope(status="403", message=MEASURED_REFUSAL_MESSAGE)
    with pytest.raises(ruleset_lib.RulesetReadError) as caught:
        ruleset_lib.assert_not_api_error(envelope, "protect-main")
    message = str(caught.value)
    assert "protect-main" in message, message
    assert '"403"' in message, message
    assert MEASURED_REFUSAL_MESSAGE in message, message
    assert "plan and visibility" in message, message
    assert "NOT an absence of drift" in message, message
    # Distinguishable in text from the absent-bypass-actors failure, which is the
    # other RulesetReadError this module raises.
    assert ruleset_lib.BYPASS_ACTORS_KEY not in message, message


def test_api_error_envelope_with_another_status_raises_and_names_that_status() -> None:
    """A refusal carrying a different status still raises, and diagnoses that status instead."""
    envelope = api_error_envelope(status="404", message="Not Found")
    with pytest.raises(ruleset_lib.RulesetReadError) as caught:
        ruleset_lib.assert_not_api_error(envelope, "protect-develop-gsd")
    message = str(caught.value)
    assert '"404"' in message, message
    assert "status 404" in message, message
    assert "must not report clean" in message, message
    assert "plan and visibility" not in message, message


def test_a_real_ruleset_payload_passes_the_envelope_guard_and_still_normalises() -> None:
    """The passing control: a real payload does not raise, and comparison still works after it."""
    live = live_payload()
    committed = committed_payload()
    ruleset_lib.assert_not_api_error(live, "protect-main")
    ruleset_lib.assert_not_api_error(committed, ".github/rulesets/main.json")
    norm_live, norm_committed, _ = ruleset_lib.normalize(live, committed)
    assert ruleset_lib.diff(norm_live, norm_committed) == []


def test_absent_bypass_actors_raises_and_names_repo_and_ruleset() -> None:
    """An absent bypass-actors key is a hard failure naming the repo and the ruleset."""
    with pytest.raises(ruleset_lib.RulesetReadError) as caught:
        ruleset_lib.normalize(live_payload(with_bypass_actors=False), committed_payload())
    message = str(caught.value)
    assert REPO in message, message
    assert RULESET_NAME in message, message
    assert str(RULESET_ID) in message, message


def test_absent_bypass_actors_never_compares_equal_to_committed_empty_list() -> None:
    """The guard fires instead of letting an absent key normalise into a clean verdict."""
    live = live_payload(with_bypass_actors=False)
    committed = committed_payload()
    assert committed["bypass_actors"] == []
    with pytest.raises(ruleset_lib.RulesetReadError):
        ruleset_lib.normalize(live, committed)


def test_explicit_empty_bypass_actors_compares_equal_without_raising() -> None:
    """An explicitly empty bypass-actors list is real data and compares clean."""
    norm_live, norm_committed, _ = ruleset_lib.normalize(live_payload(), committed_payload())
    assert norm_live["bypass_actors"] == []
    assert ruleset_lib.diff(norm_live, norm_committed) == []


def test_trigger_block_resolves_the_quoted_key_form() -> None:
    """A workflow keyed by the quoted string form resolves and is marked quoted."""
    document = yaml.safe_load('"on":\n  push:\n    branches: [main]\n')
    block, form = ruleset_lib.trigger_block(document)
    assert form == "quoted"
    assert "push" in block


def test_trigger_block_resolves_the_bare_boolean_key_form() -> None:
    """A workflow keyed by the bare form PyYAML turns into a boolean resolves and is marked bare."""
    document = yaml.safe_load("on:\n  push:\n    branches: [main]\n")
    assert True in document, "fixture must exercise the YAML 1.1 boolean-key path"
    block, form = ruleset_lib.trigger_block(document)
    assert form == "bare"
    assert "push" in block


def test_trigger_block_returns_empty_when_neither_key_form_is_present() -> None:
    """Neither key form present returns a two-tuple of Nones rather than raising."""
    document = yaml.safe_load("name: CI\njobs:\n  lint:\n    name: Lint (pre-commit)\n")
    assert ruleset_lib.trigger_block(document) == (None, None)


def test_job_without_a_name_yields_its_bare_job_id() -> None:
    """An unnamed job reports its bare job id, with no rendering suffix appended."""
    document = yaml.safe_load("jobs:\n  canary:\n    runs-on: ubuntu-latest\n  lint:\n    name: Lint (pre-commit)\n")
    assert ruleset_lib.job_names(document) == ["canary", "Lint (pre-commit)"]


def test_read_filled_key_is_dropped_when_the_committed_side_is_silent() -> None:
    """Rule (d) drops a read-filled key the committed payload does not set, and records it."""
    norm_live, _, removed = ruleset_lib.normalize(live_payload(), committed_payload())
    assert "allowed_merge_methods" not in norm_live["rules"]["pull_request"]
    assert any("allowed_merge_methods" in record for record in removed)


def test_read_filled_key_survives_when_the_committed_side_sets_it() -> None:
    """Rule (d) compares a read-filled key the committed payload sets, and reports the difference."""
    live = live_payload(allowed_merge_methods=["merge", "squash", "rebase"])
    committed = committed_payload({"allowed_merge_methods": ["squash"]})
    norm_live, norm_committed, removed = ruleset_lib.normalize(live, committed)
    assert norm_live["rules"]["pull_request"]["allowed_merge_methods"] == ["merge", "squash", "rebase"]
    assert not any("allowed_merge_methods" in record for record in removed)
    differences = ruleset_lib.diff(norm_live, norm_committed)
    assert any("allowed_merge_methods" in difference for difference in differences), differences


def test_every_removal_is_recorded_in_the_removal_list() -> None:
    """No key leaves the comparison without an audit line naming the rule that removed it."""
    _, _, removed = ruleset_lib.normalize(live_payload(), committed_payload())
    for key in ruleset_lib.LIVE_ONLY_ENVELOPE_KEYS:
        assert any(f"`{key}`" in record for record in removed), key
    assert all(record.startswith(("[live]", "[committed]")) for record in removed)


def test_to_payload_emits_exactly_the_sendable_keys_and_no_envelope_keys() -> None:
    """The send path returns the six sendable keys in committed order and nothing else."""
    payload = ruleset_lib.to_payload(committed_payload())
    assert list(payload) == list(ruleset_lib.SENDABLE_KEYS)
    for key in ruleset_lib.LIVE_ONLY_ENVELOPE_KEYS:
        assert key not in payload


def test_to_payload_keeps_rules_a_list_and_strips_null_integration_ids() -> None:
    """The send path preserves the rules list and removes the null the API schema rejects."""
    payload = ruleset_lib.to_payload(committed_payload())
    assert isinstance(payload["rules"], list)
    checks = [
        check
        for rule in payload["rules"]
        if rule.get("type") == "required_status_checks"
        for check in rule["parameters"]["required_status_checks"]
    ]
    assert checks, "fixture must carry required status checks"
    assert all("integration_id" not in check for check in checks)
    assert json.dumps(payload)  # a valid request body must be JSON-serialisable


def test_required_contexts_returns_sorted_context_strings() -> None:
    """Required contexts come back sorted, independent of the order in the payload."""
    assert ruleset_lib.required_contexts(committed_payload()) == [
        "Lint (pre-commit)",
        "Tests (pytest)",
    ]
