#!/usr/bin/env python3
# .github/scripts/test_preflight_ruleset_apply.py
# Source: Phase 13 CONTEXT.md D-03 (job-name cross-reference gate) + D-01 (apply path)
"""Unit tests for the D-03 apply preflight.

Every case here is one of the behaviours the gate exists to guarantee: that it
refuses in BOTH directions (a context nothing produces, and a job removed while a
payload still requires it), that a refusal leaves no send payload behind for the
workflow's send step to pick up, that an unreadable workflow is a violation
rather than a skip, and that the payload it does emit is a schema-valid request
body rather than a comparison artefact.

Run with ``python -m pytest .github/scripts/test_preflight_ruleset_apply.py -q``
from the repository root.
"""

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import preflight_ruleset_apply as preflight  # noqa: E402  (path shim must precede the import)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def make_payload(
    name: str = "protect-main",
    branch: str = "refs/heads/main",
    contexts: tuple[str, ...] = ("Lint (pre-commit)", "Tests (pytest)"),
) -> dict[str, object]:
    """Build a committed-shaped ruleset payload carrying `contexts`.

    Parameters
    ----------
    name
        The ruleset name.
    branch
        The single included ref name.
    contexts
        Required status-check context strings.

    Returns
    -------
    dict
        A payload with exactly the six sendable keys, in the committed order.
    """
    return {
        "name": name,
        "target": "branch",
        "enforcement": "active",
        "conditions": {"ref_name": {"include": [branch], "exclude": []}},
        "bypass_actors": [],
        "rules": [
            {"type": "pull_request", "parameters": {"required_approving_review_count": 0}},
            {
                "type": "required_status_checks",
                "parameters": {
                    "strict_required_status_checks_policy": True,
                    "required_status_checks": [{"context": c, "integration_id": None} for c in contexts],
                },
            },
            {"type": "non_fast_forward"},
        ],
    }


def write_payload(rulesets_dir: pathlib.Path, filename: str, payload: dict[str, object]) -> pathlib.Path:
    """Write `payload` into `rulesets_dir` as `filename` and return the path.

    Parameters
    ----------
    rulesets_dir
        Directory to create and write into.
    filename
        Basename of the committed payload.
    payload
        The payload object.

    Returns
    -------
    pathlib.Path
        The written file's path.
    """
    rulesets_dir.mkdir(parents=True, exist_ok=True)
    path = rulesets_dir / filename
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_workflow(workflow_dir: pathlib.Path, filename: str, text: str) -> pathlib.Path:
    """Write a raw workflow document into `workflow_dir`.

    Parameters
    ----------
    workflow_dir
        Directory to create and write into.
    filename
        Basename of the workflow file.
    text
        Raw YAML text.

    Returns
    -------
    pathlib.Path
        The written file's path.
    """
    workflow_dir.mkdir(parents=True, exist_ok=True)
    path = workflow_dir / filename
    path.write_text(text, encoding="utf-8")
    return path


CI_YML = """\
name: CI
on:
  pull_request:
jobs:
  lint:
    name: Lint (pre-commit)
    runs-on: ubuntu-latest
    steps:
      - run: 'true'
  tests:
    name: Tests (pytest)
    runs-on: ubuntu-latest
    steps:
      - run: 'true'
"""

CI_YML_WITHOUT_TESTS = """\
name: CI
on:
  pull_request:
jobs:
  lint:
    name: Lint (pre-commit)
    runs-on: ubuntu-latest
    steps:
      - run: 'true'
"""

UNNAMED_JOB_YML = """\
name: Canary
on:
  workflow_dispatch:
jobs:
  canary:
    runs-on: ubuntu-latest
    steps:
      - run: 'true'
"""

NO_TRIGGER_YML = """\
name: Broken
jobs:
  lint:
    name: Lint (pre-commit)
    runs-on: ubuntu-latest
    steps:
      - run: 'true'
"""


def test_a_matching_payload_exits_zero_and_writes_the_send_payload(tmp_path: pathlib.Path) -> None:
    """A payload whose contexts all match job names passes and emits a send payload."""
    rulesets = tmp_path / "rulesets"
    payload_path = write_payload(rulesets, "main.json", make_payload())
    workflows = tmp_path / "wf"
    write_workflow(workflows, "ci.yml", CI_YML)
    out = tmp_path / "send.json"

    status = preflight.main([str(payload_path), str(workflows), str(out)], rulesets_dir=rulesets)

    assert status == 0
    assert out.exists()


def test_the_ok_line_names_the_payload_the_branch_and_the_context_count(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A clean preflight prints one trailing OK line carrying its four facts."""
    rulesets = tmp_path / "rulesets"
    payload_path = write_payload(rulesets, "main.json", make_payload())
    workflows = tmp_path / "wf"
    write_workflow(workflows, "ci.yml", CI_YML)
    out = tmp_path / "send.json"

    preflight.main([str(payload_path), str(workflows), str(out)], rulesets_dir=rulesets)

    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith("preflight_ruleset_apply: OK")
    assert "main.json" in line
    assert "refs/heads/main" in line
    assert "2" in line
    assert str(out) in line


def test_a_context_no_job_produces_is_refused_and_named(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A required context outside the matchable set exits non-zero and is named."""
    rulesets = tmp_path / "rulesets"
    payload = make_payload(contexts=("Lint (pre-commit)", "Tests (pytest)", "Docs (sphinx -W)"))
    payload_path = write_payload(rulesets, "main.json", payload)
    workflows = tmp_path / "wf"
    write_workflow(workflows, "ci.yml", CI_YML)
    out = tmp_path / "send.json"

    status = preflight.main([str(payload_path), str(workflows), str(out)], rulesets_dir=rulesets)

    captured = capsys.readouterr()
    assert status == 1
    assert "Docs (sphinx -W)" in captured.out
    assert "main.json" in captured.out
    assert "refs/heads/main" in captured.out


def test_every_offending_context_is_named_in_one_run(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Violations accumulate: two missing contexts are both reported before the single exit."""
    rulesets = tmp_path / "rulesets"
    payload = make_payload(contexts=("Docs (sphinx -W)", "Tests (pytest, GPU)"))
    payload_path = write_payload(rulesets, "main.json", payload)
    workflows = tmp_path / "wf"
    write_workflow(workflows, "ci.yml", CI_YML)
    out = tmp_path / "send.json"

    status = preflight.main([str(payload_path), str(workflows), str(out)], rulesets_dir=rulesets)

    captured = capsys.readouterr()
    assert status == 1
    assert "Docs (sphinx -W)" in captured.out
    assert "Tests (pytest, GPU)" in captured.out


def test_a_removed_job_produces_the_same_refusal(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A job deleted from the target tip while the payload still requires it is refused."""
    rulesets = tmp_path / "rulesets"
    payload_path = write_payload(rulesets, "main.json", make_payload())
    workflows = tmp_path / "wf"
    write_workflow(workflows, "ci.yml", CI_YML_WITHOUT_TESTS)
    out = tmp_path / "send.json"

    status = preflight.main([str(payload_path), str(workflows), str(out)], rulesets_dir=rulesets)

    captured = capsys.readouterr()
    assert status == 1
    assert "Tests (pytest)" in captured.out
    assert not out.exists()


def test_a_refusal_writes_no_output_file(tmp_path: pathlib.Path) -> None:
    """A refusal leaves no send payload for the workflow's send step to find."""
    rulesets = tmp_path / "rulesets"
    payload_path = write_payload(rulesets, "main.json", make_payload(contexts=("Nothing produces this",)))
    workflows = tmp_path / "wf"
    write_workflow(workflows, "ci.yml", CI_YML)
    out = tmp_path / "send.json"

    assert preflight.main([str(payload_path), str(workflows), str(out)], rulesets_dir=rulesets) == 1
    assert not out.exists()


def test_a_stale_output_file_is_removed_by_a_refusal(tmp_path: pathlib.Path) -> None:
    """A pre-existing send payload does not survive a refusal."""
    rulesets = tmp_path / "rulesets"
    payload_path = write_payload(rulesets, "main.json", make_payload(contexts=("Nothing produces this",)))
    workflows = tmp_path / "wf"
    write_workflow(workflows, "ci.yml", CI_YML)
    out = tmp_path / "send.json"
    out.write_text('{"stale": true}\n', encoding="utf-8")

    assert preflight.main([str(payload_path), str(workflows), str(out)], rulesets_dir=rulesets) == 1
    assert not out.exists()


def test_an_unresolvable_trigger_block_is_reported_rather_than_skipped(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A workflow with no `on:` under either key form is a violation, not a pass."""
    rulesets = tmp_path / "rulesets"
    payload_path = write_payload(rulesets, "main.json", make_payload())
    workflows = tmp_path / "wf"
    write_workflow(workflows, "ci.yml", CI_YML)
    write_workflow(workflows, "broken.yml", NO_TRIGGER_YML)
    out = tmp_path / "send.json"

    status = preflight.main([str(payload_path), str(workflows), str(out)], rulesets_dir=rulesets)

    captured = capsys.readouterr()
    assert status == 1
    assert "broken.yml" in captured.out
    assert not out.exists()


def test_a_quoted_on_key_resolves(tmp_path: pathlib.Path) -> None:
    """The quoted `"on":` key form resolves, so a quoted workflow is not reported."""
    rulesets = tmp_path / "rulesets"
    payload_path = write_payload(rulesets, "main.json", make_payload(contexts=("Lint (pre-commit)",)))
    workflows = tmp_path / "wf"
    write_workflow(workflows, "quoted.yml", CI_YML.replace("\non:", '\n"on":'))
    out = tmp_path / "send.json"

    assert preflight.main([str(payload_path), str(workflows), str(out)], rulesets_dir=rulesets) == 0


def test_a_job_without_a_name_contributes_its_job_id(tmp_path: pathlib.Path) -> None:
    """An unnamed job matches on its bare job id, which is what GitHub reports."""
    rulesets = tmp_path / "rulesets"
    payload_path = write_payload(rulesets, "main.json", make_payload(contexts=("canary",)))
    workflows = tmp_path / "wf"
    write_workflow(workflows, "canary.yml", UNNAMED_JOB_YML)
    out = tmp_path / "send.json"

    assert preflight.main([str(payload_path), str(workflows), str(out)], rulesets_dir=rulesets) == 0


def test_the_emitted_payload_has_exactly_the_six_sendable_keys_in_committed_order(tmp_path: pathlib.Path) -> None:
    """The send payload carries the six sendable keys and nothing else, in committed order."""
    rulesets = tmp_path / "rulesets"
    payload_path = write_payload(rulesets, "main.json", make_payload())
    workflows = tmp_path / "wf"
    write_workflow(workflows, "ci.yml", CI_YML)
    out = tmp_path / "send.json"

    preflight.main([str(payload_path), str(workflows), str(out)], rulesets_dir=rulesets)

    emitted = json.loads(out.read_text(encoding="utf-8"))
    assert list(emitted) == ["name", "target", "enforcement", "conditions", "bypass_actors", "rules"]


def test_the_emitted_payload_has_no_null_integration_id(tmp_path: pathlib.Path) -> None:
    """The send payload strips the null integration id the API schema rejects."""
    rulesets = tmp_path / "rulesets"
    payload_path = write_payload(rulesets, "main.json", make_payload())
    workflows = tmp_path / "wf"
    write_workflow(workflows, "ci.yml", CI_YML)
    out = tmp_path / "send.json"

    preflight.main([str(payload_path), str(workflows), str(out)], rulesets_dir=rulesets)

    emitted = json.loads(out.read_text(encoding="utf-8"))
    checks = [
        entry
        for rule in emitted["rules"]
        if rule.get("type") == "required_status_checks"
        for entry in rule["parameters"]["required_status_checks"]
    ]
    assert checks
    assert not any("integration_id" in entry for entry in checks)


def test_the_emitted_payload_keeps_rules_a_list(tmp_path: pathlib.Path) -> None:
    """The send payload's `rules` stays a list -- the comparison mapping is not a valid body."""
    rulesets = tmp_path / "rulesets"
    payload_path = write_payload(rulesets, "main.json", make_payload())
    workflows = tmp_path / "wf"
    write_workflow(workflows, "ci.yml", CI_YML)
    out = tmp_path / "send.json"

    preflight.main([str(payload_path), str(workflows), str(out)], rulesets_dir=rulesets)

    assert isinstance(json.loads(out.read_text(encoding="utf-8"))["rules"], list)


def test_a_payload_outside_the_rulesets_directory_is_refused_before_it_is_read(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A path outside the committed rulesets directory is refused without being parsed."""
    rulesets = tmp_path / "rulesets"
    rulesets.mkdir()
    outside = tmp_path / "outside.json"
    # Deliberately unparseable: if the guard read the file first, the failure
    # would be a JSON error rather than the containment refusal asserted below.
    outside.write_text("this is not json", encoding="utf-8")
    workflows = tmp_path / "wf"
    write_workflow(workflows, "ci.yml", CI_YML)
    out = tmp_path / "send.json"

    status = preflight.main([str(outside), str(workflows), str(out)], rulesets_dir=rulesets)

    captured = capsys.readouterr()
    assert status == 1
    assert "outside.json" in captured.out
    assert "rulesets" in captured.out
    # The refusal is the containment rule, NOT a parse failure -- which is the
    # evidence that the file was never opened.
    assert "could not be read as JSON" not in captured.out
    assert not out.exists()


def test_a_symlink_escaping_the_rulesets_directory_is_refused(tmp_path: pathlib.Path) -> None:
    """Containment resolves symlinks, so a link out of the rulesets directory is refused."""
    rulesets = tmp_path / "rulesets"
    rulesets.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(make_payload()), encoding="utf-8")
    link = rulesets / "sneaky.json"
    link.symlink_to(outside)
    workflows = tmp_path / "wf"
    write_workflow(workflows, "ci.yml", CI_YML)
    out = tmp_path / "send.json"

    assert preflight.main([str(link), str(workflows), str(out)], rulesets_dir=rulesets) == 1
    assert not out.exists()


def test_a_missing_workflow_directory_is_refused(tmp_path: pathlib.Path) -> None:
    """An absent workflow directory cannot be cross-referenced, so it is a refusal."""
    rulesets = tmp_path / "rulesets"
    payload_path = write_payload(rulesets, "main.json", make_payload())
    out = tmp_path / "send.json"

    assert preflight.main([str(payload_path), str(tmp_path / "absent"), str(out)], rulesets_dir=rulesets) == 1
    assert not out.exists()


def test_wrong_argument_count_is_refused(tmp_path: pathlib.Path) -> None:
    """The three-argument invocation contract is enforced."""
    assert preflight.main([str(tmp_path)], rulesets_dir=tmp_path) == 1


def copy_target_tip_workflows(destination: pathlib.Path, *, omit: str | None = None) -> pathlib.Path:
    """Copy this repository's real workflow files into `destination`.

    Stands in for the apply workflow's fetch step, which lists the WHOLE
    ``.github/workflows`` directory at the target ref and reads each entry raw.
    Copying every file rather than a chosen subset is what makes the two cases
    below a measurement of the real fetch-plus-glob behaviour rather than a
    restatement of it.

    Parameters
    ----------
    destination
        Directory to create and populate.
    omit
        A filename to leave out, standing in for a workflow absent from the
        target branch tip.

    Returns
    -------
    pathlib.Path
        The populated directory.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted((REPO_ROOT / ".github/workflows").glob("*.yml")):
        if source.name == omit:
            continue
        (destination / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return destination


def test_the_real_committed_main_payload_resolves_the_anti_skip_context_from_the_target_tip(
    tmp_path: pathlib.Path,
) -> None:
    """The new context resolves from a target tip carrying `integrity.yml`.

    MEASURED, not assumed from how the fetch happens to be written. The apply
    workflow lists the whole workflow directory at the target ref and this
    preflight globs both accepted extensions, so a workflow file present on the
    tip is already in scope -- but that is a property of two pieces of code that
    could each change independently, so it is pinned here. Without this test the
    preflight could quietly stop resolving `Integrity (base-ref)` and refuse a
    correct apply, which under `bypass_actors: []` is a branch nobody can merge.
    """
    workflows = copy_target_tip_workflows(tmp_path / "wf")
    assert (workflows / "integrity.yml").is_file(), "the excepted workflow must be on the tip for this case"
    out = tmp_path / "send.json"

    status = preflight.main([str(REPO_ROOT / ".github/rulesets/main.json"), str(workflows), str(out)])

    assert status == 0
    assert out.exists()
    assert "Integrity (base-ref)" in json.dumps(json.loads(out.read_text(encoding="utf-8")))


def test_the_same_payload_is_refused_when_the_integrity_workflow_is_absent_from_the_tip(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The companion refusal: the payload requires the context, the tip stops producing it.

    This is the ordering constraint made mechanical. `integrity.yml` must be
    merged to the protected branch BEFORE the payload requiring its context is
    applied -- a workflow that is not on the default branch never fires, so the
    context would sit perpetually pending and, under an empty bypass-actor list,
    the branch would be unmergeable by anyone including an admin. The preflight
    refusing here is what stands between a dispatch and that state.
    """
    workflows = copy_target_tip_workflows(tmp_path / "wf", omit="integrity.yml")
    out = tmp_path / "send.json"

    status = preflight.main([str(REPO_ROOT / ".github/rulesets/main.json"), str(workflows), str(out)])

    captured = capsys.readouterr()
    assert status == 1
    assert "Integrity (base-ref)" in captured.out
    assert not out.exists()


def test_the_real_committed_develop_payload_passes_against_the_real_ci_yml(tmp_path: pathlib.Path) -> None:
    """The repository's own develop payload cross-references cleanly against its own `ci.yml`.

    This is the plan's acceptance case run as a test: `develop.json` requires only
    the lint and tests contexts, both of which are job names in the committed
    `ci.yml`.
    """
    workflows = tmp_path / "wf"
    workflows.mkdir()
    (workflows / "ci.yml").write_text(
        (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    out = tmp_path / "send.json"

    status = preflight.main([str(REPO_ROOT / ".github/rulesets/develop.json"), str(workflows), str(out)])

    assert status == 0
    assert json.loads(out.read_text(encoding="utf-8"))["name"] == "protect-develop-gsd"
