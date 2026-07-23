"""Static contracts for credential scoping in GitHub workflows."""

import json
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
APP_TOKEN_WORKFLOWS = (
    "close_unchecked_issues.yml",
    "dcode_release_notes.yml",
    "dependabot_lockfile_fix.yml",
    "pr_labeler.yml",
    "pr_labeler_backfill.yml",
    "tag-external-issues.yml",
)
INTEGRATION_ENV = {
    "ANTHROPIC_API_KEY": "${{ (matrix.working-directory == 'libs/deepagents' || matrix.working-directory == 'libs/partners/quickjs') && secrets.ANTHROPIC_API_KEY || '' }}",
    "DAYTONA_API_KEY": "${{ (matrix.working-directory == 'libs/partners/daytona' || matrix.working-directory == 'libs/code') && secrets.DAYTONA_API_KEY || '' }}",
    "LANGSMITH_API_KEY": "${{ (matrix.working-directory == 'libs/deepagents' || matrix.working-directory == 'libs/code') && secrets.LANGSMITH_API_KEY || '' }}",
    "MODAL_TOKEN_ID": "${{ (matrix.working-directory == 'libs/partners/modal' || matrix.working-directory == 'libs/code') && secrets.MODAL_TOKEN_ID || '' }}",
    "MODAL_TOKEN_SECRET": "${{ (matrix.working-directory == 'libs/partners/modal' || matrix.working-directory == 'libs/code') && secrets.MODAL_TOKEN_SECRET || '' }}",
    "OPENAI_API_KEY": "${{ matrix.working-directory == 'libs/deepagents' && secrets.OPENAI_API_KEY || '' }}",
    "RUNLOOP_API_KEY": "${{ (matrix.working-directory == 'libs/partners/runloop' || matrix.working-directory == 'libs/code') && secrets.RUNLOOP_API_KEY || '' }}",
}
RELEASE_INTEGRATION_ENV = {
    "ANTHROPIC_API_KEY": "${{ (needs.setup.outputs.package == 'deepagents' || needs.setup.outputs.package == 'langchain-quickjs') && secrets.ANTHROPIC_API_KEY || '' }}",
    "DAYTONA_API_KEY": "${{ needs.setup.outputs.package == 'langchain-daytona' && secrets.DAYTONA_API_KEY || '' }}",
    "LANGSMITH_API_KEY": "${{ needs.setup.outputs.package == 'deepagents' && secrets.LANGSMITH_API_KEY || '' }}",
    "MODAL_TOKEN_ID": "${{ needs.setup.outputs.package == 'langchain-modal' && secrets.MODAL_TOKEN_ID || '' }}",
    "MODAL_TOKEN_SECRET": "${{ needs.setup.outputs.package == 'langchain-modal' && secrets.MODAL_TOKEN_SECRET || '' }}",
    "OPENAI_API_KEY": "${{ needs.setup.outputs.package == 'deepagents' && secrets.OPENAI_API_KEY || '' }}",
    "RUNLOOP_API_KEY": "${{ needs.setup.outputs.package == 'langchain-runloop' && secrets.RUNLOOP_API_KEY || '' }}",
    "VERCEL_TOKEN": "${{ needs.setup.outputs.package == 'langchain-vercel-sandbox' && secrets.VERCEL_TOKEN || '' }}",
}
OVERRIDE_ONLY_INTEGRATION_TARGETS = {
    "libs/code",
    "libs/partners/quickjs",
}


def _load_workflow(name: str) -> dict:
    workflow = yaml.safe_load((WORKFLOWS / name).read_text())
    workflow["on"] = workflow.pop(True, workflow.get("on"))
    return workflow


def _find_step(workflow: dict, *, job: str, name: str) -> dict:
    matches = [
        step for step in workflow["jobs"][job]["steps"] if step.get("name") == name
    ]
    assert len(matches) == 1, (
        f"Expected one {name!r} step in {job!r}, got {len(matches)}"
    )
    return matches[0]


def test_github_app_client_id_uses_repository_variable() -> None:
    for name in APP_TOKEN_WORKFLOWS:
        workflow = _load_workflow(name)
        token_steps = [
            step
            for job in workflow["jobs"].values()
            for step in job["steps"]
            if str(step.get("uses", "")).startswith("actions/create-github-app-token@")
        ]
        assert token_steps, f"No GitHub App token step found in {name}"
        for step in token_steps:
            assert step["with"]["client-id"] == (
                "${{ vars.ORG_MEMBERSHIP_APP_CLIENT_ID }}"
            )
            assert step["with"]["private-key"] == (
                "${{ secrets.ORG_MEMBERSHIP_APP_PRIVATE_KEY }}"
            )


def test_integration_credentials_are_scoped_by_package() -> None:
    workflow = _load_workflow("integration_tests.yml")
    run_step = _find_step(
        workflow,
        job="integration-tests",
        name="🚀 Run Integration Tests",
    )
    assert run_step["env"] == INTEGRATION_ENV


def test_integration_secret_targets_are_selectable_or_explicit_overrides() -> None:
    workflow = _load_workflow("integration_tests.yml")
    options = set(
        workflow["on"]["workflow_dispatch"]["inputs"]["working-directory"]["options"]
    )
    selectable_targets = options - {"all"}
    assert set(json.loads(workflow["env"]["DEFAULT_LIBS"])) == selectable_targets

    condition_targets = {
        target
        for expression in INTEGRATION_ENV.values()
        for target in re.findall(r"matrix\.working-directory == '([^']+)'", expression)
    }
    assert condition_targets - selectable_targets == OVERRIDE_ONLY_INTEGRATION_TARGETS
    for target in condition_targets:
        assert (ROOT / target).is_dir(), (
            f"Credential condition targets missing path: {target}"
        )


def test_release_keeps_disabled_package_scoped_integration_wiring() -> None:
    workflow = _load_workflow("release.yml")
    run_step = _find_step(
        workflow,
        job="pre-release-checks",
        name="Run integration tests",
    )
    assert run_step["if"] is False
    assert run_step["env"] == RELEASE_INTEGRATION_ENV


def test_openwiki_uses_dedicated_environment() -> None:
    workflow = _load_workflow("openwiki-update.yml")
    assert workflow["jobs"]["update"]["environment"] == "openwiki"
