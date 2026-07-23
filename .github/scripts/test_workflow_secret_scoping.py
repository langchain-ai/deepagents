"""Static contracts for credential scoping in GitHub workflows."""

from pathlib import Path


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


def test_github_app_client_id_uses_repository_variable() -> None:
    for name in APP_TOKEN_WORKFLOWS:
        workflow = (WORKFLOWS / name).read_text()
        assert "secrets.ORG_MEMBERSHIP_APP_CLIENT_ID" not in workflow
        assert "vars.ORG_MEMBERSHIP_APP_CLIENT_ID" in workflow


def test_integration_credentials_are_scoped_by_package() -> None:
    workflow = (WORKFLOWS / "integration_tests.yml").read_text()
    run_step = workflow.split('- name: "🚀 Run Integration Tests"', maxsplit=1)[1]
    run_step = run_step.split("      - name:", maxsplit=1)[0]

    expected = {
        "ANTHROPIC_API_KEY": "matrix.working-directory == 'libs/deepagents' || matrix.working-directory == 'libs/partners/quickjs'",
        "DAYTONA_API_KEY": "matrix.working-directory == 'libs/partners/daytona' || matrix.working-directory == 'libs/code'",
        "LANGSMITH_API_KEY": "matrix.working-directory == 'libs/deepagents' || matrix.working-directory == 'libs/code'",
        "MODAL_TOKEN_ID": "matrix.working-directory == 'libs/partners/modal' || matrix.working-directory == 'libs/code'",
        "MODAL_TOKEN_SECRET": "matrix.working-directory == 'libs/partners/modal' || matrix.working-directory == 'libs/code'",
        "OPENAI_API_KEY": "matrix.working-directory == 'libs/deepagents'",
        "RUNLOOP_API_KEY": "matrix.working-directory == 'libs/partners/runloop' || matrix.working-directory == 'libs/code'",
    }
    for secret, condition in expected.items():
        assert f"{secret}: ${{{{" in run_step
        assert condition in run_step
        assert f"secrets.{secret}" in run_step


def test_release_keeps_disabled_package_scoped_integration_wiring() -> None:
    workflow = (WORKFLOWS / "release.yml").read_text()
    run_step = workflow.split("      - name: Run integration tests", maxsplit=1)[1]
    run_step = run_step.split("      working-directory:", maxsplit=1)[0]

    assert "        if: false" in run_step
    expected = {
        "ANTHROPIC_API_KEY": "needs.setup.outputs.package == 'deepagents' || needs.setup.outputs.package == 'langchain-quickjs'",
        "DAYTONA_API_KEY": "needs.setup.outputs.package == 'langchain-daytona'",
        "LANGSMITH_API_KEY": "needs.setup.outputs.package == 'deepagents'",
        "MODAL_TOKEN_ID": "needs.setup.outputs.package == 'langchain-modal'",
        "MODAL_TOKEN_SECRET": "needs.setup.outputs.package == 'langchain-modal'",
        "OPENAI_API_KEY": "needs.setup.outputs.package == 'deepagents'",
        "RUNLOOP_API_KEY": "needs.setup.outputs.package == 'langchain-runloop'",
        "VERCEL_TOKEN": "needs.setup.outputs.package == 'langchain-vercel-sandbox'",
    }
    for secret, condition in expected.items():
        assert f"{secret}: ${{{{" in run_step
        assert condition in run_step
        assert f"secrets.{secret}" in run_step


def test_openwiki_uses_dedicated_environment() -> None:
    workflow = (WORKFLOWS / "openwiki-update.yml").read_text()
    assert (
        "  update:\n    runs-on: ubuntu-latest\n    environment: openwiki\n" in workflow
    )
