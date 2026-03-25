from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from deepagents import create_deep_agent

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from tests.evals.utils import (
    TrajectoryScorer,
    file_contains,
    file_excludes,
    final_text_contains,
    final_text_excludes,
    run_agent,
    tool_call,
)

pytestmark = [pytest.mark.eval_category("unit_test")]


def _skill_content(name: str, description: str, body: str) -> str:
    """Build a minimal SKILL.md string with YAML frontmatter."""
    return f"---\nname: {name}\ndescription: {description}\n---\n\n{body}"


@pytest.mark.langsmith
def test_read_skill_full_content(model: BaseChatModel) -> None:
    """Agent reads a skill's SKILL.md and retrieves specific embedded content."""
    agent = create_deep_agent(model=model, skills=["/skills/user/"])
    run_agent(
        agent,
        model=model,
        initial_files={
            "/skills/user/data-analysis/SKILL.md": _skill_content(
                name="data-analysis",
                description="Step-by-step workflow for analyzing datasets using Lunar tool",
                body="## Steps\n1. Load dataset\n2. Clean data\n3. Explore\n\nMagic number: ALPHA-7-ZULU\n",
            ),
        },
        query="What magic number do i need for explore analysing using lunar?",
        # Step 1: read_file to get the skill content.
        # Step 2: answer with the magic number.
        # 1 tool call request: read_file.
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=1,
                tool_calls=[
                    tool_call(
                        name="read_file",
                        step=1,
                        args_contains={"file_path": "/skills/user/data-analysis/SKILL.md"},
                    )
                ],
            )
            .success(
                final_text_contains("ALPHA-7-ZULU"),
            )
        ),
    )


@pytest.mark.langsmith
def test_read_skill_by_name(model: BaseChatModel) -> None:
    """Agent finds and reads the correct skill by name when multiple skills are available."""
    agent = create_deep_agent(model=model, skills=["/skills/user/"])
    run_agent(
        agent,
        model=model,
        initial_files={
            "/skills/user/code-review/SKILL.md": _skill_content(
                name="code-review",
                description="Guidelines for conducting thorough code reviews",
                body="## Process\n1. Check correctness\n2. Review style\n\nCode: BRAVO-LIMA\n",
            ),
            "/skills/user/deployment/SKILL.md": _skill_content(
                name="deployment",
                description="Steps for deploying applications to production",
                body="## Steps\n1. Build\n2. Test\n3. Deploy\n\nCode: CHARLIE-ECHO\n",
            ),
        },
        query="Read only the code-review skill and tell me the code it contains. Do not read the deployment skill.",
        # Step 1: read_file for code-review only.
        # Step 2: answer with the code.
        # 1 tool call request: read_file (code-review only).
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=1,
                tool_calls=[
                    tool_call(
                        name="read_file",
                        step=1,
                        args_contains={"file_path": "/skills/user/code-review/SKILL.md"},
                    )
                ],
            )
            .success(
                final_text_contains("BRAVO-LIMA"),
                final_text_excludes("CHARLIE-ECHO"),
            )
        ),
    )


@pytest.mark.langsmith
def test_combine_two_skills(model: BaseChatModel) -> None:
    """Agent reads two skills in parallel and combines their information to answer a query."""
    agent = create_deep_agent(model=model, skills=["/skills/user/"])
    run_agent(
        agent,
        model=model,
        initial_files={
            "/skills/user/frontend-deploy/SKILL.md": _skill_content(
                name="frontend-deploy",
                description="Deploy frontend applications to the CDN",
                body="## Steps\n1. Build with npm\n2. Upload to CDN\n\nFrontend port: 3000\n",
            ),
            "/skills/user/backend-deploy/SKILL.md": _skill_content(
                name="backend-deploy",
                description="Deploy backend services via Docker",
                body="## Steps\n1. Build Docker image\n2. Push to registry\n\nBackend port: 8080\n",
            ),
        },
        query=(
            "What ports do the front and backend deploys use? List them as 'frontend: X, backend: Y'."
        ),
        # Step 1: read_file for both skills in parallel.
        # Step 2: answer combining both ports.
        # 2 tool call requests: read_file (frontend-deploy) + read_file (backend-deploy).
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=2,
                tool_calls=[
                    tool_call(
                        name="read_file",
                        step=1,
                        args_contains={"file_path": "/skills/user/frontend-deploy/SKILL.md"},
                    ),
                    tool_call(
                        name="read_file",
                        step=1,
                        args_contains={"file_path": "/skills/user/backend-deploy/SKILL.md"},
                    ),
                ],
            )
            .success(
                final_text_contains("3000"),
                final_text_contains("8080"),
            )
        ),
    )


@pytest.mark.langsmith
def test_update_skill_typo_fix_no_read(model: BaseChatModel) -> None:
    """Agent fixes a known typo with a single direct edit, without reading first."""
    agent = create_deep_agent(model=model, skills=["/skills/user/"])
    run_agent(
        agent,
        model=model,
        initial_files={
            "/skills/user/testing/SKILL.md": _skill_content(
                name="testing",
                description="Guidelines for writing and running tests",
                body="## Steps\n1. Write unit tests\n2. Run test suiet\n3. Check coverage\n",
            ),
        },
        query=(
            "Fix the typo in /skills/user/testing/SKILL.md: replace the exact string 'test suiet' with 'test suite'. "
            "Do not read the file before editing it. Edit the file directly. "
            "After editing, do NOT add any explanation; reply DONE only."
        ),
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=1,
                tool_calls=[
                    tool_call(
                        name="edit_file",
                        step=1,
                        args_contains={"file_path": "/skills/user/testing/SKILL.md"},
                    )
                ],
            )
            .success(
                final_text_contains("DONE"),
                file_excludes("/skills/user/testing/SKILL.md", "test suiet"),
                file_contains("/skills/user/testing/SKILL.md", "test suite"),
            )
        ),
    )


@pytest.mark.langsmith
def test_update_skill_typo_fix_requires_read(model: BaseChatModel) -> None:
    """Agent must read a skill file to discover and fix an unknown typo."""
    agent = create_deep_agent(model=model, skills=["/skills/user/"])
    run_agent(
        agent,
        model=model,
        initial_files={
            "/skills/user/testing/SKILL.md": _skill_content(
                name="testing",
                description="Guidelines for writing and running tests",
                body="## Steps\n1. Write unit tests\n2. Run test suite\n3. Check covreage\n",
            ),
        },
        query=(
            "There is a misspelled word somewhere in /skills/user/testing/SKILL.md. Read the file, identify the typo, and fix it."
        ),
        # Step 1: read_file to discover the typo.
        # Step 2: edit_file to fix it.
        # Step 3: confirm.
        # 2 tool call requests: read_file + edit_file.
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=3,
                tool_call_requests=2,
                tool_calls=[
                    tool_call(
                        name="read_file",
                        step=1,
                        args_contains={"file_path": "/skills/user/testing/SKILL.md"},
                    ),
                    tool_call(
                        name="edit_file",
                        step=2,
                        args_contains={"file_path": "/skills/user/testing/SKILL.md"},
                    ),
                ],
            )
            .success(
                file_excludes("/skills/user/testing/SKILL.md", "covreage"),
                file_contains("/skills/user/testing/SKILL.md", "coverage"),
            )
        ),
    )


@pytest.mark.langsmith
def test_find_skill_in_correct_path(model: BaseChatModel) -> None:
    """Agent uses the skill path shown in the system prompt to update the right skill file.

    Two source paths are configured: /skills/base/ and /skills/project/. The
    deployment skill lives in /skills/project/, the logging skill in /skills/base/.
    The agent must edit the deployment skill without touching the logging skill.
    """
    agent = create_deep_agent(model=model, skills=["/skills/base/", "/skills/project/"])
    run_agent(
        agent,
        model=model,
        initial_files={
            "/skills/base/logging/SKILL.md": _skill_content(
                name="logging",
                description="Structured logging guidelines for all services",
                body="## Guidelines\n1. Use JSON logging\n2. Include request ID\n",
            ),
            "/skills/project/deployment/SKILL.md": _skill_content(
                name="deployment",
                description="Steps for deploying the application to production",
                body="## Steps\n1. Run CI pipeline\n2. Deploy to staging\n3. Deploy to production\n",
            ),
        },
        query=(
            "Update the deployment skill to add a new final step: 'Send Slack notification after deploy'. "
            "The skill path is shown in your system prompt. Edit the file directly."
        ),
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=3,
                tool_call_requests=2,
                tool_calls=[
                    tool_call(
                        name="read_file",
                        step=1,
                        args_contains={"file_path": "/skills/project/deployment/SKILL.md"},
                    ),
                    tool_call(
                        name="edit_file",
                        step=2,
                        args_contains={"file_path": "/skills/project/deployment/SKILL.md"},
                    ),
                ],
            )
            .success(
                file_contains("/skills/project/deployment/SKILL.md", "Slack notification"),
                file_excludes("/skills/base/logging/SKILL.md", "Slack notification"),
            )
        ),
    )
