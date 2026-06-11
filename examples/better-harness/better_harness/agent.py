"""Outer-loop Deep Agent and proposer workspace helpers."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from better_harness.core import Experiment, Proposal, RunLayout, SplitResult, Variant
from better_harness.patching import build_variant, prepend_pythonpath

DEFAULT_SYSTEM_PROMPT = """You are Better Agent, an outer-loop Deep Agent that improves another agent harness.

Your job is to read eval feedback and edit the provided harness surface files so the next eval run passes more cases.

Rules:
- Edit only files under /current.
- Do not edit train_cases, history, or bookkeeping files except /proposal.md.
- Prefer general harness fixes over case-specific hacks.
- Do not overfit to the visible examples. Infer the broader policy or behavior they expose, then encode that as general instructions, tools, skills, or middleware changes.
- The files under /current are the actual harness surfaces. Edit them as final prompt text, code, or config that the target agent should load during eval.
- If a surface is a code file such as a tool or middleware file, write the real code or registration needed there, not notes or pseudocode.
- If you change tool or middleware behavior, update both the implementation and any registration or wiring surfaces you were given.
- Use surface_manifest.json and task.md to understand how each editable file maps back to the target harness.
- Use the visible failures and the train case files to decide what to change.
- Keep changes concise and coherent.
- Make the smallest set of edits needed for the visible train failures in this iteration.
- Stop as soon as /current and /proposal.md are updated.
- When done, write a short explanation to /proposal.md."""


@dataclass(frozen=True)
class ProposerWorkspace:
    """Materialized workspace for the outer Deep Agent."""

    root: Path
    current_dir: Path
    proposal_file: Path
    surface_files: dict[str, Path]


def build_proposer_workspace(
    *,
    experiment: Experiment,
    current: Variant,
    train_result: SplitResult,
    layout: RunLayout,
    iteration: int,
) -> ProposerWorkspace:
    """Create one proposer workspace for the current iteration."""
    root = layout.proposer_workspace_dir(iteration)
    if root.exists():
        shutil.rmtree(root)
    current_dir = root / "current"
    current_dir.mkdir(parents=True, exist_ok=True)

    surface_files: dict[str, Path] = {}
    manifest: dict[str, dict[str, str]] = {}
    for name, surface in experiment.surfaces.items():
        path = current_dir / surface.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(current.values[name])
        surface_files[name] = path
        manifest[name] = {
            "kind": surface.kind,
            "target": surface.target,
            "file": str(path.relative_to(root)),
        }

    (root / "surface_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write_train_artifacts(
        experiment=experiment,
        train_result=train_result,
        root=root,
    )
    _write_visible_history(layout=layout, root=root)
    _copy_prior_visible_artifacts(layout=layout, root=root, iteration=iteration)
    _write_task_file(
        experiment=experiment,
        current=current,
        train_result=train_result,
        root=root,
    )
    proposal_file = root / "proposal.md"
    proposal_file.write_text(
        "# Proposal\n\n"
        "- Summary:\n"
        "- Why this should help:\n"
        "- Surfaces changed:\n"
    )
    return ProposerWorkspace(
        root=root,
        current_dir=current_dir,
        proposal_file=proposal_file,
        surface_files=surface_files,
    )


def load_candidate_values(*, current: Variant, workspace: ProposerWorkspace) -> dict[str, str]:
    """Load surface values back out of one proposer workspace."""
    values = dict(current.values)
    for name, path in workspace.surface_files.items():
        values[name] = path.read_text().strip()
    return values


def read_proposal_summary(workspace: ProposerWorkspace) -> str:
    """Read the proposer summary if present."""
    if not workspace.proposal_file.exists():
        return ""
    return workspace.proposal_file.read_text().strip()


def propose_variant(
    *,
    experiment: Experiment,
    current: Variant,
    train_result: SplitResult,
    layout: RunLayout,
    iteration: int,
) -> tuple[Proposal, Variant]:
    """Run the outer Deep Agent once and return its candidate variant."""
    workspace = build_proposer_workspace(
        experiment=experiment,
        current=current,
        train_result=train_result,
        layout=layout,
        iteration=iteration,
    )
    final_message = invoke_deepagents_proposer(
        experiment=experiment,
        workspace=workspace,
    )
    values = load_candidate_values(current=current, workspace=workspace)
    changed_surfaces = tuple(
        sorted(
            name
            for name in experiment.surfaces
            if values[name] != current.values[name]
        )
    )
    summary = read_proposal_summary(workspace)
    proposal = Proposal(
        changed_surfaces=changed_surfaces,
        workspace_dir=str(workspace.root),
        summary=summary,
        final_message=final_message,
    )
    candidate = build_variant(
        experiment=experiment,
        label=f"iter-{iteration:03d}",
        values=values,
    )
    (workspace.root / "result.json").write_text(
        json.dumps(
            {
                "proposal": proposal.to_dict(),
                "candidate_variant": candidate.to_dict(),
            },
            indent=2,
        )
        + "\n"
    )
    return proposal, candidate


def invoke_deepagents_proposer(
    *,
    experiment: Experiment,
    workspace: ProposerWorkspace,
) -> str | None:
    """Run the outer Deep Agent against one proposer workspace."""
    deepagents_root = _resolve_deepagents_root(experiment.better_agent_deepagents_root)
    if deepagents_root is not None:
        return _invoke_via_uv_project_with_retries(
            experiment=experiment,
            workspace=workspace,
            deepagents_root=deepagents_root,
        )

    with _deepagents_import_context(experiment.better_agent_deepagents_root):
        try:
            filesystem_module = importlib.import_module("deepagents.backends")
            graph_module = importlib.import_module("deepagents.graph")
            messages_module = importlib.import_module("langchain_core.messages")
        except ImportError as exc:  # pragma: no cover - real env only
            msg = (
                "Could not import deepagents. Install the package or set "
                "better_agent.deepagents_root / DEEPAGENTS_ROOT to a local checkout."
            )
            raise RuntimeError(msg) from exc

        filesystem_backend_cls = filesystem_module.FilesystemBackend
        create_deep_agent = graph_module.create_deep_agent
        human_message_cls = messages_module.HumanMessage
        backend = filesystem_backend_cls(root_dir=str(workspace.root), virtual_mode=True)
        agent = create_deep_agent(
            model=experiment.better_agent_model,
            system_prompt=_compose_system_prompt(experiment),
            backend=backend,
        )
        result = None
        for attempt in range(3):
            try:
                result = agent.invoke(
                    {
                        "messages": [
                            human_message_cls(
                                content=(
                                    "Read /task.md first. Then inspect the current surface files, visible history, and failing "
                                    "train cases, edit only /current, and finish by updating /proposal.md."
                                )
                            )
                        ]
                    },
                    config={"recursion_limit": experiment.better_agent_max_turns},
                )
                break
            except Exception as exc:  # pragma: no cover - real env only
                if attempt == 2 or not _is_transient_model_error(str(exc)):
                    raise
                time.sleep(2 * (attempt + 1))
    if result is None:  # pragma: no cover - defensive fallback
        msg = "outer Deep Agent produced no result"
        raise RuntimeError(msg)
    final_message = _final_ai_message_text(result)
    _write_outer_agent_result(
        workspace_root=workspace.root,
        result=result,
        final_message=final_message,
    )
    return final_message


def _write_train_artifacts(
    *,
    experiment: Experiment,
    train_result: SplitResult,
    root: Path,
) -> None:
    failures_payload = [
        {
            "case_id": outcome.case_id,
            "stratum": outcome.stratum,
            "status": outcome.status,
            "failure_message": outcome.failure_message,
        }
        for outcome in train_result.failing_outcomes()
    ]
    (root / "train_failures.json").write_text(json.dumps(failures_payload, indent=2) + "\n")
    (root / "train_summary.json").write_text(json.dumps(train_result.to_dict(), indent=2) + "\n")

    train_cases_dir = root / "train_cases"
    train_cases_dir.mkdir(parents=True, exist_ok=True)
    if experiment.runner == "pytest":
        project_root = Path(str(experiment.runner_config["project_root"]))
        copied: set[str] = set()
        for case in experiment.cases_for_split("train"):
            rendered = case.render(model=experiment.model)
            file_part = rendered.partition("::")[0]
            if not file_part or file_part in copied:
                continue
            copied.add(file_part)
            source = project_root / file_part
            if source.exists():
                target = train_cases_dir / file_part
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    elif experiment.runner == "harbor":
        tasks_root = Path(str(experiment.runner_config["tasks_root"]))
        for case in experiment.cases_for_split("train"):
            rendered = case.render(model=experiment.model)
            task_dir = tasks_root / rendered
            if task_dir.exists():
                shutil.copytree(task_dir, train_cases_dir / rendered, dirs_exist_ok=True)


def _write_visible_history(*, layout: RunLayout, root: Path) -> None:
    history_dir = root / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[str] = []
    for path in sorted(layout.visible_iterations_dir.glob("*/decision.json")):
        payload = json.loads(path.read_text())
        summaries.append(
            f"- Iteration {payload['iteration']}: {payload['decision']} "
            f"(train {payload['train_passed']}/{payload['train_total']})"
        )
    if not summaries:
        summaries.append("- No previous iterations yet.")
    (history_dir / "visible_history.md").write_text("# Visible History\n\n" + "\n".join(summaries) + "\n")


def _copy_prior_visible_artifacts(*, layout: RunLayout, root: Path, iteration: int) -> None:
    prior_root = root / "history" / "prior_visible"
    prior_root.mkdir(parents=True, exist_ok=True)

    train_root = layout.visible_root / "train"
    if train_root.exists():
        shutil.copytree(train_root, prior_root / "train", dirs_exist_ok=True)

    iterations_root = prior_root / "iterations"
    iterations_root.mkdir(parents=True, exist_ok=True)
    for decision_path in sorted(layout.visible_iterations_dir.glob("*/decision.json")):
        if decision_path.parent.name == f"{iteration:03d}":
            continue
        target_dir = iterations_root / decision_path.parent.name
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(decision_path, target_dir / decision_path.name)
        markdown_path = decision_path.with_suffix(".md")
        if markdown_path.exists():
            shutil.copy2(markdown_path, target_dir / markdown_path.name)
        proposer_workspace = decision_path.parent / "proposer_workspace"
        if proposer_workspace.exists():
            proposer_target = target_dir / "proposer_workspace"
            proposer_target.mkdir(parents=True, exist_ok=True)
            for name in (
                "outer_agent_request.json",
                "outer_agent_result.json",
                "outer_agent_stdout.log",
                "outer_agent_stderr.log",
                "proposal.md",
                "result.json",
                "task.md",
            ):
                source = proposer_workspace / name
                if source.exists():
                    shutil.copy2(source, proposer_target / name)


def _write_task_file(
    *,
    experiment: Experiment,
    current: Variant,
    train_result: SplitResult,
    root: Path,
) -> None:
    surface_lines = [
        f"- `{name}` -> `current/{surface.filename}` ({surface.kind}, target `{surface.target}`)"
        for name, surface in experiment.surfaces.items()
    ]
    failure_lines = [
        f"- `{outcome.case_id}` [{outcome.stratum}]: {outcome.failure_message or outcome.status}"
        for outcome in train_result.failing_outcomes()
    ]
    if not failure_lines:
        failure_lines.append("- No train failures are currently visible.")
    (root / "task.md").write_text(
        "\n".join(
            [
                "# Better Agent Task",
                "",
                "You are improving another agent harness using eval feedback.",
                "",
                "Rules:",
                "- Edit only files under `current/`.",
                "- Do not edit files under `train_cases/`, `history/`, or this task file.",
                "- Prefer general harness improvements over task-specific hacks.",
                "- Do not overfit to the visible examples. Infer the broader policy they suggest and encode that policy into the harness.",
                "- Treat files under `current/` as the real harness surfaces. Write final prompt text, code, or config there.",
                "- For code surfaces such as tools or middleware, write the actual code or registration that should run during eval.",
                "- If you change tool or middleware behavior, update both the implementation and any registration or wiring surfaces you were given.",
                "- Use `surface_manifest.json` to understand how each editable file maps back to the target harness.",
                "- Use the visible train failures and train case files to decide what to change.",
                "- Keep changes concise and coherent.",
                "- When you finish, update `proposal.md` with a short summary.",
                "",
                f"Current variant: `{current.key}`",
                f"Current train score: `{train_result.passed}/{train_result.total}`",
                "",
                "Editable surfaces:",
                *surface_lines,
                "",
                "Visible train failures:",
                *failure_lines,
                "",
            ]
        )
        + "\n"
    )


def _compose_system_prompt(experiment: Experiment) -> str:
    if experiment.better_agent_system_prompt:
        return experiment.better_agent_system_prompt.strip() + "\n\n" + DEFAULT_SYSTEM_PROMPT
    return DEFAULT_SYSTEM_PROMPT


def _final_ai_message_text(result: dict[str, Any]) -> str | None:
    messages = result.get("messages", [])
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(str(item) for item in content)
            return None if content is None else str(content)
    return None


def _write_outer_agent_result(
    *,
    workspace_root: Path,
    result: dict[str, Any],
    final_message: str | None,
) -> None:
    payload = {
        "final_message": final_message,
        "result": _jsonify(result),
    }
    (workspace_root / "outer_agent_result.json").write_text(json.dumps(payload, indent=2) + "\n")


def _jsonify(value: Any) -> Any:  # noqa: PLR0911
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonify(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(child) for child in value]
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json")
        except TypeError:
            dumped = value.model_dump()
        return _jsonify(dumped)
    if hasattr(value, "dict"):
        try:
            return _jsonify(value.dict())
        except TypeError:
            return repr(value)
    if hasattr(value, "type"):
        payload: dict[str, Any] = {"type": value.type}
        for name in (
            "id",
            "name",
            "content",
            "content_blocks",
            "tool_calls",
            "invalid_tool_calls",
            "additional_kwargs",
            "response_metadata",
            "usage_metadata",
        ):
            if hasattr(value, name):
                payload[name] = _jsonify(getattr(value, name))
        return payload
    if hasattr(value, "__dict__"):
        return {
            key: _jsonify(child)
            for key, child in vars(value).items()
            if not key.startswith("_")
        }
    return repr(value)


def _invoke_via_uv_project_with_retries(
    *,
    experiment: Experiment,
    workspace: ProposerWorkspace,
    deepagents_root: Path,
) -> str | None:
    last_error: str | None = None
    for attempt in range(3):
        try:
            return _invoke_via_uv_project_once(
                experiment=experiment,
                workspace=workspace,
                deepagents_root=deepagents_root,
            )
        except RuntimeError as exc:
            last_error = str(exc)
            if attempt == 2 or not _is_transient_model_error(last_error):
                raise
            time.sleep(2 * (attempt + 1))
    if last_error is None:
        return None
    raise RuntimeError(last_error)


def _invoke_via_uv_project_once(
    *,
    experiment: Experiment,
    workspace: ProposerWorkspace,
    deepagents_root: Path,
) -> str | None:
    repo_root = Path(__file__).resolve().parents[1]
    project_root = deepagents_root / "libs" / "deepagents"
    if not project_root.exists():
        project_root = deepagents_root

    request_path = workspace.root / "outer_agent_request.json"
    result_path = workspace.root / "outer_agent_result.json"
    request_path.write_text(
        json.dumps(
            {
                "workspace_root": str(workspace.root),
                "model": experiment.better_agent_model,
                "max_turns": experiment.better_agent_max_turns,
                "system_prompt": _compose_system_prompt(experiment),
            },
            indent=2,
        )
        + "\n"
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = prepend_pythonpath([repo_root], env.get("PYTHONPATH"))
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(project_root),
            "python",
            "-m",
            "better_harness.agent",
            str(request_path),
            str(result_path),
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )
    (workspace.root / "outer_agent_stdout.log").write_text(completed.stdout)
    (workspace.root / "outer_agent_stderr.log").write_text(completed.stderr)
    if completed.returncode != 0:
        msg = completed.stderr.strip() or completed.stdout.strip() or "outer Deep Agent subprocess failed"
        raise RuntimeError(msg)
    payload = json.loads(result_path.read_text())
    final_message = payload.get("final_message")
    return None if final_message is None else str(final_message)


def _is_transient_model_error(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in (
            "overloaded",
            "overloaded_error",
            "error code: 529",
            "529 -",
            "rate limit",
            "timeout",
        )
    )


def _resolve_deepagents_root(root: Path | None) -> Path | None:
    resolved_root = root
    if resolved_root is None:
        sibling = Path(__file__).resolve().parents[2] / "deepagents"
        if sibling.exists():
            resolved_root = sibling
    return resolved_root


@contextmanager
def _deepagents_import_context(root: Path | None) -> Iterator[None]:
    paths: list[str] = []
    resolved_root = _resolve_deepagents_root(root)
    if resolved_root is not None:
        package_root = resolved_root / "libs" / "deepagents"
        if package_root.exists():
            paths.append(str(package_root))
        else:
            paths.append(str(resolved_root))
    previous = list(sys.path)
    if paths:
        sys.path[:0] = paths
    try:
        yield
    finally:
        sys.path[:] = previous


def main(argv: list[str] | None = None) -> int:
    """Subprocess entrypoint for the outer Deep Agent."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        msg = "usage: python -m better_harness.agent <request.json> <result.json>"
        raise SystemExit(msg)

    request_path = Path(args[0]).resolve()
    result_path = Path(args[1]).resolve()
    payload = json.loads(request_path.read_text())

    filesystem_module = importlib.import_module("deepagents.backends")
    graph_module = importlib.import_module("deepagents.graph")
    messages_module = importlib.import_module("langchain_core.messages")

    filesystem_backend_cls = filesystem_module.FilesystemBackend
    create_deep_agent = graph_module.create_deep_agent
    human_message_cls = messages_module.HumanMessage

    backend = filesystem_backend_cls(root_dir=str(payload["workspace_root"]), virtual_mode=True)
    agent = create_deep_agent(
        model=str(payload["model"]),
        system_prompt=str(payload["system_prompt"]),
        backend=backend,
    )
    result = agent.invoke(
        {
            "messages": [
                human_message_cls(
                    content=(
                        "Read /task.md first. Then inspect the current surface files and the failing train cases, "
                        "edit only /current, and finish by updating /proposal.md."
                    )
                )
            ]
        },
        config={"recursion_limit": int(payload["max_turns"])},
    )
    final_message = _final_ai_message_text(result)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "final_message": final_message,
                "result": _jsonify(result),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
