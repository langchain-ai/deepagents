"""Generate Harbor tasks from Context-Bench filesystem records."""

from __future__ import annotations

import json
import re
import shlex
import shutil
from pathlib import Path

_TASK_ID_RE = re.compile(r"^cb-(?P<suite>[a-z0-9]+)-(?P<index>\d+)$")


def vendor_dir() -> Path:
    """Return the directory containing vendored Context-Bench data.

    Defined as a function (rather than a module-level constant) so tests can
    monkeypatch it to point at a fixture directory.

    Returns:
        Path to the `vendor/` directory shipped alongside this module.
    """
    return Path(__file__).resolve().parent / "vendor"


def parse_task_id(task_id: str) -> tuple[str, int]:
    """Parse a `cb-<suite>-<i>` task id.

    Args:
        task_id: Identifier of the form `cb-<suite>-<i>`, where `<i>` is the
            zero-based line index into `filesystem_<suite>.jsonl`.

    Returns:
        A `(suite, line_index)` tuple.

    Raises:
        ValueError: If `task_id` does not match the expected `cb-<suite>-<i>` form.
    """
    match = _TASK_ID_RE.match(task_id)
    if match is None:
        msg = f"`task_id` {task_id!r} must match `cb-<suite>-<i>` (e.g. `cb-cloud-1`)"
        raise ValueError(msg)
    return match.group("suite"), int(match.group("index"))


def record_for_task_id(task_id: str) -> dict[str, object]:
    """Look up the Context-Bench record identified by a `cb-<suite>-<i>` task id.

    Args:
        task_id: Identifier of the form `cb-<suite>-<i>`.

    Returns:
        The parsed Context-Bench record (a JSON object).

    Raises:
        ValueError: If `task_id` does not match the expected `cb-<suite>-<i>` form.
        FileNotFoundError: If no vendored data exists for the parsed suite.
        IndexError: If the parsed line index does not identify a record.
    """
    suite, line_index = parse_task_id(task_id)
    source_jsonl = vendor_dir() / f"filesystem_{suite}.jsonl"
    if not source_jsonl.is_file():
        msg = f"No vendored Context-Bench data for suite {suite!r} (expected {source_jsonl})"
        raise FileNotFoundError(msg)
    return _read_record(source_jsonl, line_index)


def generate_task(
    *,
    source_jsonl: Path,
    source_files_dir: Path,
    output_dir: Path,
    task_id: str,
    line_index: int,
) -> Path:
    """Generate one self-contained Harbor task from a Context-Bench record.

    Args:
        source_jsonl: JSONL file containing Context-Bench records.
        source_files_dir: Directory containing the complete Context-Bench corpus.
        output_dir: Dataset directory that will contain the generated task.
        task_id: Identifier for the generated task directory.
        line_index: Zero-based record index in `source_jsonl`.

    Returns:
        Path to the generated Harbor task directory.

    Raises:
        TypeError: If the selected record has an unexpected shape.
        ValueError: If `task_id` can escape the output directory.
        IndexError: If `line_index` does not identify a record.
    """
    if Path(task_id).name != task_id:
        msg = "`task_id` must be a single directory name"
        raise ValueError(msg)

    record = _read_record(source_jsonl, line_index)
    task_dir = output_dir / task_id
    files_dir = task_dir / "environment" / "files"
    files_dir.mkdir(parents=True)
    _copy_corpus(source_files_dir, files_dir)

    agent_args = record.get("agent_args")
    question = record.get("input")
    answer = record.get("ground_truth")
    if (
        not isinstance(agent_args, dict)
        or not isinstance(question, str)
        or not isinstance(answer, str)
    ):
        msg = "Context-Bench record has an unexpected shape"
        raise TypeError(msg)
    extra = _extra_mapping(agent_args.get("extra"))

    _write_task_files(task_dir, question, answer, extra)
    return task_dir


def _read_record(source_jsonl: Path, line_index: int) -> dict[str, object]:
    records = [json.loads(line) for line in source_jsonl.read_text().splitlines() if line]
    return records[line_index]


def _extra_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        msg = "Context-Bench record has an unexpected shape"
        raise TypeError(msg)
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _copy_corpus(source_files_dir: Path, destination: Path) -> None:
    for source_file in sorted(source_files_dir.glob("*.txt")):
        shutil.copy2(source_file, destination / source_file.name)


def _write_task_files(
    task_dir: Path,
    question: str,
    answer: str,
    extra: dict[str, object],
) -> None:
    environment_dir = task_dir / "environment"
    (environment_dir / "Dockerfile").write_text(
        "FROM python:3.12-slim\n\n"
        "# Pre-install curl at build time (the build phase has network) so the\n"
        "# in-sandbox agent's runtime bootstrap skips apt; runtime egress is then\n"
        "# all-HTTPS via the task's network allowlist.\n"
        "RUN apt-get update \\\n"
        "    && apt-get install -y --no-install-recommends curl ca-certificates \\\n"
        "    && rm -rf /var/lib/apt/lists/*\n\n"
        "COPY files/ /app/files/\n"
    )
    (environment_dir / ".dockerignore").write_text(
        ".env\n.env.*\n*.pem\n*.key\n*.crt\ncredentials.json\n.git\n__pycache__/\n.venv/\n.DS_Store\n"
    )
    (task_dir / "instruction.md").write_text(
        f"{question}\n\n"
        "Use only the files under `/app/files`. Write your final answer (and nothing else) "
        "to `/app/answer.txt`.\n"
    )

    solution_dir = task_dir / "solution"
    solution_dir.mkdir()
    (solution_dir / "solve.sh").write_text(
        f"#!/bin/sh\nset -eu\nprintf '%s\\n' {shlex.quote(answer)} > /app/answer.txt\n"
    )

    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    expected = shlex.quote(answer.lower())
    (tests_dir / "test.sh").write_text(
        "#!/bin/sh\nset -eu\n"
        "answer=$(tr '[:upper:]' '[:lower:]' < /app/answer.txt | tr -cd '[:alnum:][:space:]')\n"
        f"expected=$(printf '%s' {expected} | tr -cd '[:alnum:][:space:]')\n"
        'if [ "$answer" = "$expected" ]; then\n'
        "  printf '1.0\\n' > /logs/verifier/reward.txt\n"
        "else\n"
        "  printf '0.0\\n' > /logs/verifier/reward.txt\n"
        "fi\n"
    )

    difficulty = _string_extra(extra, "difficulty")
    question_type = _string_extra(extra, "question_type")
    (task_dir / "task.toml").write_text(
        'version = "1.3"\n\n'
        "[metadata]\n"
        'source = "contextbench"\n'
        'suite = "cloud"\n'
        f'difficulty = "{difficulty}"\n'
        f'question_type = "{question_type}"\n\n'
        "[environment]\n"
        # Allowlist (not no-network): the langgraph/dcode agent runs in-sandbox
        # and must reach its own infra (model API + package mirrors) to bootstrap
        # and call the model. Arbitrary web stays blocked, so answer-lookup is
        # still prevented. LangSmith enforces this statically via its egress proxy.
        'network_mode = "allowlist"\n'
        'allowed_hosts = ["astral.sh", "*.astral.sh", "github.com", '
        '"*.githubusercontent.com", "pypi.org", "*.pythonhosted.org", '
        '"api.anthropic.com", "api.smith.langchain.com"]\n'
    )


def _string_extra(extra: dict[str, object], name: str) -> str:
    value = extra.get(name)
    if not isinstance(value, str):
        msg = f"Context-Bench record `agent_args.extra.{name}` must be a string"
        raise TypeError(msg)
    return value
