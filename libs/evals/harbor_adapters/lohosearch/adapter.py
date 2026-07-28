"""Generate Harbor tasks from the LoHoSearch benchmark.

LoHoSearch ships its questions and answers as XOR-obfuscated base64 (the
BrowseComp scheme) so the benchmark is not scraped into training data as plain
text. The obfuscation is not a security control -- the key is the `canary`
column shipped alongside -- but the canary does forbid republishing the
plaintext, so generated task directories are git-ignored and rebuilt on demand.

Upstream publishes no ID column, so tasks are bound to rows by
`sha256(question_ciphertext)`. All rows share one canary and the cipher is a
fixed XOR, making each row's ciphertext byte-stable; the hash therefore survives
upstream re-ordering, while an edited question fails to resolve rather than
being silently substituted.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import re
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

_HF_REPO = "meituan-longcat/LoHoSearch"
_CSV_FILENAME = "LoHoSearch.csv"
_CSV_URL = f"https://huggingface.co/datasets/{_HF_REPO}/resolve/main/{_CSV_FILENAME}"
_REQUIRED_COLUMNS = ("question", "answer", "canary")
_CACHE_DIRNAME = ".cache"

# Task directory names are used as Harbor task ids and as filesystem paths.
_TASK_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

# Placeholders substituted into the grader templates. Matched in a single pass so
# a value containing a placeholder string cannot cascade into another slot.
_PLACEHOLDER_RE = re.compile(r"\{(question|response|correct_answer|target|predicted_answer)\}")

# The BrowseComp response format, so answers are shaped the way both graders
# expect. Verbatim from openai/simple-evals `browsecomp_eval.QUERY_TEMPLATE`.
_RESPONSE_FORMAT = """Your response should be in the following format:
Explanation: {your explanation for your final answer}
Exact Answer: {your succinct, final answer}
Confidence: {your confidence score between 0% and 100% for your answer}"""

_DOCKERFILE = """FROM python:3.12-slim

# Pre-install curl at build time (the build phase has network) so the
# in-sandbox agent's runtime bootstrap skips apt.
RUN apt-get update \\
    && apt-get install -y --no-install-recommends curl ca-certificates \\
    && rm -rf /var/lib/apt/lists/*
"""

_DOCKERIGNORE = (
    ".env\n.env.*\n*.pem\n*.key\n*.crt\ncredentials.json\n.git\n__pycache__/\n.venv/\n.DS_Store\n"
)

# Package mirrors plus every model-provider API the scorecard workflow can
# select. The agent phase overrides this with `public` (see `_task_toml`); this
# baseline covers sandbox bootstrap and the verifier's judge calls.
_ALLOWED_HOSTS = (
    "astral.sh",
    "*.astral.sh",
    "github.com",
    "*.githubusercontent.com",
    "pypi.org",
    "*.pythonhosted.org",
    "api.smith.langchain.com",
    "api.anthropic.com",
    "api.openai.com",
    "generativelanguage.googleapis.com",
    "openrouter.ai",
    "*.baseten.co",
    "api.fireworks.ai",
    "ollama.com",
    "api.groq.com",
    "integrate.api.nvidia.com",
    "api.x.ai",
)


@dataclass(frozen=True)
class Row:
    """One decrypted LoHoSearch record plus its content-addressed id."""

    question_sha256: str
    question: str
    answer: str


def templates_dir() -> Path:
    """Return the directory holding the verifier templates.

    Defined as a function (rather than a module-level constant) so tests can
    monkeypatch it to point at a fixture directory.

    Returns:
        Path to the `templates/` directory shipped alongside this module.
    """
    return Path(__file__).resolve().parent / "templates"


def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length XOR key from `password` using SHA256.

    Verbatim from openai/simple-evals `browsecomp_eval.derive_key`.

    Args:
        password: The row's `canary` value.
        length: Number of key bytes required.

    Returns:
        A key of exactly `length` bytes.
    """
    key = hashlib.sha256(password.encode()).digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt one base64 XOR-obfuscated LoHoSearch field.

    Args:
        ciphertext_b64: The base64 field as published.
        password: The row's `canary` value, used in full.

    Returns:
        The decoded UTF-8 plaintext.
    """
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key, strict=False)).decode()


def question_id(question_ciphertext: str) -> str:
    """Return the durable upstream id for a row: sha256 of its question ciphertext.

    Args:
        question_ciphertext: The row's `question` field, still encrypted.

    Returns:
        Lowercase hex sha256 digest.
    """
    return hashlib.sha256(question_ciphertext.encode()).hexdigest()


def fetch_rows(cache_dir: Path, *, refresh: bool = True) -> list[Row]:
    """Download the LoHoSearch CSV and return its decrypted rows.

    The upstream revision is deliberately not pinned so published corrections
    flow through; task identity comes from the content hash instead.

    Args:
        cache_dir: Directory for the downloaded CSV. Created if absent.
        refresh: Re-download even when a cached copy exists.

    Returns:
        Every row in the benchmark split, decrypted.

    Raises:
        ValueError: If the CSV is missing a required column.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / _CSV_FILENAME
    if refresh or not csv_path.is_file():
        # Fixed HTTPS URL to a public dataset; no user- or agent-supplied input.
        with urllib.request.urlopen(_CSV_URL, timeout=120) as response:  # noqa: S310
            csv_path.write_bytes(response.read())
    return parse_rows(csv_path.read_text(encoding="utf-8"))


def parse_rows(csv_text: str) -> list[Row]:
    """Parse and decrypt LoHoSearch CSV text.

    Args:
        csv_text: Raw CSV contents.

    Returns:
        One `Row` per record, in file order.

    Raises:
        ValueError: If a required column is missing.
    """
    reader = csv.DictReader(csv_text.splitlines())
    missing = [column for column in _REQUIRED_COLUMNS if column not in (reader.fieldnames or ())]
    if missing:
        msg = f"LoHoSearch CSV is missing required column(s): {missing}"
        raise ValueError(msg)

    rows: list[Row] = []
    for record in reader:
        canary = record["canary"]
        rows.append(
            Row(
                question_sha256=question_id(record["question"]),
                question=decrypt(record["question"], canary),
                answer=decrypt(record["answer"], canary),
            )
        )
    return rows


def resolve_tasks(manifest: Mapping[str, object], rows: list[Row]) -> dict[str, Row]:
    """Match each manifest entry to an upstream row by content hash.

    Args:
        manifest: Parsed `manifest.json`.
        rows: Rows returned by `fetch_rows`.

    Returns:
        A mapping of task name to its resolved row.

    Raises:
        ValueError: If a task name is unsafe, an entry lacks `question_sha256`,
            or a recorded hash is absent upstream. A missing hash means the
            question was edited or removed, which invalidates any calibration
            recorded against it, so it fails rather than silently substituting.
    """
    tasks = manifest.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        msg = "manifest.json must contain a non-empty `tasks` object"
        raise ValueError(msg)

    by_hash = {row.question_sha256: row for row in rows}
    resolved: dict[str, Row] = {}
    for task_name, entry in tasks.items():
        if not _TASK_NAME_RE.match(task_name) or Path(task_name).name != task_name:
            msg = f"task name {task_name!r} must be a single lowercase path component"
            raise ValueError(msg)
        digest = entry.get("question_sha256") if isinstance(entry, dict) else None
        if not isinstance(digest, str) or not digest:
            msg = f"manifest entry {task_name!r} must declare a `question_sha256`"
            raise ValueError(msg)
        row = by_hash.get(digest)
        if row is None:
            msg = (
                f"task {task_name!r} (question_sha256 {digest[:12]}...) is no longer present "
                f"upstream in {_HF_REPO}. The question was edited or removed, so any calibration "
                f"recorded against it is void; re-select the task deliberately rather than "
                f"substituting a different question."
            )
            raise ValueError(msg)
        resolved[task_name] = row
    return resolved


def generate_task(row: Row, task_name: str, output_dir: Path) -> Path:
    """Write one self-contained Harbor task directory.

    Args:
        row: The decrypted record to render.
        task_name: Directory name for the generated task.
        output_dir: Dataset directory that will contain it.

    Returns:
        Path to the generated task directory.

    Raises:
        ValueError: If `task_name` is unsafe or would escape `output_dir`.
    """
    if not _TASK_NAME_RE.match(task_name) or Path(task_name).name != task_name:
        msg = f"task name {task_name!r} must be a single lowercase path component"
        raise ValueError(msg)

    dataset_root = output_dir.resolve()
    task_dir = dataset_root / task_name
    if task_dir.parent != dataset_root:
        msg = f"task {task_name!r} would escape {dataset_root}"
        raise ValueError(msg)
    if task_dir.exists():
        # Regenerate cleanly so a rerun overwrites instead of failing on existing
        # subdirectories. Safe because the guards above proved `task_name` is a
        # single path component directly under the dataset root.
        shutil.rmtree(task_dir)

    environment_dir = task_dir / "environment"
    environment_dir.mkdir(parents=True)
    (environment_dir / "Dockerfile").write_text(_DOCKERFILE)
    (environment_dir / ".dockerignore").write_text(_DOCKERIGNORE)

    (task_dir / "instruction.md").write_text(
        f"{row.question}\n\n"
        f"{_RESPONSE_FORMAT}\n\n"
        "Write your final answer (and nothing else) to `/app/answer.txt`.\n"
    )

    solution_dir = task_dir / "solution"
    solution_dir.mkdir()
    # The answer comes from a third-party CSV, so it is never interpolated into
    # the shell. Base64 restricts the payload to [A-Za-z0-9+/=], leaving no
    # injection surface regardless of what upstream publishes.
    encoded = base64.b64encode(f"{row.answer}\n".encode()).decode()
    (solution_dir / "solve.sh").write_text(
        f"#!/bin/sh\nset -eu\nprintf '%s' '{encoded}' | base64 -d > /app/answer.txt\n"
    )

    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    copy_verifier_templates(tests_dir)
    # Verifier-only: Harbor mounts tests/ at verify time, after the agent is
    # killed, and the Dockerfile copies no task content, so the answer key is
    # never present in the agent's container.
    (tests_dir / "case.json").write_text(
        json.dumps({"question": row.question, "ground_truth": row.answer}, ensure_ascii=False)
        + "\n"
    )

    (task_dir / "task.toml").write_text(_task_toml(row))
    return task_dir


def populate_tasks(dataset_dir: Path, *, refresh: bool = True) -> int:
    """Generate every task listed in a dataset's `manifest.json`.

    Generated task directories are git-ignored, so this must run before
    `harbor run --path <dataset_dir>`.

    Args:
        dataset_dir: Dataset directory containing `manifest.json`.
        refresh: Re-download the upstream CSV rather than reusing the cache.

    Returns:
        The number of task directories generated.

    Raises:
        FileNotFoundError: If the dataset has no `manifest.json`.
    """
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        msg = f"No manifest at {manifest_path}"
        raise FileNotFoundError(msg)

    manifest = json.loads(manifest_path.read_text())
    rows = fetch_rows(dataset_dir / _CACHE_DIRNAME, refresh=refresh)
    for task_name, row in resolve_tasks(manifest, rows).items():
        generate_task(row, task_name, dataset_dir)
    return len(manifest["tasks"])


def render_grader(template: str, **fields: str) -> str:
    """Substitute values into a grader template in a single pass.

    `str.format` is avoided because the templates contain literal braces and
    because a value containing a placeholder string must not cascade into
    another slot.

    Args:
        template: Grader prompt text containing `{name}` placeholders.
        **fields: Values keyed by placeholder name.

    Returns:
        The rendered prompt, with unknown placeholders left untouched.
    """
    return _PLACEHOLDER_RE.sub(lambda match: fields.get(match.group(1), match.group(0)), template)


def copy_verifier_templates(tests_dir: Path) -> None:
    """Copy the task-invariant verifier files into `tests_dir`.

    `test.sh`, `judge.py`, and the two grader prompts are byte-identical across
    every task, so they are single-sourced in `templates/` and git-ignored per
    task; only `case.json` differs.

    Args:
        tests_dir: The task's `tests/` directory.
    """
    tests_dir.mkdir(parents=True, exist_ok=True)
    source = templates_dir()
    for name in ("test.sh", "judge.py", "browsecomp_grader.txt", "simpleqa_grader.txt"):
        shutil.copy2(source / name, tests_dir / name)


def _task_toml(row: Row) -> str:
    hosts = ", ".join(f'"{host}"' for host in _ALLOWED_HOSTS)
    return (
        'version = "1.3"\n\n'
        "[metadata]\n"
        'source = "lohosearch"\n'
        f'question_sha256 = "{row.question_sha256}"\n\n'
        "[environment]\n"
        # Baseline for sandbox bootstrap and the verifier's judge calls.
        'network_mode = "allowlist"\n'
        f"allowed_hosts = [{hosts}]\n\n"
        "[agent]\n"
        # LoHoSearch is an open-web benchmark and the agent runs inside the
        # sandbox, so the agent phase needs unrestricted egress. On the LangSmith
        # sandbox `public` is also the only mode that grants any egress at all.
        'network_mode = "public"\n'
    )
