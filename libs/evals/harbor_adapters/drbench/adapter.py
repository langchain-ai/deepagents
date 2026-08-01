"""Generate Harbor tasks from DRBench enterprise deep-research records.

DRBench ships each task as a company/persona profile, a deep-research question, a
manifest of enterprise documents spread across four apps, and a set of ground-truth
insights. This adapter targets DRBench's *local* mode: the document corpus is laid
down on the task filesystem rather than served from the upstream Nextcloud /
Mattermost / Roundcube container, so the agent researches over files plus the open
web. See `README.md` in the generated dataset for the runtime contract.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import tarfile
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath

UPSTREAM_REPO = "ServiceNow/drbench"
UPSTREAM_SHA = "0d699ecf6aa96b1de378595b432e9b16a82f0ed9"
_CORPUS_URL = f"https://codeload.github.com/{UPSTREAM_REPO}/tar.gz/{UPSTREAM_SHA}"

# Apps a DRBench file can be served from upstream. Local mode keeps the app as the
# top-level corpus directory so a report can cite a document by its origin, which is
# what the upstream citation extractor expects.
_KNOWN_APPS = frozenset({"nextcloud", "email", "mattermost", "file_system"})

_TASK_ID_RE = re.compile(r"^(?:DR\d{4}|SANITY\d+)$")
_SOURCE_RE = re.compile(r"^drbench/data/tasks/(?P<task_id>[A-Za-z0-9]+)/files/(?P<rest>.+)$")

# The verifier scores recall over the insight-bearing ground truth only; upstream's
# `QASimilarityV2` filters `qa_type == "insight"` and ignores distractor entries.
_INSIGHT_QA_TYPE = "insight"

_CONFIG_FILENAMES = ("task.json", "env.json", "eval.json", "info.json")


def vendor_dir() -> Path:
    """Return the directory containing vendored DRBench task configs.

    Defined as a function (rather than a module-level constant) so tests can
    monkeypatch it to point at a fixture directory.

    Returns:
        Path to the `vendor/` directory shipped alongside this module.
    """
    return Path(__file__).resolve().parent / "vendor"


def _templates_dir() -> Path:
    """Return the directory holding the verifier templates (`test.sh`, `judge.py`)."""
    return Path(__file__).resolve().parent / "templates"


def parse_task_id(task_id: str) -> str:
    """Validate a DRBench task id.

    Args:
        task_id: Identifier of the form `DR0001` or `SANITY0`.

    Returns:
        The validated `task_id`.

    Raises:
        ValueError: If `task_id` is not a bare DRBench id. Rejecting anything that
            is not a single path component keeps the id safe to join onto an
            output directory.
    """
    if _TASK_ID_RE.match(task_id) is None or Path(task_id).name != task_id:
        msg = f"`task_id` {task_id!r} must be a DRBench id such as `DR0001` or `SANITY0`"
        raise ValueError(msg)
    return task_id


def available_task_ids() -> list[str]:
    """Return the benchmark's task ids, sorted.

    Upstream also ships `SANITY0`, a single-document smoke task used to check a
    local install. It is excluded here so `--all` yields exactly the 100 scored
    DRBench tasks and cannot skew a dataset average; generate it explicitly with
    `--task-ids SANITY0` when debugging.

    Returns:
        Sorted `DR<nnnn>` task ids that have a vendored config.

    Raises:
        FileNotFoundError: If the vendored configs are missing.
    """
    tasks_root = vendor_dir() / "tasks"
    if not tasks_root.is_dir():
        msg = f"No vendored DRBench configs at {tasks_root}"
        raise FileNotFoundError(msg)
    return sorted(
        entry.name
        for entry in tasks_root.iterdir()
        if entry.is_dir()
        and entry.name.startswith("DR")
        and (entry / "task.json").is_file()
    )


def record_for_task_id(task_id: str) -> dict[str, dict]:
    """Load the vendored DRBench config bundle for one task.

    Args:
        task_id: Identifier of the form `DR0001` or `SANITY0`.

    Returns:
        A mapping with `task`, `env`, `eval`, and `info` keys holding the parsed
        upstream config JSON.

    Raises:
        ValueError: If `task_id` is not a bare DRBench id.
        FileNotFoundError: If no vendored config exists for `task_id`.
        TypeError: If a config file does not hold a JSON object.
    """
    parse_task_id(task_id)
    task_root = vendor_dir() / "tasks" / task_id
    record: dict[str, dict] = {}
    for filename in _CONFIG_FILENAMES:
        path = task_root / filename
        if not path.is_file():
            msg = f"No vendored DRBench config for {task_id!r} (expected {path})"
            raise FileNotFoundError(msg)
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            msg = f"DRBench config {path} must hold a JSON object"
            raise TypeError(msg)
        record[path.stem] = parsed
    return record


def corpus_layout(env_config: dict) -> list[tuple[str, str]]:
    """Map a task's upstream file manifest to its local-mode corpus layout.

    Each entry becomes `<app>/<name>` under the task's `environment/files/`. Only
    files named in `env.json` are laid down, which keeps the sibling `qa_dict.json`
    ground-truth sidecars and plaintext `.md` twins of the binary documents out of
    the agent's reach.

    Upstream declares the same destination more than once in some tasks, pointing at
    genuinely different documents; upstream's own loader overwrites, dropping content
    that may carry an insight. Duplicates therefore get a `-N` suffix on the stem, in
    manifest order, so every declared document survives.

    Duplicates are detected case-insensitively. Some tasks declare two documents whose
    names differ only in case (`EV-Battery-Analysis.docx` and `ev-battery-analysis.docx`),
    which a case-insensitive filesystem would silently collapse into one file — making
    the generated corpus depend on the host the adapter ran on. Case-folding the key
    keeps the layout identical on macOS and Linux.

    Args:
        env_config: Parsed `env.json` for one task.

    Returns:
        A list of `(source, relative_destination)` pairs. `source` is the
        repo-relative upstream path; `relative_destination` is POSIX-style and
        contains no `..` components.

    Raises:
        TypeError: If `env.json` has an unexpected shape.
        ValueError: If an entry names an unknown app or an unusable source path.
    """
    env_files = env_config.get("env_files")
    if not isinstance(env_files, list):
        msg = "DRBench `env.json` must hold an `env_files` list"
        raise TypeError(msg)

    layout: list[tuple[str, str]] = []
    seen: Counter[str] = Counter()
    for entry in env_files:
        if not isinstance(entry, dict):
            msg = "Each DRBench `env_files` entry must be a mapping"
            raise TypeError(msg)
        source = entry.get("source")
        destination = entry.get("destination")
        app = entry.get("app")
        if not isinstance(source, str) or not isinstance(destination, str):
            msg = "DRBench `env_files` entry must declare string `source` and `destination`"
            raise TypeError(msg)
        if app not in _KNOWN_APPS:
            msg = f"DRBench `env_files` entry declares unknown app {app!r}"
            raise ValueError(msg)
        if _SOURCE_RE.match(source) is None:
            msg = f"DRBench `env_files` source {source!r} is not under a task's `files/`"
            raise ValueError(msg)

        name = PurePosixPath(destination).name
        if not name or name in {".", ".."}:
            msg = f"DRBench `env_files` destination {destination!r} has no usable file name"
            raise ValueError(msg)

        key = f"{app}/{name}".lower()
        seen[key] += 1
        if seen[key] > 1:
            stem, dot, suffix = name.partition(".")
            name = f"{stem}-{seen[key]}{dot}{suffix}"
        layout.append((source, f"{app}/{name}"))
    return layout


def insight_ground_truth(eval_config: dict) -> list[dict[str, str]]:
    """Extract the insight-bearing ground truth the verifier scores recall against.

    Mirrors upstream `QASimilarityV2.compute`, which keeps only `qa_type == "insight"`
    entries and scores one LLM judgement per entry.

    Args:
        eval_config: Parsed `eval.json` for one task.

    Returns:
        One `{id, question, answer, type}` mapping per gold insight, in upstream order.

    Raises:
        TypeError: If `eval.json` has an unexpected shape.
    """
    qa_list = eval_config.get("dr_report_evaluation_qa")
    if not isinstance(qa_list, list):
        msg = "DRBench `eval.json` must hold a `dr_report_evaluation_qa` list"
        raise TypeError(msg)

    insights: list[dict[str, str]] = []
    for qa in qa_list:
        if not isinstance(qa, dict):
            msg = "Each DRBench `dr_report_evaluation_qa` entry must be a mapping"
            raise TypeError(msg)
        if qa.get("qa_type") != _INSIGHT_QA_TYPE:
            continue
        answer = qa.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            continue
        insights.append(
            {
                "id": str(qa.get("id", "")),
                "question": str(qa.get("question", "")),
                "answer": answer,
                "type": str(qa.get("type", "")),
            }
        )
    return insights


def generate_task(*, output_dir: Path, task_id: str) -> Path:
    """Generate one self-contained Harbor task from a vendored DRBench record.

    Args:
        output_dir: Dataset directory that will contain the generated task.
        task_id: Identifier of the form `DR0001` or `SANITY0`.

    Returns:
        Path to the generated Harbor task directory.

    Raises:
        ValueError: If `task_id` is not a bare DRBench id, or the record declares
            an unusable file manifest.
        FileNotFoundError: If no vendored config exists for `task_id`.
        TypeError: If a vendored config has an unexpected shape.
    """
    parse_task_id(task_id)
    record = record_for_task_id(task_id)

    task_dir = output_dir / task_id
    if task_dir.exists():
        # Regenerate cleanly: replace any existing task dir so a rerun overwrites
        # instead of failing on already-created subdirectories. Safe because
        # `parse_task_id` proved `task_id` is a single path component.
        shutil.rmtree(task_dir)
    (task_dir / "environment").mkdir(parents=True)

    layout = corpus_layout(record["env"])
    insights = insight_ground_truth(record["eval"])
    _write_task_files(task_dir, record=record, layout=layout, insights=insights)
    return task_dir


def populate_corpus(dataset_dir: Path, *, archive: Path | None = None) -> int:
    """Regenerate each DRBench task's single-sourced, git-ignored files.

    Two kinds of per-task files are not committed:

    * the document corpus under `environment/files/` (~87 MB across the dataset),
      fetched from the pinned upstream tree;
    * the invariant build and verifier files `environment/extract_text.py` and
      `tests/{test.sh,judge.py}` (single-sourced in `templates/`).

    This regenerates both so Harbor can build and grade each task — run it before
    `harbor run --path <dataset_dir>`. The committed per-task `tests/case.json`
    (question + ground-truth insights) is left untouched.

    Args:
        dataset_dir: Dataset directory containing generated task directories.
        archive: Optional pre-downloaded upstream tarball. Downloads the pinned
            archive when omitted.

    Returns:
        The number of DRBench task directories populated.

    Raises:
        FileNotFoundError: If a task's corpus is missing from the archive.
    """
    dataset_root = dataset_dir.resolve()
    wanted: dict[str, Path] = {}
    for task_toml in sorted(dataset_root.glob("*/task.toml")):
        task_dir = task_toml.parent
        # Containment: only populate direct children of the dataset directory.
        if task_dir.resolve().parent != dataset_root:
            continue
        if 'source = "drbench"' not in task_toml.read_text(encoding="utf-8"):
            continue
        wanted[task_dir.name] = task_dir

    if not wanted:
        return 0

    with tempfile.TemporaryDirectory() as scratch:
        archive_path = archive if archive is not None else _download_corpus(Path(scratch))
        for task_id, task_dir in wanted.items():
            files_dir = task_dir / "environment" / "files"
            if files_dir.exists():
                shutil.rmtree(files_dir)
            files_dir.mkdir(parents=True)
        _extract_corpus(archive_path, wanted)

    for task_dir in wanted.values():
        _copy_environment_invariants(task_dir / "environment")
        _copy_verifier_invariants(task_dir / "tests")
    return len(wanted)


def _download_corpus(destination_dir: Path) -> Path:
    """Download the pinned upstream tarball into `destination_dir`."""
    archive_path = destination_dir / f"drbench-{UPSTREAM_SHA}.tar.gz"
    # Fixed https URL built from a pinned commit SHA; no caller-supplied input.
    with urllib.request.urlopen(_CORPUS_URL, timeout=600) as response:  # noqa: S310
        with archive_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    return archive_path


def _extract_corpus(archive_path: Path, wanted: dict[str, Path]) -> None:
    """Copy each wanted task's declared documents out of the upstream tarball.

    Members are streamed and written to destinations this module computes, so no
    archive-controlled path is ever used to open a file: a member name only ever
    selects a precomputed destination. Non-regular members (links, devices) are
    ignored.

    Args:
        archive_path: Upstream `tar.gz` archive.
        wanted: Map of task id to generated task directory.

    Raises:
        FileNotFoundError: If a task's declared documents are absent from the archive.
    """
    # source path -> (destination file, ...). A single upstream document can back
    # more than one destination when a task declares it under several apps.
    destinations: dict[str, list[Path]] = {}
    for task_id, task_dir in wanted.items():
        files_dir = task_dir / "environment" / "files"
        for source, relative in corpus_layout(record_for_task_id(task_id)["env"]):
            destinations.setdefault(source, []).append(files_dir / relative)

    remaining = set(destinations)
    # Upstream's manifest disagrees with the tree on filename case for one distractor
    # (DR0038 declares `PR-engagements-overview.docx`; the file is
    # `PR-Engagements-Overview.docx`). Upstream only loads it on a case-insensitive
    # filesystem, so resolve case-only mismatches the same way rather than dropping a
    # document the task is supposed to include. Consulted only for sources no exact
    # member matched, so it cannot shadow an exact hit.
    case_insensitive = {source.lower(): source for source in destinations}

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            if not member.isreg():
                continue
            # Upstream archives nest everything under a single `<repo>-<sha>/` root.
            _, _, repo_relative = member.name.partition("/")
            targets = destinations.get(repo_relative)
            if targets is None:
                aliased = case_insensitive.get(repo_relative.lower())
                if aliased is None or aliased not in remaining:
                    continue
                repo_relative = aliased
                targets = destinations[aliased]
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            payload = extracted.read()
            for target in targets:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            remaining.discard(repo_relative)

    if remaining:
        missing = ", ".join(sorted(remaining)[:5])
        msg = f"Upstream archive {archive_path.name} is missing {len(remaining)} file(s): {missing}"
        raise FileNotFoundError(msg)


def _copy_environment_invariants(environment_dir: Path) -> None:
    """Copy the task-invariant build inputs into `environment_dir`.

    `extract_text.py` is a `COPY` input of the committed Dockerfile and is identical
    across every task, so it is single-sourced in `templates/` and git-ignored per
    task alongside the corpus it exists to read.
    """
    environment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_templates_dir() / "extract_text.py", environment_dir / "extract_text.py")


def _copy_verifier_invariants(tests_dir: Path) -> None:
    """Copy the task-invariant verifier files into `tests_dir`.

    `test.sh` and `judge.py` are byte-identical across every task, so they are
    single-sourced in `templates/` and git-ignored per task. Both task generation
    and `populate_corpus` lay them down from the single copy, mirroring how the
    corpus is handled. Only `case.json` (the per-task question + ground-truth
    insights) is committed per task.
    """
    tests_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = _templates_dir()
    shutil.copy2(templates_dir / "test.sh", tests_dir / "test.sh")
    shutil.copy2(templates_dir / "judge.py", tests_dir / "judge.py")


def _company_brief(company: dict) -> str:
    """Render the company profile block shown to the agent."""
    lines = [f"- **Company:** {company.get('name', 'Unknown')}"]
    for label, key in (
        ("Industry", "industry"),
        ("Headquarters", "headquarters"),
        ("Size", "size"),
        ("Employees", "employee_count"),
        ("Annual revenue", "annual_revenue"),
        ("Market position", "market_position"),
    ):
        value = company.get(key)
        if value:
            lines.append(f"- **{label}:** {value}")
    description = company.get("description")
    if description:
        lines.append(f"- **Description:** {description}")
    for label, key in (
        ("Key products and services", "key_products_services"),
        ("Target markets", "target_markets"),
        ("Compliance certifications", "compliance_certifications"),
    ):
        value = company.get(key)
        if isinstance(value, list) and value:
            lines.append(f"- **{label}:** {'; '.join(str(item) for item in value)}")
    return "\n".join(lines)


def _persona_brief(persona: dict) -> str:
    """Render the persona block shown to the agent."""
    lines = [f"- **Name:** {persona.get('name', 'Unknown')}"]
    for label, key in (
        ("Role", "role"),
        ("Department", "department"),
        ("Seniority", "seniority"),
        ("Email", "email"),
        ("Responsibilities", "responsibilities"),
    ):
        value = persona.get(key)
        if value:
            lines.append(f"- **{label}:** {value}")
    return "\n".join(lines)


def _instruction(record: dict[str, dict], layout: list[tuple[str, str]]) -> str:
    """Compose the task prompt: persona, company, question, and output contract."""
    task_config = record["task"]
    question = str(task_config.get("dr_question", "")).strip()
    date = str(task_config.get("date", "")).strip()
    company = task_config.get("company_info") or {}
    persona = task_config.get("persona") or {}
    app_dirs = sorted({relative.split("/", 1)[0] for _, relative in layout})

    app_lines = "\n".join(
        f"- `/app/files/{app}/` — {_APP_DESCRIPTIONS[app]}" for app in app_dirs
    )
    return f"""# Deep research request

{question}

## Who is asking

{_persona_brief(persona)}

## Company context

{_company_brief(company)}

{f"Today's date is {date}." if date else ""}

## Where to research

Your company's internal documents have been exported to `/app/files/`, grouped by the
system they came from:

{app_lines}

Formats include PDF, DOCX, XLSX, PPTX, and JSONL email exports. Run
`extract-text <path>` to convert any of them to plain text. Not every document is
relevant — the export contains unrelated material alongside what you need.

You also have internet access. Some of what this question needs is public information
that is not in the export at all, so research the open web as well as the files.

## What to deliver

Write a research report to `/app/report.md` as Markdown.

- Ground every factual claim in a source, cited inline with a bracketed number
  (`[1]`, `[2]`, ...).
- End the report with a `## References` section listing each number against the
  document file name or the URL it came from. Cite the file name, not the number,
  in that list.
- Report only what your sources support. Uncited assertions and claims you cannot
  trace back to a document or web page do not count in your favour.
- Cover the question thoroughly: the report is scored on how many of the findings a
  domain expert would consider essential you actually surface.
"""


_APP_DESCRIPTIONS = {
    "nextcloud": "files shared on the company's cloud drive",
    "email": "mailbox exports (JSONL, one message per line)",
    "mattermost": "exported team chat conversations",
    "file_system": "documents from local and shared drives",
}


def _write_task_files(
    task_dir: Path,
    *,
    record: dict[str, dict],
    layout: list[tuple[str, str]],
    insights: list[dict[str, str]],
) -> None:
    """Write every generated file for one Harbor task."""
    task_config = record["task"]
    info = record["info"]
    task_id = str(task_config["task_id"])
    question = str(task_config.get("dr_question", "")).strip()

    environment_dir = task_dir / "environment"
    (environment_dir / "Dockerfile").write_text(_DOCKERFILE, encoding="utf-8")
    _copy_environment_invariants(environment_dir)
    (environment_dir / ".dockerignore").write_text(
        ".env\n.env.*\n*.pem\n*.key\n*.crt\ncredentials.json\n.git\n__pycache__/\n.venv/\n.DS_Store\n",
        encoding="utf-8",
    )
    (task_dir / "instruction.md").write_text(
        _instruction(record, layout),
        encoding="utf-8",
    )

    # The oracle writes every gold insight straight into the report, so a passing
    # oracle run proves the judge wires up (corpus, case.json, reward channel) rather
    # than proving anything about research ability.
    solution_dir = task_dir / "solution"
    solution_dir.mkdir()
    oracle_report = "# Reference report\n\n" + "\n\n".join(
        f"{index}. {insight['answer']}" for index, insight in enumerate(insights, 1)
    )
    (solution_dir / "solve.sh").write_text(
        f"#!/bin/sh\nset -eu\nprintf '%s\\n' {shlex.quote(oracle_report)} > /app/report.md\n",
        encoding="utf-8",
    )

    # `case.json` is the only per-task verifier input, so it is committed; the
    # invariant verifier files are single-sourced and git-ignored (regenerated by
    # `populate_corpus`), like the corpus. Ground truth lives only here, under
    # `tests/`, which Harbor copies to the verifier and never to the agent workdir.
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    _copy_verifier_invariants(tests_dir)
    (tests_dir / "case.json").write_text(
        json.dumps(
            {"task_id": task_id, "question": question, "insights": insights},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    difficulty = str(info.get("difficulty", "medium"))
    industry = str(info.get("industry", ""))
    domain = str(info.get("domain", ""))
    external = sum(1 for insight in insights if insight["type"] == "external_fact")
    (task_dir / "task.toml").write_text(
        f"""version = "1.3"

[metadata]
source = "drbench"
task_id = "{task_id}"
industry = "{industry}"
domain = "{domain}"
difficulty = "{difficulty}"
insight_count = {len(insights)}
external_insight_count = {external}
document_count = {len(layout)}

[environment]
# Public egress, unlike the context-retrieval tasks: DRBench splits its ground truth
# into enterprise facts (in the corpus) and external facts that only exist on the open
# web, so an allowlist would make the external insights unreachable by construction.
network_mode = "public"
build_timeout_sec = 900.0

[agent]
# Deep research over a multi-format corpus plus open-web search; the scorecard scales
# this with `agent_timeout_multiplier`.
timeout_sec = 3600.0

[verifier]
# One judge call to split the report into claims, then one per gold insight.
timeout_sec = 1800.0
""",
        encoding="utf-8",
    )


_DOCKERFILE = """FROM python:3.12-slim

# Installed at build time (the build phase has network) so the in-sandbox agent's
# runtime bootstrap skips apt. `poppler-utils` provides pdftotext for the corpus'
# PDFs; curl lets the agent fetch web pages it finds.
RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
        ca-certificates \\
        curl \\
        poppler-utils \\
    && rm -rf /var/lib/apt/lists/*

# DRBench's corpus is PDF/DOCX/XLSX/PPTX/JSONL. The benchmark scores research and
# synthesis, not container-format parsing, so the task ships the same extraction
# libraries upstream's own agent uses, behind one `extract-text` entry point.
#
# Both the install and the launcher name the interpreter by absolute path. Harbor's
# LangGraph agent builds its own uv venv inside this container, so a bare `python3`
# would resolve to whichever interpreter is first on the agent's PATH -- likely that
# venv, which does not have these libraries.
RUN /usr/local/bin/python3 -m pip install --no-cache-dir \\
        openpyxl==3.1.5 \\
        pypdf==6.1.1 \\
        python-docx==1.2.0 \\
        python-pptx==1.0.2

COPY extract_text.py /usr/local/lib/extract_text.py
RUN printf '#!/bin/sh\\nexec /usr/local/bin/python3 /usr/local/lib/extract_text.py "$@"\\n' \\
        > /usr/local/bin/extract-text \\
    && chmod 0755 /usr/local/bin/extract-text

# Fail the build, not the run, if the launcher's interpreter cannot import an
# extractor: at runtime this surfaces as an unreadable corpus and a zero score.
RUN printf 'x' > /tmp/probe.txt \\
    && extract-text /tmp/probe.txt > /dev/null \\
    && /usr/local/bin/python3 -c "import openpyxl, pypdf, docx, pptx" \\
    && rm /tmp/probe.txt

COPY files/ /app/files/
WORKDIR /app
"""
