"""Generate a Harbor dataset from the OOLONG-synth long-context benchmark.

Reads a ``(dataset, context_len)`` bucket via :mod:`loader` and emits one
self-contained Harbor task directory per example, plus the dataset-level
``dataset.toml`` and ``metric.py``. Each task is **agent-agnostic**: it seeds the
document at ``/context.txt`` and asks the agent to write its answer to
``/app/answer.txt``; the verifier grades that file with the official OOLONG
scorer (:mod:`official_scorer`) and writes ``/logs/verifier/reward.json``. Which
arm (plain subagents vs. code-interpreter) runs is a ``harbor run`` choice, not
part of the task — mirroring PR #4213, where the arm is runtime metadata.

The gold answer lives **only** in ``tests/datapoint.json`` (verifier side), never
in the agent's environment or instruction, so the agent cannot read it.

Usage::

    python -m deepagents_harbor.oolong.generate_oolong_tasks \
        --dataset trec_coarse --context-len 1024 --n-examples 1

Run from ``libs/evals/`` (so the package is importable), or anywhere with
``PYTHONPATH`` pointing at ``libs/evals``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
import textwrap
from pathlib import Path

if __package__ in (None, ""):
    # Run standalone (``python generate_oolong_tasks.py``): import the stdlib-only
    # loader directly by adding this dir to the path, so we skip the parent
    # ``deepagents_harbor`` package __init__ (which pulls in aiohttp et al.).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from loader import OolongExample, load_oolong_examples  # ty: ignore[unresolved-import]
else:
    from deepagents_harbor.oolong.loader import OolongExample, load_oolong_examples

#: Verbatim upstream scorer, copied into every task's ``tests/`` dir so the
#: dataset is self-contained (the verifier needs no installed package).
_SCORER_SRC = Path(__file__).parent / "official_scorer.py"

#: Default Harbor registry org for generated task names.
_DEFAULT_ORG = "langchain-ai"

#: metric.py — averages the per-task ``score`` across the rewards JSONL. Matches
#: Harbor's ``harbor init --with-metric`` template shape (single-key rewards).
_METRIC_PY = '''\
# /// script
# dependencies = []
# ///
"""Dataset metric: mean OOLONG ``score`` across all tasks."""

import argparse
import json
from pathlib import Path


def main(input_path: Path, output_path: Path) -> None:
    rewards: list[float] = []
    for line in input_path.read_text().splitlines():
        reward = json.loads(line)
        if reward is None:
            rewards.append(0.0)
        elif len(reward) != 1:
            raise ValueError(f"Expected exactly one key in reward, got {len(reward)}")
        else:
            rewards.extend(reward.values())

    mean = sum(rewards) / len(rewards) if rewards else 0.0
    output_path.write_text(json.dumps({"mean_score": mean}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", type=Path, required=True)
    parser.add_argument("-o", "--output-path", type=Path, required=True)
    args = parser.parse_args()
    main(args.input_path, args.output_path)
'''


def _task_short_name(example: OolongExample) -> str:
    """Stable, unique, registry-valid task short name for one example."""
    return f"oolong-synth-{example.dataset}-{example.context_len}-{example.task_id}"


def _render_task_toml(
    example: OolongExample,
    org: str,
    agent_timeout: float,
    verifier_timeout: float,
    network_mode: str,
    allowed_hosts: list[str],
) -> str:
    name = f"{org}/{_task_short_name(example)}"
    description = (
        f"OOLONG-synth long-context aggregation: {example.dataset} subset, "
        f"{example.context_len}-token context, {example.task_group} task."
    )

    # Network policy. Harbor's `langgraph` agent installs its venv *inside* the
    # container during agent setup (needs PyPI), and the agent then calls the LLM.
    # - "public": environment baseline is public, so both agent setup (pip) and the
    #   LLM call work out of the box. This is the runnable default.
    # - "no-network": hermetic baseline (verifier-safe; python-dateutil is baked at
    #   build) with the agent phase narrowed to an LLM-host allowlist. This only
    #   works once the agent's deps are pre-baked into the image so setup needs no
    #   network — a future hardening, not the default.
    if network_mode == "no-network":
        agent_net = f'network_mode = "allowlist"\nallowed_hosts = {json.dumps(allowed_hosts)}'
        env_mode = "no-network"
    else:
        agent_net = ""  # agent inherits the public environment baseline
        env_mode = "public"
    agent_section = f"timeout_sec = {agent_timeout}"
    if agent_net:
        agent_section += "\n" + agent_net

    # toml string values: json.dumps gives correct double-quoted, escaped strings.
    return textwrap.dedent(
        f"""\
        schema_version = "1.3"

        [task]
        name = {json.dumps(name)}
        description = {json.dumps(description)}
        authors = [{{ name = "LangChain" }}]
        keywords = ["oolong", "long-context", "aggregation", {json.dumps(example.dataset)}]

        [metadata]
        benchmark = "oolong-synth"
        subset = {json.dumps(example.dataset)}
        context_len = {example.context_len}
        task_group = {json.dumps(example.task_group)}
        task_type = {json.dumps(example.task_type)}
        answer_type = {json.dumps(example.answer_type)}
        origin = "https://github.com/abertsch72/oolong"

        [agent]
        {agent_section}

        [verifier]
        timeout_sec = {verifier_timeout}

        [environment]
        build_timeout_sec = 600.0
        network_mode = "{env_mode}"
        """
    )


def _render_instruction(example: OolongExample) -> str:
    return (
        f"{example.question}\n\n"
        "The document you must analyze is at `/app/context.txt`. Read it in full, "
        "determine the answer, then write **only** your final answer to "
        "`/app/answer.txt` (create the file if it does not exist). Answer in the "
        "exact format the question requests."
    )


def _render_dockerfile() -> str:
    # Build-time network is available regardless of the runtime no-network policy,
    # so python-dateutil (needed by the official scorer) is baked in here and the
    # agent + verifier can both run hermetically.
    return textwrap.dedent(
        """\
        FROM python:3.13-slim

        RUN pip install --no-cache-dir python-dateutil==2.9.0.post0

        RUN mkdir -p /app
        WORKDIR /app

        COPY context.txt /app/context.txt
        """
    )


def _render_solve_sh(example: OolongExample) -> str:
    # Oracle: emit an answer the official parser extracts to gold (→ score 1.0).
    # The dataset value is shlex-quoted before interpolation into the shell script.
    gold = example.gold_answers[0] if example.gold_answers else example.gold_answer_raw
    answer_line = shlex.quote(f"Answer: {gold}")
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        mkdir -p /app
        printf '%s\\n' {answer_line} > /app/answer.txt
        """
    )


def _render_test_sh() -> str:
    return textwrap.dedent(
        """\
        #!/usr/bin/env bash
        set -uo pipefail
        mkdir -p /logs/verifier
        # score.py writes reward.json on every path; this is a last-resort fallback
        # so the verifier never lacks a reward file.
        if ! python3 /tests/score.py; then
          echo '{"score": 0.0}' > /logs/verifier/reward.json
        fi
        """
    )


_SCORE_PY = '''\
"""Harbor verifier: grade /app/answer.txt with the official OOLONG scorer."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from official_scorer import synth_process_response  # noqa: E402

_HERE = Path(os.path.dirname(os.path.abspath(__file__)))
_ANSWER = Path("/app/answer.txt")
_DATAPOINT = _HERE / "datapoint.json"
_REWARD = Path("/logs/verifier/reward.json")


def main() -> None:
    datapoint = json.loads(_DATAPOINT.read_text())
    output = _ANSWER.read_text() if _ANSWER.exists() else ""
    graded = synth_process_response(datapoint, output, datapoint.get("model", "unknown"))
    score = float(graded["score"])
    _REWARD.parent.mkdir(parents=True, exist_ok=True)
    _REWARD.write_text(json.dumps({"score": score}))
    print(
        f"OOLONG score={score} parse={graded['attempted_parse']!r} "
        f"gold={graded['answer']!r} confidence={graded['parse_confidence']}"
    )


if __name__ == "__main__":
    main()
'''


def _render_datapoint(example: OolongExample, model: str) -> str:
    # Exactly the fields synth_process_response reads — gold lives ONLY here.
    return json.dumps(
        {
            "id": example.task_id,
            "context_window_id": example.context_window_id,
            "dataset": example.dataset,
            "answer": example.gold_answer_raw,
            "answer_type": example.answer_type,
            "model": model,
        },
        indent=2,
    )


def _write_task_dir(
    task_dir: Path,
    example: OolongExample,
    *,
    org: str,
    model: str,
    agent_timeout: float,
    verifier_timeout: float,
    network_mode: str,
    allowed_hosts: list[str],
) -> None:
    env_dir = task_dir / "environment"
    sol_dir = task_dir / "solution"
    tests_dir = task_dir / "tests"
    for d in (env_dir, sol_dir, tests_dir):
        d.mkdir(parents=True, exist_ok=True)

    (task_dir / "task.toml").write_text(
        _render_task_toml(
            example, org, agent_timeout, verifier_timeout, network_mode, allowed_hosts
        )
    )
    (task_dir / "instruction.md").write_text(_render_instruction(example))

    (env_dir / "Dockerfile").write_text(_render_dockerfile())
    (env_dir / "context.txt").write_text(example.context_window_text)

    solve = sol_dir / "solve.sh"
    solve.write_text(_render_solve_sh(example))
    solve.chmod(0o755)

    test = tests_dir / "test.sh"
    test.write_text(_render_test_sh())
    test.chmod(0o755)
    (tests_dir / "score.py").write_text(_SCORE_PY)
    (tests_dir / "datapoint.json").write_text(_render_datapoint(example, model))
    shutil.copyfile(_SCORER_SRC, tests_dir / "official_scorer.py")


def _write_dataset_files(out_dir: Path, org: str) -> None:
    (out_dir / "metric.py").write_text(_METRIC_PY)
    (out_dir / "dataset.toml").write_text(
        textwrap.dedent(
            f"""\
            [dataset]
            name = "{org}/oolong-synth"
            description = "OOLONG-synth long-context aggregation, generated as a Harbor dataset."

            [[files]]
            path = "metric.py"
            """
        )
    )


def main() -> None:
    """CLI entry point: generate the OOLONG Harbor dataset from one HF bucket."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="trec_coarse", help="OOLONG subset.")
    parser.add_argument("--context-len", type=int, default=1024, help="Token bucket.")
    parser.add_argument(
        "--n-examples",
        type=int,
        default=1,
        help="Examples to emit; 0 means the full bucket.",
    )
    parser.add_argument("--split", default=None, help="HF split override.")
    parser.add_argument("--org", default=_DEFAULT_ORG, help="Registry org for task names.")
    parser.add_argument("--model", default="oracle", help="Model label in datapoint.json.")
    parser.add_argument("--agent-timeout", type=float, default=1200.0)
    parser.add_argument("--verifier-timeout", type=float, default=120.0)
    parser.add_argument(
        "--network",
        choices=("public", "no-network"),
        default="public",
        help=(
            "Environment network policy. 'public' (default) lets the installed "
            "agent pip-install and call the LLM. 'no-network' is hermetic with an "
            "agent-phase allowlist (requires pre-baked agent deps)."
        ),
    )
    parser.add_argument(
        "--agent-allowed-hosts",
        default="api.anthropic.com",
        help="Comma-separated hosts the agent phase may reach (used only with --network no-network).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "dataset",
        help="Output dataset directory.",
    )
    args = parser.parse_args()

    allowed_hosts = [h.strip() for h in args.agent_allowed_hosts.split(",") if h.strip()]
    n_examples = None if args.n_examples == 0 else args.n_examples
    examples = load_oolong_examples(
        dataset=args.dataset,
        context_len=args.context_len,
        n_examples=n_examples,
        split=args.split,
    )
    if not examples:
        msg = (
            f"No rows for dataset={args.dataset!r} context_len={args.context_len} "
            f"split={args.split!r}. Try a different bucket."
        )
        raise SystemExit(msg)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_dataset_files(out_dir, args.org)

    for example in examples:
        task_dir = out_dir / _task_short_name(example)
        _write_task_dir(
            task_dir,
            example,
            org=args.org,
            model=args.model,
            agent_timeout=args.agent_timeout,
            verifier_timeout=args.verifier_timeout,
            network_mode=args.network,
            allowed_hosts=allowed_hosts,
        )
        print(
            f"wrote {task_dir.relative_to(out_dir.parent)}  ({example.answer_type}, gold={example.gold_answers})"
        )

    print(f"\n{len(examples)} task(s) written to {out_dir}")


if __name__ == "__main__":
    main()
