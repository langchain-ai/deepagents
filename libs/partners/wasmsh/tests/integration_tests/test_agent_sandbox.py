"""LLM-driven agent integration tests for WasmshSandbox.

Tests create_deep_agent with WasmshSandbox as backend across 5 scenarios:
- CSV analysis, skill loading, filesystem ops, memory, code execution

Requires: deno 2+, built Pyodide assets, ANTHROPIC_API_KEY
"""

from __future__ import annotations

import json
import os
import shutil
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import HumanMessage

if TYPE_CHECKING:
    from langchain_wasmsh import WasmshSandbox

try:
    from wasmsh_pyodide_runtime import get_dist_dir

    _assets_available = get_dist_dir().joinpath("pyodide.asm.wasm").exists()
except (ImportError, FileNotFoundError):
    _assets_available = False

pytestmark = [
    pytest.mark.skipif(
        shutil.which("deno") is None and shutil.which("node") is None,
        reason="deno or node is required",
    ),
    pytest.mark.skipif(not _assets_available, reason="Pyodide assets not built"),
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
    ),
]

MODEL = "claude-haiku-4-5-20251001"


@pytest.mark.timeout(120)
def test_csv_analysis(sandbox: WasmshSandbox) -> None:
    """Agent analyzes CSV with Python and verifies with shell."""
    from deepagents.graph import create_deep_agent  # noqa: PLC0415

    sandbox.upload_files(
        [
            (
                "/workspace/temps.csv",
                b"city,temp_c\nTokyo,22\nBerlin,15\nCairo,35\nSydney,28\nOslo,5\n",
            )
        ]
    )

    agent = create_deep_agent(model=MODEL, backend=sandbox)
    agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Analyze the CSV file at "
                        "/workspace/temps.csv. "
                        "Calculate the average temperature "
                        "and find the hottest city. "
                        "Write the results as JSON to "
                        "/workspace/analysis.json with keys "
                        '"average", "hottest_city", '
                        'and "hottest_temp".'
                    )
                )
            ]
        }
    )

    cat = sandbox.execute("cat /workspace/analysis.json")
    assert cat.exit_code == 0
    analysis = json.loads(cat.output.strip())
    assert analysis["average"] == 21  # noqa: PLR2004
    assert analysis["hottest_city"] == "Cairo"
    assert analysis["hottest_temp"] == 35  # noqa: PLR2004


@pytest.mark.timeout(120)
def test_skill_loading(sandbox: WasmshSandbox) -> None:
    """Agent loads SKILL.md from sandbox and follows instructions."""
    from deepagents.graph import create_deep_agent  # noqa: PLC0415

    skill_md = (
        "---\nname: md-table-formatter\n"
        "description: Format data as a markdown "
        "table and write it to a file\n---\n\n"
        "When asked to format data, you MUST:\n"
        "1. Create a markdown table with columns: Name, Score, Grade\n"
        "2. Assign grades: A for score >= 90, B for score >= 80, C otherwise\n"
        "3. Write the table to /workspace/output.md\n"
        "4. Write a JSON summary to /workspace/summary.json with keys:\n"
        '   - "count" (number of rows)\n'
        '   - "top_scorer" (name of the person with highest score)\n'
    )

    sandbox.execute("mkdir -p /workspace/skills/md-table-formatter")
    sandbox.upload_files(
        [("/workspace/skills/md-table-formatter/SKILL.md", skill_md.encode())]
    )

    agent = create_deep_agent(
        model=MODEL, backend=sandbox, skills=["/workspace/skills"]
    )
    agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Use the md-table-formatter skill "
                        "to format this student data: "
                        "Alice 92, Bob 85, Carol 97"
                    )
                )
            ]
        }
    )

    table = sandbox.execute("cat /workspace/output.md")
    assert "Alice" in table.output
    assert "|" in table.output

    summary_result = sandbox.execute("cat /workspace/summary.json")
    summary = json.loads(summary_result.output.strip())
    assert summary["count"] == 3  # noqa: PLR2004
    assert summary["top_scorer"] == "Carol"


@pytest.mark.timeout(120)
def test_filesystem_reliability(sandbox: WasmshSandbox) -> None:
    """Edit and write_file work through sandbox."""
    from deepagents.graph import create_deep_agent  # noqa: PLC0415

    sandbox.execute("mkdir -p /workspace/project/src")
    sandbox.upload_files(
        [
            ("/workspace/project/src/main.py", b'print("hello world")\n'),
            ("/workspace/project/src/utils.py", b"def add(a, b):\n    return a + b\n"),
            ("/workspace/project/README.md", b"# My Project\n"),
        ]
    )

    agent = create_deep_agent(model=MODEL, backend=sandbox)
    agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Edit /workspace/project/src/main.py "
                        'to change "hello" to "goodbye". '
                        "Then create a new file "
                        "/workspace/project/src/config.py "
                        "with the content: DEBUG = True"
                    )
                )
            ]
        }
    )

    main = sandbox.execute("cat /workspace/project/src/main.py")
    assert "goodbye" in main.output
    assert "hello" not in main.output

    config = sandbox.execute("cat /workspace/project/src/config.py")
    assert config.exit_code == 0


@pytest.mark.timeout(120)
def test_memory_usage(sandbox: WasmshSandbox) -> None:
    """Agent uses AGENTS.md context loaded from sandbox."""
    from deepagents.graph import create_deep_agent  # noqa: PLC0415

    memory_content = (
        "# Agent Memory\n\n## Project Configuration\n"
        "- Deployment region: eu-central-1 (Frankfurt)\n"
        "- Unit system: metric (Celsius, meters, kilograms)\n"
        "- Team lead: Dr. Weber\n"
        "- Database: PostgreSQL 16\n"
    )

    sandbox.execute("mkdir -p /workspace/memory")
    sandbox.upload_files([("/workspace/memory/AGENTS.md", memory_content.encode())])

    agent = create_deep_agent(
        model=MODEL,
        backend=sandbox,
        memory=["/workspace/memory/AGENTS.md"],
    )
    agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Write a deployment config to "
                        "/workspace/deploy.json with our "
                        'project\'s "region", "unit_system",'
                        ' "team_lead", and "database".'
                    )
                )
            ]
        }
    )

    deploy_result = sandbox.execute("cat /workspace/deploy.json")
    deploy = json.loads(deploy_result.output.strip())
    assert (
        "frankfurt" in deploy["region"].lower()
        or "eu-central" in deploy["region"].lower()
    )
    assert "metric" in deploy["unit_system"].lower()
    assert "Weber" in deploy["team_lead"]
    assert "postgres" in deploy["database"].lower()


@pytest.mark.timeout(120)
def test_code_execution(sandbox: WasmshSandbox) -> None:
    """Python computes median and stddev from data file."""
    from deepagents.graph import create_deep_agent  # noqa: PLC0415

    sandbox.execute("mkdir -p /workspace/data")
    sandbox.upload_files(
        [("/workspace/data/numbers.txt", b"42\n17\n88\n3\n56\n91\n25\n64\n10\n73\n")]
    )

    agent = create_deep_agent(model=MODEL, backend=sandbox)
    agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Read /workspace/data/numbers.txt, "
                        "sort the numbers, compute the "
                        "median and population standard "
                        "deviation. Write the results to "
                        "/workspace/data/stats.json with "
                        'keys "sorted" (array), '
                        '"median" (number), '
                        '"stddev" (rounded to 1 decimal).'
                    )
                )
            ]
        }
    )

    stats_result = sandbox.execute("cat /workspace/data/stats.json")
    stats = json.loads(stats_result.output.strip())
    assert stats["sorted"] == [3, 10, 17, 25, 42, 56, 64, 73, 88, 91]
    assert stats["median"] == 49  # noqa: PLR2004
    assert 29 < stats["stddev"] < 32  # noqa: PLR2004
