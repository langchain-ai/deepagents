"""Static contracts for the unified evaluation workflows."""

from __future__ import annotations

import os
import re
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UNIFIED_WORKFLOW = ROOT / ".github/workflows/unified_evals.yml"
HARBOR_WORKFLOW = ROOT / ".github/workflows/_harbor_run.yml"
HARBOR_DISPATCH_WORKFLOW = ROOT / ".github/workflows/harbor.yml"
EVALS_WORKFLOW = ROOT / ".github/workflows/evals.yml"
CI_WORKFLOW = ROOT / ".github/workflows/ci.yml"
PREP_SCRIPT = ROOT / ".github/scripts/unified_prep.py"


def _indented_block(text: str, marker: str) -> str:
    """Return an indentation-delimited block without comment-only lines."""
    lines = text.splitlines()
    try:
        start = lines.index(marker)
    except ValueError:
        msg = f"Missing expected marker: {marker}"
        raise AssertionError(msg) from None

    indent = len(marker) - len(marker.lstrip())
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        current_indent = len(line) - len(line.lstrip())
        if current_indent <= indent:
            end = index
            break
    return "\n".join(
        line for line in lines[start:end] if not line.lstrip().startswith("#")
    )


def _download_steps() -> list[tuple[str, str, str]]:
    """Return the isolated leaf and shard download steps."""
    unified = UNIFIED_WORKFLOW.read_text()
    combine = _indented_block(unified, "  combine:")
    leaves = _indented_block(combine, '      - name: "⬇️ Download leaf summaries"')

    reusable = HARBOR_WORKFLOW.read_text()
    aggregate = _indented_block(reusable, "  aggregate:")
    shards = _indented_block(aggregate, '      - name: "⬇️ Download shard results"')
    return [("leaf", leaves, "_leaves"), ("shard", shards, "_shards")]


def _step_script(step: str) -> str:
    """Extract a workflow step's shell body."""
    script_match = re.search(
        r"^\s+run: \|\n(?P<script>.*)$",
        step,
        re.DOTALL | re.MULTILINE,
    )
    assert script_match is not None
    return textwrap.dedent(script_match.group("script"))


def _write_executable(path: Path, content: str) -> None:
    """Write an executable used by a subprocess contract test."""
    path.write_text(content)
    path.chmod(0o755)


def test_dispatch_inputs_reach_every_provider_without_changing_categories() -> None:
    """Keep dispatch controls and the conversation category wired end to end."""
    workflow = UNIFIED_WORKFLOW.read_text()
    dispatch = _indented_block(workflow, "  workflow_dispatch:")

    force_build = _indented_block(dispatch, "      force_build:")
    assert "type: boolean" in force_build
    assert "default: false" in force_build
    assert "first time a local dataset runs" in force_build
    assert "snapshot" in force_build

    harbor_override = _indented_block(dispatch, "      harbor_package_override:")
    assert "type: string" in harbor_override
    assert 'default: ""' in harbor_override
    assert "Optional:" in harbor_override
    assert "Leave empty to use the pinned Harbor" in harbor_override

    categories = _indented_block(dispatch, "      categories:")
    assert 'default: "autonomous,conversation,context"' in categories

    # The deep-agents harness list for autonomous/context defaults to bare.
    agent_impls = _indented_block(dispatch, "      agent_impls:")
    assert "type: string" in agent_impls
    assert 'default: "bare"' in agent_impls

    # Exactly one reusable-workflow call: the flat-pool `eval` job (see
    # test_eval_job_uses_single_flat_pool_matrix below for its shape).
    reusable_call = "uses: ./.github/workflows/_harbor_run.yml"
    assert workflow.count(reusable_call) == 1
    eval_job = _indented_block(workflow, "  eval:")
    assert eval_job.count("force_build: ${{ inputs.force_build }}") == 1
    assert (
        eval_job.count("harbor_package_override: ${{ inputs.harbor_package_override }}")
        == 1
    )

    prep_job = _indented_block(workflow, "  prep:")
    assert "UNIFIED_CATEGORIES: ${{ inputs.categories }}" in prep_job
    assert "run: python .github/scripts/unified_prep.py" in prep_job

    # The run-configuration summary runs even when prep fails and writes to the
    # job summary, so every dispatch records what it was asked to run.
    assert '- name: "📝 Summarize dispatch inputs"' in prep_job
    assert "if: ${{ always() }}" in prep_job
    assert 'echo "## Unified evals — run configuration"' in prep_job
    assert '} >> "$GITHUB_STEP_SUMMARY"' in prep_job

    prep_source = PREP_SCRIPT.read_text()
    conversation = _indented_block(prep_source, '    "conversation": {')
    assert '"dataset": "tau3-subset"' in conversation
    assert '"dataset_path": ""' in conversation
    assert '"agent_impl": "tau3"' in conversation


def test_eval_job_uses_single_flat_pool_matrix() -> None:
    """One eval job matrixes over per-model flat matrices, capped by model_parallel."""
    workflow = UNIFIED_WORKFLOW.read_text()

    prep_job = _indented_block(workflow, "  prep:")
    prep_outputs = _indented_block(prep_job, "    outputs:")
    assert "eval_matrix: ${{ steps.p.outputs.eval_matrix }}" in prep_outputs
    assert "max_parallel: ${{ steps.p.outputs.max_parallel }}" in prep_outputs
    assert "model_parallel: ${{ steps.p.outputs.model_parallel }}" in prep_outputs
    assert "models: ${{ steps.p.outputs.models }}" in prep_outputs
    assert "categories: ${{ steps.p.outputs.categories }}" in prep_outputs
    # No per-provider output or gate exists anywhere in the workflow.
    assert "_has_models" not in workflow

    eval_job = _indented_block(workflow, "  eval:")
    assert "needs: prep" in eval_job
    strategy = _indented_block(eval_job, "    strategy:")
    assert "fail-fast: false" in strategy
    assert (
        "max-parallel: ${{ fromJson(needs.prep.outputs.model_parallel) }}" in strategy
    )
    assert "matrix: ${{ fromJson(needs.prep.outputs.eval_matrix) }}" in strategy
    assert "max-parallel: 1" not in workflow

    eval_with = _indented_block(eval_job, "    with:")
    assert "model: ${{ matrix.model }}" in eval_with
    assert "flat_matrix: ${{ matrix.flat_matrix }}" in eval_with
    assert "max_parallel: ${{ needs.prep.outputs.max_parallel }}" in eval_with
    # n_shards/shard_parallel/langsmith_dataset/include_tasks are per-shard
    # values now carried inside flat_matrix, not passed at the top level.
    assert "n_shards:" not in eval_with
    assert "shard_parallel:" not in eval_with
    assert "langsmith_dataset:" not in eval_with
    assert "include_tasks:" not in eval_with


def test_branches_input_present() -> None:
    dispatch = UNIFIED_WORKFLOW.read_text()
    assert "branches_to_compare:" in dispatch
    assert "UNIFIED_BRANCHES: ${{ inputs.branches_to_compare }}" in dispatch
    assert "the harness graph factory" not in dispatch


def test_eval_job_passes_branch() -> None:
    reusable = UNIFIED_WORKFLOW.read_text()
    assert "branch: ${{ matrix.branch }}" in reusable
    assert "branch_sha: ${{ matrix.branch_sha }}" in reusable
    assert "branches: ${{ steps.p.outputs.branches }}" in reusable


def test_enumerate_step_gated_on_full_profile() -> None:
    """The task-enumeration step only runs for the full profile; lite skips it."""
    workflow = UNIFIED_WORKFLOW.read_text()
    prep_job = _indented_block(workflow, "  prep:")
    enumerate_step = _indented_block(
        prep_job, '      - name: "🔢 Enumerate full-profile tasks"'
    )
    assert "if: ${{ inputs.profile == 'full' }}" in enumerate_step
    assert "ENUM_DATASET" in enumerate_step
    assert "ENUM_DATASET_PATH" in enumerate_step
    assert "harbor_adapters.contextbench.main" in enumerate_step
    assert "--populate" in enumerate_step
    assert "UNIFIED_TASKS_JSON" in enumerate_step

    p_step = _indented_block(
        prep_job, '      - name: "🧮 Parse models + build the per-model flat matrix"'
    )
    p_env = _indented_block(p_step, "        env:")
    assert "UNIFIED_MODELS: ${{ inputs.models }}" in p_env
    assert "UNIFIED_CATEGORIES: ${{ inputs.categories }}" in p_env
    assert "UNIFIED_AGENT_IMPLS: ${{ inputs.agent_impls }}" in p_env
    assert "UNIFIED_PROFILE: ${{ inputs.profile }}" in p_env
    assert "UNIFIED_CONCURRENCY: ${{ inputs.concurrency }}" in p_env
    assert "UNIFIED_ROLLOUTS: ${{ inputs.rollouts }}" in p_env
    assert "UNIFIED_TASKS_JSON: ${{ env.UNIFIED_TASKS_JSON }}" in p_env
    assert "UNIFIED_SHARD_PARALLEL" not in workflow
    assert "UNIFIED_N_SHARDS_" not in workflow


def test_combine_needs_prep_and_eval() -> None:
    """Combine waits on the single eval job, not a fixed provider job list."""
    workflow = UNIFIED_WORKFLOW.read_text()
    combine_job = _indented_block(workflow, "  combine:")
    needs = _indented_block(combine_job, "    needs:")
    assert "- prep" in needs
    assert "- eval" in needs
    # marker line ("needs:") plus exactly the two job names, no leftover
    # provider jobs.
    assert len([line for line in needs.splitlines() if line.strip()]) == 3


def test_combine_receives_expected_leaves() -> None:
    reusable = UNIFIED_WORKFLOW.read_text()
    assert "EXPECTED_LEAVES: ${{ needs.prep.outputs.expected_leaves }}" in reusable
    assert "expected_leaves: ${{ steps.p.outputs.expected_leaves }}" in reusable


def test_unified_dispatch_forwards_exact_task_filter() -> None:
    workflow = UNIFIED_WORKFLOW.read_text()
    prep = _indented_block(workflow, "  prep:")
    parse = _indented_block(
        prep, '      - name: "🧮 Parse models + build the per-model flat matrix"'
    )

    assert "include_tasks:" in workflow.split("permissions:", 1)[0]
    assert "UNIFIED_INCLUDE_TASKS: ${{ inputs.include_tasks }}" in parse
    assert "IN_INCLUDE_TASKS: ${{ inputs.include_tasks }}" in prep


def test_unified_dispatch_forwards_retry_and_timeout_controls() -> None:
    workflow = UNIFIED_WORKFLOW.read_text()
    dispatch = _indented_block(workflow, "  workflow_dispatch:")
    prep = _indented_block(workflow, "  prep:")
    eval_job = _indented_block(workflow, "  eval:")
    reusable = HARBOR_WORKFLOW.read_text()
    harbor = _indented_block(reusable, "  harbor:")
    run_harbor = _indented_block(harbor, '      - name: "⚓ Run Harbor"')
    latest_job = _indented_block(harbor, '      - name: "🔍 Find latest Harbor job"')
    summary = _indented_block(harbor, '      - name: "📝 Write workflow summary"')

    retries = _indented_block(dispatch, "      n_retries:")
    timeout = _indented_block(dispatch, "      agent_timeout_multiplier:")
    assert 'default: "0"' in retries
    assert 'default: "1.0"' in timeout
    assert "UNIFIED_N_RETRIES: ${{ inputs.n_retries }}" in prep
    assert (
        "UNIFIED_AGENT_TIMEOUT_MULTIPLIER: ${{ inputs.agent_timeout_multiplier }}"
        in prep
    )
    assert "n_retries: ${{ inputs.n_retries }}" in eval_job
    assert (
        "agent_timeout_multiplier: ${{ inputs.agent_timeout_multiplier }}" in eval_job
    )

    assert "retry_reward_flag=(--retry-if-reward-below 1.0)" in run_harbor
    assert '"${retry_reward_flag[@]}"' in run_harbor
    assert "retry_include_exceptions" not in workflow
    assert "retry_exclude_exceptions" not in workflow
    assert "--retry-include" not in reusable
    assert "--retry-exclude" not in reusable
    assert "actual_retries=" in latest_job
    assert "Configured retries per eligible failed trial" in summary
    assert "Agent timeout multiplier" in summary
    assert "Actual retries" in summary


def test_latest_harbor_job_reports_actual_retry_count(tmp_path: Path) -> None:
    reusable = HARBOR_WORKFLOW.read_text()
    harbor = _indented_block(reusable, "  harbor:")
    latest_job = _indented_block(harbor, '      - name: "🔍 Find latest Harbor job"')
    job = tmp_path / "harbor-jobs" / "terminal-bench" / "2026-07-21__12-00-00"
    job.mkdir(parents=True)
    (job / "result.json").write_text('{"stats":{"n_retries":2}}')
    output = tmp_path / "github-output"
    env = {**os.environ, "GITHUB_OUTPUT": str(output)}

    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", _step_script(latest_job)],
        cwd=tmp_path,
        env=env,
        check=True,
    )

    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values == {"job_dir": str(job.relative_to(tmp_path)), "actual_retries": "2"}


def test_combine_generates_allocation_driven_comparison_report() -> None:
    workflow = UNIFIED_WORKFLOW.read_text()
    combine = _indented_block(workflow, "  combine:")
    compare = _indented_block(
        combine, '      - name: "🔀 Compare active branches and configs"'
    )

    assert "if: ${{ always() && needs.prep.result == 'success' }}" in compare
    assert "SOURCES: ${{ needs.prep.outputs.sources }}" in compare
    assert "EXPECTED_LEAVES: ${{ needs.prep.outputs.expected_leaves }}" in compare
    assert "EXPECTED_CATEGORIES: ${{ needs.prep.outputs.categories }}" in compare
    assert "aggregate_unified_compare.py _leaves" in compare
    assert '--sources-json "$SOURCES"' in compare
    assert '--expected-leaves-json "$EXPECTED_LEAVES"' in compare
    assert '--categories-json "$EXPECTED_CATEGORIES"' in compare

    upload = _indented_block(
        combine, '      - name: "📤 Upload deterministic comparisons"'
    )
    assert "hashFiles('_comparison/comparison_summary.json') != ''" in upload
    assert "name: unified-comparison" in upload
    assert "path: _comparison/" in upload
    assert "continue-on-error: true" in compare
    assert "continue-on-error: true" in upload


def test_post_run_analysis_jobs_are_warning_only() -> None:
    unified = UNIFIED_WORKFLOW.read_text()
    combine = _indented_block(unified, "  combine:")
    harbor = HARBOR_WORKFLOW.read_text()
    aggregate = _indented_block(harbor, "  aggregate:")

    assert "continue-on-error: true" in combine.split("    runs-on:", 1)[0]
    assert "continue-on-error: true" in aggregate.split("    runs-on:", 1)[0]
    assert 'name: "⚠️ Summarize analysis step failures"' in combine
    assert 'name: "⚠️ Summarize analysis step failures"' in aggregate
    assert 'echo "## Analysis warnings"' in combine
    assert 'echo "## Analysis warnings"' in aggregate


def test_combine_prepares_uv_cache_for_cleanup(tmp_path: Path) -> None:
    """Keep setup-uv cleanup valid when optional chart dependencies are skipped."""
    workflow = UNIFIED_WORKFLOW.read_text()
    combine = _indented_block(workflow, "  combine:")
    prepare = _indented_block(combine, '      - name: "🗂️ Prepare UV cache directory"')
    cache = tmp_path / "uv-cache"
    env = {**os.environ, "UV_CACHE_DIR": str(cache)}

    script = _step_script(prepare)
    subprocess.run(["bash", "-e", "-c", script], env=env, check=True)
    subprocess.run(["bash", "-e", "-c", script], env=env, check=True)

    assert cache.is_dir()
    assert combine.index("🗂️ Prepare UV cache directory") < combine.index(
        "⬇️ Download leaf summaries"
    )


def test_combine_download_classifies_no_artifacts_and_retries_failures() -> None:
    """Only a genuine empty-artifact response may let combine continue."""
    workflow = UNIFIED_WORKFLOW.read_text()
    combine = _indented_block(workflow, "  combine:")
    download = _indented_block(combine, '      - name: "⬇️ Download leaf summaries"')

    assert "mkdir -p _leaves" in download
    assert "attempt=1" in download
    assert "while :; do" in download
    command = (
        'gh run download "$RUN_ID" --repo "$REPO" '
        "--pattern 'harbor-*' --dir \"$attempt_dir\" >dl.log 2>&1"
    )
    assert download.count(f"if {command}; then") == 1

    empty_match = re.search(
        r"if grep -Eqi '[^']+' dl\.log; then(?P<body>.*?)\n\s*fi",
        download,
        re.DOTALL,
    )
    assert empty_match is not None
    empty_body = empty_match.group("body")
    assert "::warning::" in empty_body
    assert "break" in empty_body
    assert "exit 1" not in empty_body
    assert download.count("::warning::") == 2

    assert 'echo "Leaf download attempt ${attempt} failed:"' in download
    assert "cat dl.log" in download
    assert 'if [ "$attempt" -ge 3 ]; then' in download
    assert "::warning::Leaf download failed after ${attempt} attempts" in download
    assert "artifact-download-error.log" in download
    assert "attempt=$((attempt + 1))" in download
    assert "sleep $((attempt * 5))" in download
    assert download.count("break") == 3
    assert "|| echo" not in download


def test_download_classifiers_match_only_empty_artifact_messages() -> None:
    """Recognize gh's empty responses without swallowing operational failures."""
    empty_messages = [
        "no valid artifacts found to download",
        "no artifacts found",
        "no artifact matched",
        "no artifact matches any of the names or patterns provided",
        "no artifacts were found",
    ]
    failure_messages = [
        "HTTP 403: permission denied; no artifact access",
        "authentication failed while downloading artifacts",
        "network error while downloading artifacts",
        "API rate limit exceeded while downloading artifacts",
    ]

    for name, step, _destination in _download_steps():
        grep_match = re.search(
            r"grep -(?P<flags>[A-Za-z]+) '(?P<pattern>[^']+)' dl\.log",
            step,
        )
        assert grep_match is not None, name
        pattern = grep_match.group("pattern")
        for message in empty_messages:
            assert re.search(pattern, message, re.IGNORECASE) is not None, (
                name,
                message,
            )
        for message in failure_messages:
            assert re.search(pattern, message, re.IGNORECASE) is None, (
                name,
                message,
            )
        flags = set(grep_match.group("flags"))
        assert {"E", "q", "i"} <= flags, name


def test_download_retries_discard_partial_attempts(tmp_path: Path) -> None:
    """Promote only a successful extraction from a fresh attempt directory."""
    fake_gh = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import os
        import sys
        from pathlib import Path

        state = Path(os.environ["FAKE_GH_STATE"])
        attempt = int(state.read_text()) + 1 if state.exists() else 1
        state.write_text(str(attempt))
        destination = Path(sys.argv[sys.argv.index("--dir") + 1])
        destination.mkdir(parents=True, exist_ok=True)
        if attempt == 1:
            (destination / "stale.txt").write_text("partial")
            print("network error while downloading artifacts", file=sys.stderr)
            raise SystemExit(1)
        (destination / "success.txt").write_text("complete")
        """
    )

    for name, step, destination_name in _download_steps():
        work = tmp_path / name
        fake_bin = work / "bin"
        temp_root = work / "tmp"
        fake_bin.mkdir(parents=True)
        temp_root.mkdir()
        state = work / "attempts.txt"
        _write_executable(fake_bin / "gh", fake_gh)
        _write_executable(fake_bin / "sleep", "#!/bin/sh\nexit 0\n")

        env = os.environ.copy()
        env.update(
            {
                "FAKE_GH_STATE": str(state),
                "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
                "REPO": "owner/repository",
                "RUN_ID": "123",
                "SHARD_PATTERN": "shard-test-*",
                "TMPDIR": str(temp_root),
            }
        )
        result = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", _step_script(step)],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (name, result.stdout, result.stderr)
        destination = work / destination_name
        assert sorted(path.name for path in destination.iterdir()) == ["success.txt"]
        assert state.read_text() == "2"
        assert list(temp_root.iterdir()) == []

        loop = step[step.index("while :; do") :]
        assert loop.count("attempt_dir=$(mktemp -d)") == 1
        assert loop.index("attempt_dir=$(mktemp -d)") < loop.index("gh run download")
        assert '--dir "$attempt_dir"' in loop
        assert loop.count('rm -rf "$attempt_dir"') == 2
        assert f'mv "$attempt_dir" {destination_name}' in loop
        empty_match = re.search(
            r"if grep -Eqi '[^']+' dl\.log; then(?P<body>.*?)\n\s*fi",
            loop,
            re.DOTALL,
        )
        assert empty_match is not None
        empty_body = empty_match.group("body")
        assert 'rm -rf "$attempt_dir"' in empty_body
        assert f"mkdir -p {destination_name}" in empty_body


def test_combined_diagnostics_upload_is_warning_only() -> None:
    """Upload a written summary without failing on analysis publication."""
    workflow = UNIFIED_WORKFLOW.read_text()
    combine = _indented_block(workflow, "  combine:")
    upload = _indented_block(combine, '      - name: "📤 Upload combined results"')
    condition = "        if: ${{ always() && hashFiles('_combined/unified_summary.json') != '' }}"
    assert upload.count(condition) == 1
    assert "continue-on-error: true" in upload


def test_leaf_aggregation_requires_every_expected_shard() -> None:
    """Count successful empty shards while detecting missing artifacts."""
    workflow = HARBOR_WORKFLOW.read_text()
    harbor = _indented_block(workflow, "  harbor:")
    prep_job = _indented_block(workflow, "  prep:")
    aggregate = _indented_block(workflow, "  aggregate:")

    # Category-scoped so independently-sharded categories in a flat multi-category
    # run can't collide on the same shard index's marker basename (aggregate_shards.py
    # counts markers by basename alone).
    assert (
        'touch "harbor-jobs/terminal-bench/empty-shard-${HARBOR_CATEGORY}-${HARBOR_SHARD_INDEX}"'
        in harbor
    )
    assert "    needs: [prep, harbor]" in aggregate
    # expected_shards now flows per-category via aggregate_matrix (derived in prep
    # from prep's own shard-matrix output on the single-dataset path), not a
    # single job-level env var on the aggregate job.
    assert (
        "SINGLE_EXPECTED_SHARDS: ${{ steps.shard-matrix.outputs.n_shards }}" in prep_job
    )
    assert "EXPECTED_SHARDS: ${{ matrix.expected_shards }}" in aggregate
    compute = _indented_block(aggregate, '      - name: "📊 Compute pass@k / avg@k"')
    assert 'expected_shards_args=(--expected-shards "$EXPECTED_SHARDS")' in compute
    assert '"${expected_shards_args[@]}"' in compute


def test_aggregate_runs_per_category() -> None:
    """Aggregate matrixes over categories instead of hardcoding a single one."""
    text = HARBOR_WORKFLOW.read_text()
    # aggregate loops over the categories present in the flat matrix
    assert "for cat in" in text or "matrix.category" in text
    assert "aggregate_shards.py" in text
    assert "--category" in text

    workflow = HARBOR_WORKFLOW.read_text()
    prep_job = _indented_block(workflow, "  prep:")
    aggregate_job = _indented_block(workflow, "  aggregate:")

    assert (
        "aggregate_matrix: ${{ steps.agg-matrix.outputs.aggregate_matrix }}" in prep_job
    )
    derive_step = _indented_block(prep_job, '      - name: "🗂️ Derive aggregate matrix"')
    assert "FLAT_MATRIX: ${{ inputs.flat_matrix }}" in derive_step
    assert "expected_shards" in derive_step

    aggregate_strategy = _indented_block(aggregate_job, "    strategy:")
    assert (
        "matrix: ${{ fromJson(needs.prep.outputs.aggregate_matrix) }}"
        in aggregate_strategy
    )

    compute = _indented_block(
        aggregate_job, '      - name: "📊 Compute pass@k / avg@k"'
    )
    assert "DATASET: ${{ matrix.dataset }}" in compute
    assert "CATEGORY: ${{ matrix.category }}" in compute
    assert "--category" in compute

    upload = _indented_block(
        aggregate_job, '      - name: "📤 Upload combined results"'
    )
    assert "format('harbor-combined-{0}', steps.slug.outputs.slug)" in upload
    assert (
        "format('harbor-combined-{0}-{1}-{2}-{3}', steps.branch-slug.outputs.slug, "
        "matrix.agent_impl, matrix.category, steps.slug.outputs.slug)" in upload
    )


def test_shard_artifact_name_includes_agent() -> None:
    harbor = HARBOR_WORKFLOW.read_text()
    # The agent-safe slug is computed and folded into the shard artifact name so
    # two configs of the same model+category do not collide. (The branch-safe
    # slug prefixes it; the full branch-first name is pinned in
    # test_shard_artifact_name_includes_branch.)
    assert "HARBOR_AGENT_SAFE=" in harbor
    assert (
        "${{ env.HARBOR_AGENT_SAFE }}-${{ env.HARBOR_CATEGORY_SAFE }}-"
        "${{ env.LEAF_SLUG }}-${{ strategy.job-index }}" in harbor
    )


def test_aggregate_passes_config() -> None:
    harbor = HARBOR_WORKFLOW.read_text()
    assert "--config" in harbor
    # agg-matrix groups by (category, agent_impl).
    assert 'entry.get("agent_impl")' in harbor


def test_harbor_overlays_branch_source() -> None:
    harbor = HARBOR_WORKFLOW.read_text()
    assert "Overlay branch agent source" in harbor
    assert "BRANCH_SHA: ${{ inputs.branch_sha }}" in harbor
    assert 'git fetch origin "$BRANCH_SHA" --depth=1' in harbor
    assert "fetched_sha=$(git rev-parse FETCH_HEAD)" in harbor
    assert "git checkout FETCH_HEAD --" in harbor
    # Only the agent-under-test libraries are overlaid; the harness graph factory
    # (langgraph_agent.py) stays at the eval ref.
    assert "libs/deepagents" in harbor
    assert "libs/code" in harbor
    assert "libs/partners/quickjs" in harbor
    assert "deepagents_harbor/langgraph_project/langgraph_agent.py" not in harbor


def test_unified_comparison_does_not_build_product_wheels() -> None:
    unified = UNIFIED_WORKFLOW.read_text()
    harbor = HARBOR_WORKFLOW.read_text()
    assert "build-products:" not in unified
    assert "product_artifact" not in unified
    assert "dependency_overrides" not in harbor
    assert "branch_wheels" not in harbor


def test_shard_artifact_name_includes_branch() -> None:
    harbor = HARBOR_WORKFLOW.read_text()
    assert "HARBOR_BRANCH_SAFE=" in harbor
    assert (
        "shard-${{ env.HARBOR_BRANCH_SAFE }}-${{ env.HARBOR_AGENT_SAFE }}-"
        "${{ env.HARBOR_CATEGORY_SAFE }}-${{ env.LEAF_SLUG }}-"
        "${{ strategy.job-index }}" in harbor
    )
    assert "--branch" in harbor


def test_langsmith_experiment_branch_is_hash_disambiguated() -> None:
    harbor = HARBOR_WORKFLOW.read_text()
    assert (
        "experiment_branch=\"${experiment_branch}-$(printf '%s' "
        '"$HARBOR_BRANCH" | sha256sum | cut -c1-8)"'
    ) in harbor


def test_artifact_name_comment_attributes_agent_safety_to_enum() -> None:
    harbor = HARBOR_WORKFLOW.read_text()
    assert "the upstream enum" in harbor
    assert "cat-slug step's `exit 1` validation" not in harbor


def test_chart_publishers_are_serialized_and_replace_rerun_assets() -> None:
    """Protect the shared branch from concurrent pushes and stale rerun files."""
    publishers = [
        (UNIFIED_WORKFLOW, "  combine:"),
        (EVALS_WORKFLOW, "  aggregate:"),
    ]
    for workflow_path, job_marker in publishers:
        job = _indented_block(workflow_path.read_text(), job_marker)
        concurrency = _indented_block(job, "    concurrency:")
        assert "      group: eval-assets-publication" in concurrency
        assert "      cancel-in-progress: false" in concurrency

        publish = _indented_block(
            job, '      - name: "🖼️ Publish charts to eval-assets branch"'
        )
        remove = 'rm -rf "${asset_dir}"'
        create = 'mkdir -p "${asset_dir}"'
        copy = 'cp "$GITHUB_WORKSPACE/'
        assert publish.count(remove) == 1
        assert publish.index(remove) < publish.index(create) < publish.index(copy)


def test_credential_check_rejects_unsupported_model_providers() -> None:
    """Fail closed for unknown providers without changing known key checks."""
    workflow = HARBOR_WORKFLOW.read_text()
    harbor_job = _indented_block(workflow, "  harbor:")
    credentials = _indented_block(
        harbor_job, '      - name: "🔑 Verify sandbox credentials"'
    )
    provider_match = re.search(
        r'case "\$model_provider" in(?P<body>.*?)^\s+esac',
        credentials,
        re.DOTALL | re.MULTILINE,
    )
    assert provider_match is not None
    provider_case = provider_match.group("body")

    provider_keys = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google_genai": "GOOGLE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "baseten": "BASETEN_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
        "ollama": "OLLAMA_API_KEY",
        "groq": "GROQ_API_KEY",
        "xai": "XAI_API_KEY",
        "nvidia": "NVIDIA_API_KEY",
    }
    for provider, key in provider_keys.items():
        expected = (
            rf"^\s+{re.escape(provider)}\)\s+"
            rf'\[ -z "\${key}" \] && missing\+\=\("{key}"\)'
        )
        assert re.search(expected, provider_case, re.MULTILINE) is not None

    default_match = re.search(
        r"^\s+\*\)(?P<body>.*?)^\s+;;",
        provider_case,
        re.DOTALL | re.MULTILINE,
    )
    assert default_match is not None
    default = default_match.group("body")
    assert "::error::Unsupported model provider" in default
    assert "exit 1" in default
    assert "::warning::" not in default

    log_lines = [
        line
        for line in credentials.splitlines()
        if re.search(r"\b(?:echo|printf)\b", line)
    ]
    for key in provider_keys.values():
        assert all(
            f"${key}" not in line and f"${{{key}}}" not in line for line in log_lines
        )


def test_harbor_job_preserves_override_without_project_resync() -> None:
    """Prevent later `uv run` commands from replacing an installed override."""
    workflow = HARBOR_WORKFLOW.read_text()
    harbor_job = _indented_block(workflow, "  harbor:")
    assert '      - name: "⚓ Install Harbor override"' in harbor_job

    job_env = _indented_block(harbor_job, "    env:")
    assert job_env.count('      UV_NO_SYNC: "true"') == 1


def test_harbor_job_uses_read_only_token_permissions() -> None:
    """Limit the secret-bearing job while retaining aggregate artifact cleanup."""
    workflow = HARBOR_WORKFLOW.read_text()
    harbor_job = _indented_block(workflow, "  harbor:")
    harbor_permissions = _indented_block(harbor_job, "    permissions:")
    assert harbor_permissions.count("      contents: read") == 1
    assert harbor_permissions.count("      actions: read") == 1
    assert "write" not in harbor_permissions

    aggregate_job = _indented_block(workflow, "  aggregate:")
    aggregate_permissions = _indented_block(aggregate_job, "    permissions:")
    assert aggregate_permissions.count("      contents: read") == 1
    assert aggregate_permissions.count("      actions: write") == 1


def test_override_inputs_warn_against_mutable_or_credentialed_sources() -> None:
    """Keep trusted-source guidance consistent on both dispatch surfaces."""
    descriptions: list[str] = []
    for path in (UNIFIED_WORKFLOW, HARBOR_DISPATCH_WORKFLOW):
        workflow = path.read_text()
        override = _indented_block(workflow, "      harbor_package_override:")
        description_match = re.search(
            r'^\s+description: "(?P<description>.+)"$',
            override,
            re.MULTILINE,
        )
        assert description_match is not None
        description = description_match.group("description")
        assert "trusted package source" in description
        assert "Prefer an immutable commit SHA" in description
        assert "never embed credentials" in description
        assert 'default: ""' in override
        assert "type: string" in override
        descriptions.append(description)

    assert descriptions[0] == descriptions[1]


def test_harbor_run_accepts_flat_matrix_and_derives_parallel_pool() -> None:
    """Wire a flat per-model matrix through prep without losing the single-dataset path."""
    workflow = HARBOR_WORKFLOW.read_text()
    call_inputs = _indented_block(workflow, "    inputs:")
    assert "flat_matrix:" in call_inputs
    assert "max_parallel:" in call_inputs
    flat_matrix_input = _indented_block(call_inputs, "      flat_matrix:")
    assert 'default: ""' in flat_matrix_input
    max_parallel_input = _indented_block(call_inputs, "      max_parallel:")
    assert 'default: "0"' in max_parallel_input

    prep_job = _indented_block(workflow, "  prep:")
    assert "matrix: ${{ steps.resolve-matrix.outputs.matrix }}" in prep_job
    assert (
        "effective_max_parallel: ${{ steps.resolve-matrix.outputs.effective_max_parallel }}"
        in prep_job
    )
    expand_step = _indented_block(prep_job, '      - name: "🔀 Expand matrix by shard"')
    assert "if: ${{ inputs.flat_matrix == '' }}" in expand_step

    resolve_step = _indented_block(
        prep_job, '      - name: "🧮 Resolve matrix + parallel pool"'
    )
    assert "FLAT_MATRIX: ${{ inputs.flat_matrix }}" in resolve_step
    assert "MAX_PARALLEL: ${{ inputs.max_parallel }}" in resolve_step
    assert "SHARD_PARALLEL: ${{ inputs.shard_parallel }}" in resolve_step
    assert 'if [ -n "$FLAT_MATRIX" ]; then' in resolve_step
    assert 'matrix="$FLAT_MATRIX"' in resolve_step
    assert 'echo "matrix=$matrix"' in resolve_step
    assert (
        'if [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]] && [ "$MAX_PARALLEL" -gt 0 ]; then'
        in resolve_step
    )
    assert 'effective_max_parallel="$MAX_PARALLEL"' in resolve_step
    assert 'effective_max_parallel="$SHARD_PARALLEL"' in resolve_step
    assert 'echo "effective_max_parallel=$effective_max_parallel"' in resolve_step

    harbor_job = _indented_block(workflow, "  harbor:")
    strategy = _indented_block(harbor_job, "    strategy:")
    assert (
        "max-parallel: ${{ fromJson(needs.prep.outputs.effective_max_parallel) }}"
        in strategy
    )

    job_env = _indented_block(harbor_job, "    env:")
    assert (
        "HARBOR_DATASET: ${{ matrix.dataset || inputs.dataset || 'terminal-bench/terminal-bench-2' }}"
        in job_env
    )
    assert (
        "HARBOR_DATASET_PATH: ${{ matrix.dataset_path || inputs.dataset_path }}"
        in job_env
    )
    assert "HARBOR_AGENT_IMPL: ${{ matrix.agent_impl || inputs.agent_impl }}" in job_env
    assert (
        "HARBOR_INCLUDE_TASKS: ${{ matrix.include_tasks || inputs.include_tasks }}"
        in job_env
    )
    assert (
        "HARBOR_N_SHARDS: ${{ matrix.n_shards || needs.prep.outputs.n_shards || '1' }}"
        in job_env
    )
    assert "HARBOR_CATEGORY: ${{ matrix.category || inputs.category }}" in job_env
    assert "HARBOR_SHARD_INDEX: ${{ matrix.shard }}" in job_env


def test_evals_ci_filter_includes_unified_workflows() -> None:
    """Run evals CI when either unified workflow changes in isolation."""
    workflow = CI_WORKFLOW.read_text()
    evals_filter = _indented_block(workflow, "            evals:")

    expected_paths = [
        ".github/workflows/_harbor_run.yml",
        ".github/workflows/unified_evals.yml",
    ]
    for path in expected_paths:
        assert evals_filter.count(f"              - '{path}'") == 1
