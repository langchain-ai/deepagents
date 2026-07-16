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

    branches = _indented_block(dispatch, "      branches_to_compare:")
    assert "two or more branches enable N-way comparison" in branches
    assert 'default: ""' in branches

    include_tasks = _indented_block(dispatch, "      include_tasks:")
    assert "every compared branch receives the same tasks" in include_tasks

    agent_timeout = _indented_block(dispatch, "      agent_timeout_multiplier:")
    assert "type: string" in agent_timeout
    assert 'default: "1.0"' in agent_timeout
    assert "20-minute timeout to 40 minutes" in agent_timeout

    retries = _indented_block(dispatch, "      n_retries:")
    assert "type: string" in retries
    assert 'default: "0"' in retries
    assert "retry_agent_timeouts" in retries

    retry_timeouts = _indented_block(dispatch, "      retry_agent_timeouts:")
    assert "type: boolean" in retry_timeouts
    assert "default: false" in retry_timeouts
    assert "model cost" in retry_timeouts

    # Exactly one reusable-workflow call: the flat-pool `eval` job (see
    # test_eval_job_uses_single_flat_pool_matrix below for its shape).
    reusable_call = "uses: ./.github/workflows/_harbor_run.yml"
    assert workflow.count(reusable_call) == 1
    eval_job = _indented_block(workflow, "  eval:")
    assert eval_job.count("force_build: ${{ inputs.force_build }}") == 1
    assert (
        eval_job.count(
            "agent_timeout_multiplier: ${{ inputs.agent_timeout_multiplier }}"
        )
        == 1
    )
    assert eval_job.count("n_retries: ${{ inputs.n_retries }}") == 1
    assert (
        eval_job.count("retry_agent_timeouts: ${{ inputs.retry_agent_timeouts }}")
        == 1
    )
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
    assert (
        "IN_AGENT_TIMEOUT_MULTIPLIER: ${{ inputs.agent_timeout_multiplier }}"
        in prep_job
    )
    assert "IN_N_RETRIES: ${{ inputs.n_retries }}" in prep_job
    assert "IN_RETRY_AGENT_TIMEOUTS: ${{ inputs.retry_agent_timeouts }}" in prep_job
    assert 'echo "| agent_timeout_multiplier |' in prep_job
    assert 'echo "| n_retries |' in prep_job
    assert 'echo "| retry_agent_timeouts |' in prep_job

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
    assert "needs: [prep, build-products]" in eval_job
    strategy = _indented_block(eval_job, "    strategy:")
    assert "fail-fast: false" in strategy
    assert (
        "max-parallel: ${{ fromJson(needs.prep.outputs.version_model_parallel) }}"
        in strategy
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


def test_comparison_builds_branch_wheels_and_forwards_immutable_source() -> None:
    """Evaluate branch products while keeping the controller workflow fixed."""
    workflow = UNIFIED_WORKFLOW.read_text()
    build = _indented_block(workflow, "  build-products:")
    assert "ref: ${{ matrix.sha }}" in build
    assert "libs/deepagents" in build
    assert "libs/code" in build
    assert "eval_product_packages.py build" in build
    assert "name: ${{ matrix.product_artifact }}" in build

    eval_job = _indented_block(workflow, "  eval:")
    eval_with = _indented_block(eval_job, "    with:")
    assert "version_id: ${{ matrix.version_id }}" in eval_with
    assert "source_branch: ${{ matrix.branch }}" in eval_with
    assert "source_sha: ${{ matrix.sha }}" in eval_with
    assert "product_artifact: ${{ matrix.product_artifact }}" in eval_with

    harbor = HARBOR_WORKFLOW.read_text()
    run = _indented_block(harbor, '      - name: "⚓ Run Harbor"')
    assert 'eval_product_packages.py" overrides' in run
    assert '"${dependency_override_args[@]}"' in run
    assert run.index('"${dependency_override_args[@]}"') < run.index(
        '"${dataset_args[@]}"'
    )
    assert "eval_agent_configs.py" in run
    assert "deepagents-compare-${VERSION_ID}-${branch_slug}" in run
    assert "UnifiedComparisonLangSmithPlugin" in run
    assert '--plugin-kwarg "source_sha=$SOURCE_SHA"' in run


def test_comparison_emits_one_safe_archive_per_branch_model_config() -> None:
    workflow = HARBOR_WORKFLOW.read_text()
    bundle = _indented_block(workflow, "  bundle:")
    assert "matrix: ${{ fromJson(needs.prep.outputs.bundle_matrix) }}" in bundle
    assert "bundle_unified_run.py" in bundle
    assert "tar --zstd -cf _bundle/run.tar.zst" in bundle
    assert "name: ${{ steps.slug.outputs.artifact }}" in bundle
    assert "path: _bundle/run.tar.zst" in bundle

    unified_workflow = UNIFIED_WORKFLOW.read_text()
    combine = _indented_block(unified_workflow, "  combine:")
    assert "pattern='unified-run-*'" in combine
    assert "aggregate_unified_compare.py" in combine
    assert "--sources-json" in combine
    assert "radar_by_config" in combine


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


def test_combine_download_classifies_no_artifacts_and_retries_failures() -> None:
    """Only a genuine empty-artifact response may let combine continue."""
    workflow = UNIFIED_WORKFLOW.read_text()
    combine = _indented_block(workflow, "  combine:")
    download = _indented_block(combine, '      - name: "⬇️ Download leaf summaries"')

    assert "destination='_leaves'" in download
    assert 'mkdir -p "$destination"' in download
    assert "attempt=1" in download
    assert "while :; do" in download
    command = (
        'gh run download "$RUN_ID" --repo "$REPO" '
        '--pattern "$pattern" --dir "$attempt_dir" >dl.log 2>&1'
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
    assert download.count("::warning::") == 1

    assert 'echo "Leaf download attempt ${attempt} failed:"' in download
    assert "cat dl.log" in download
    assert 'if [ "$attempt" -ge 3 ]; then' in download
    assert "::error::Leaf download failed after ${attempt} attempts" in download
    assert "exit 1" in download
    assert "attempt=$((attempt + 1))" in download
    assert "sleep $((attempt * 5))" in download
    assert download.count("break") == 2
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


def test_combined_diagnostics_upload_after_aggregation_failure() -> None:
    """Upload a written summary without masking the aggregate job failure."""
    workflow = UNIFIED_WORKFLOW.read_text()
    combine = _indented_block(workflow, "  combine:")
    upload = _indented_block(combine, '      - name: "📤 Upload combined results"')
    assert "hashFiles('_combined/unified_summary.json') != ''" in upload
    assert "hashFiles('_combined/comparison_summary.json') != ''" in upload
    assert "if: ${{ always()" in upload
    assert "continue-on-error" not in upload


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


def test_harbor_artifacts_are_archived_and_extracted_for_aggregation() -> None:
    """Keep sandbox-native paths inside an archive until after download."""
    workflow = HARBOR_WORKFLOW.read_text()
    package = _indented_block(workflow, '      - name: "📦 Package Harbor artifacts"')
    upload = _indented_block(workflow, '      - name: "📤 Upload Harbor artifacts"')
    extract = _indented_block(workflow, '      - name: "📦 Extract shard results"')
    compute = _indented_block(workflow, '      - name: "📊 Compute pass@k / avg@k"')

    assert "tar --zstd -cf harbor-shard.tar.zst" in package
    assert "path: libs/evals/harbor-shard.tar.zst" in upload
    assert "compression-level: 0" in upload
    assert "tar --zstd -xf" in extract
    assert "aggregate_shards.py _results" in compute


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
        "format('harbor-combined-{0}-{1}-{2}', matrix.agent_impl, matrix.category, "
        "steps.slug.outputs.slug)" in upload
    )


def test_shard_artifact_name_includes_agent() -> None:
    harbor = HARBOR_WORKFLOW.read_text()
    # The agent-safe slug is computed and folded into the shard artifact name so
    # two configs of the same model+category do not collide.
    assert "HARBOR_AGENT_SAFE=" in harbor
    assert (
        "shard-${{ env.HARBOR_VERSION_SAFE }}-${{ env.HARBOR_AGENT_SAFE }}-"
        "${{ env.HARBOR_CATEGORY_SAFE }}-"
        "${{ env.LEAF_SLUG }}-${{ strategy.job-index }}" in harbor
    )


def test_aggregate_passes_config() -> None:
    harbor = HARBOR_WORKFLOW.read_text()
    assert "--config" in harbor
    # agg-matrix groups by (category, agent_impl).
    assert 'entry.get("agent_impl")' in harbor


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


def test_harbor_override_preserves_locked_transitive_dependencies() -> None:
    """A source override must not silently replace the locked environment."""
    workflow = HARBOR_WORKFLOW.read_text()
    harbor_job = _indented_block(workflow, "  harbor:")
    override = _indented_block(harbor_job, '      - name: "⚓ Install Harbor override"')
    script = _step_script(override)
    assert 'uv pip install --no-deps --reinstall --refresh "${specs[@]}"' in script
    assert "uv pip check" in script


def test_harbor_agent_dependencies_exclude_mcp_prereleases() -> None:
    """Keep Fireworks prerelease support from selecting the MCP 2.0 beta."""
    workflow = HARBOR_WORKFLOW.read_text()
    harbor_job = _indented_block(workflow, "  harbor:")
    run_harbor = _indented_block(harbor_job, '      - name: "⚓ Run Harbor"')
    assert "UV_PRERELEASE=allow" not in run_harbor


def test_docker_daemon_is_recovered_before_harbor_runs() -> None:
    """Retry a transient hosted-runner Docker failure before starting trials."""
    workflow = HARBOR_WORKFLOW.read_text()
    harbor_job = _indented_block(workflow, "  harbor:")
    docker = _indented_block(harbor_job, '      - name: "🐳 Ensure Docker daemon"')
    assert "if: ${{ inputs.sandbox_env == 'docker' }}" in docker
    script = _step_script(docker)
    assert script.count("docker info") == 2
    assert "sudo systemctl restart docker" in script
    assert "for attempt in 1 2 3 4 5" in script
    assert "exit 1" in script


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
        assert "installed without dependencies" in description
        assert "compatible with the locked environment" in description
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


def test_harbor_run_can_retry_agent_timeouts_explicitly() -> None:
    """Keep costly timeout retries opt-in while preserving Harbor exclusions."""
    workflow = HARBOR_WORKFLOW.read_text()
    call_inputs = _indented_block(workflow, "    inputs:")
    retry_timeouts = _indented_block(call_inputs, "      retry_agent_timeouts:")
    assert "type: boolean" in retry_timeouts
    assert "default: false" in retry_timeouts

    harbor_job = _indented_block(workflow, "  harbor:")
    job_env = _indented_block(harbor_job, "    env:")
    assert (
        "HARBOR_RETRY_AGENT_TIMEOUTS: ${{ inputs.retry_agent_timeouts }}"
        in job_env
    )

    run_step = _indented_block(harbor_job, '      - name: "⚓ Run Harbor"')
    assert 'case "$HARBOR_RETRY_AGENT_TIMEOUTS" in' in run_step
    assert 'retry_args=(--max-retries "$HARBOR_N_RETRIES")' in run_step
    assert "--retry-exclude AgentTimeoutError" not in run_step
    for exception in (
        "VerifierTimeoutError",
        "RewardFileNotFoundError",
        "RewardFileEmptyError",
        "VerifierOutputParseError",
        "ApiUsageLimitError",
    ):
        assert f"--retry-exclude {exception}" in run_step
    assert '"${retry_args[@]}"' in run_step
    syntax = subprocess.run(
        ["bash", "-n"],
        input=_step_script(run_step),
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr


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
