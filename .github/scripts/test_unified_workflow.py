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
    conversation_shards = _indented_block(dispatch, "      n_shards_conversation:")
    assert 'default: "3"' in conversation_shards

    provider_jobs = [
        "eval-anthropic",
        "eval-baseten",
        "eval-fireworks",
        "eval-google_genai",
        "eval-groq",
        "eval-nvidia",
        "eval-ollama",
        "eval-openai",
        "eval-openrouter",
        "eval-xai",
        "eval-other",
    ]
    reusable_call = "uses: ./.github/workflows/_harbor_run.yml"
    assert workflow.count(reusable_call) == len(provider_jobs)
    for job_name in provider_jobs:
        job = _indented_block(workflow, f"  {job_name}:")
        assert job.count(reusable_call) == 1
        assert job.count("force_build: ${{ inputs.force_build }}") == 1
        assert (
            job.count("harbor_package_override: ${{ inputs.harbor_package_override }}")
            == 1
        )

    prep_job = _indented_block(workflow, "  prep:")
    assert "UNIFIED_CATEGORIES: ${{ inputs.categories }}" in prep_job
    assert "UNIFIED_AGENT_IMPL: ${{ inputs.agent_impl }}" in prep_job
    assert (
        "UNIFIED_N_SHARDS_CONVERSATION: ${{ inputs.n_shards_conversation }}" in prep_job
    )
    assert "run: python .github/scripts/unified_prep.py" in prep_job
    # A run-configuration summary in prep makes a dispatch's inputs debuggable.
    assert "$GITHUB_STEP_SUMMARY" in prep_job
    # ...but the harbor_package_override spec (which can carry credentials) must
    # never reach the summary step's environment or public output. Derive only
    # a boolean in the expression context and report whether an override was set.
    summary_step = _indented_block(
        prep_job, '      - name: "📝 Summarize dispatch inputs"'
    )
    summary_env = _indented_block(summary_step, "        env:")
    assert (
        "HARBOR_OVERRIDE_SET: ${{ inputs.harbor_package_override != '' }}"
        in summary_env
    )
    assert "IN_HARBOR_OVERRIDE:" not in summary_env
    assert '[ "${HARBOR_OVERRIDE_SET}" = "true" ]' in summary_step

    # The harness selector is a constrained choice defaulting to bare.
    agent_impl_input = _indented_block(workflow, "      agent_impl:")
    assert "type: choice" in agent_impl_input
    assert 'default: "bare"' in agent_impl_input
    assert "- bare" in agent_impl_input
    assert "- dcode" in agent_impl_input

    prep_source = PREP_SCRIPT.read_text()
    conversation = _indented_block(prep_source, '    "conversation": {')
    assert '"dataset": "tau3-subset"' in conversation
    assert '"dataset_path": ""' in conversation
    assert '"agent_impl": "tau3"' in conversation
    # The deep-agents categories default to the bare harness (dcode is opt-in).
    for marker in ('    "autonomous": {', '    "context": {'):
        assert '"agent_impl": "bare"' in _indented_block(prep_source, marker)


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
    condition = (
        "        if: ${{ always() && "
        "hashFiles('_combined/unified_summary.json') != '' }}"
    )
    assert upload.count(condition) == 1
    assert "continue-on-error" not in upload


def test_leaf_aggregation_requires_every_expected_shard() -> None:
    """Count successful empty shards while detecting missing artifacts."""
    workflow = HARBOR_WORKFLOW.read_text()
    harbor = _indented_block(workflow, "  harbor:")
    aggregate = _indented_block(workflow, "  aggregate:")

    assert (
        'touch "harbor-jobs/terminal-bench/empty-shard-$HARBOR_SHARD_INDEX"' in harbor
    )
    assert "    needs: [prep, harbor]" in aggregate
    assert "EXPECTED_SHARDS: ${{ needs.prep.outputs.n_shards }}" in aggregate
    compute = _indented_block(aggregate, '      - name: "📊 Compute pass@k / avg@k"')
    assert 'expected_shards_args=(--expected-shards "$EXPECTED_SHARDS")' in compute
    assert '"${expected_shards_args[@]}"' in compute


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
