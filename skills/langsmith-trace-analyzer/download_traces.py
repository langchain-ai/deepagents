#!/usr/bin/env python3
"""
Download all LangSmith traces for a Harbor benchmark run.

Usage:
    python download_traces.py --job-id <id> --project <name> --result-file result.json --output-dir ./traces

Or run from a directory containing result.json and langsmith_trace_mapping.json:
    python download_traces.py
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def create_trace_mapping(job_id, project_name, output_file):
    """Query LangSmith API to create trace mapping."""
    try:
        from langsmith import Client
    except ImportError:
        print("Error: langsmith package not installed. Run: pip install langsmith")
        sys.exit(1)

    client = Client()
    filter_query = f'and(eq(metadata_key, "job_id"), eq(metadata_value, "{job_id}"))'

    print(f"Querying LangSmith for job_id={job_id} in project={project_name}...")
    traces = {}
    for run in client.list_runs(project_name=project_name, filter=filter_query, is_root=True):
        metadata = run.metadata or {}
        trial_name = metadata.get('harbor_session_id', str(run.id))
        traces[trial_name] = {
            'trace_id': str(run.id),
            'task_name': metadata.get('task_name', 'unknown'),
            'status': run.status
        }

    mapping = {
        'job_id': job_id,
        'project_name': project_name,
        'total_traces': len(traces),
        'created_at': datetime.now().isoformat(),
        'traces': traces
    }

    save_json(output_file, mapping)
    print(f"Created mapping with {len(traces)} traces: {output_file}")
    return mapping


def get_outcome(trial_name, result_data):
    """Determine the outcome (passed/failed/error) for a trial."""
    try:
        evals = result_data['stats']['evals']['deepagent-harbor__terminal-bench']
        reward_stats = evals['reward_stats']['reward']
        exception_stats = evals.get('exception_stats', {})
    except KeyError:
        return 'unknown', None, None

    # Check errors first
    for error_type, trials in exception_stats.items():
        if trial_name in trials:
            return 'error', error_type, None

    # Check rewards
    if trial_name in reward_stats.get('1.0', []):
        return 'passed', None, 1.0
    elif trial_name in reward_stats.get('0.0', []):
        return 'failed', None, 0.0

    return 'unknown', None, None


def fetch_trace(trace_id, output_path, retries=3):
    """Fetch single trace using langsmith-fetch CLI."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "langsmith-fetch", "trace", trace_id,
        "--format", "raw",
        "--include-metadata",
        "--file", str(output_path)
    ]

    for attempt in range(retries):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and output_path.exists():
                # Verify valid JSON
                with open(output_path) as f:
                    json.load(f)
                return True, None
            error = result.stderr or "Unknown error"
        except subprocess.TimeoutExpired:
            error = "Timeout"
        except json.JSONDecodeError as e:
            error = f"Invalid JSON: {e}"
        except Exception as e:
            error = str(e)

        if attempt < retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff

    return False, error


def main():
    parser = argparse.ArgumentParser(description="Download LangSmith traces for a benchmark run")
    parser.add_argument("--job-id", help="Job ID from result.json")
    parser.add_argument("--project", help="LangSmith project name")
    parser.add_argument("--result-file", default="result.json", help="Path to result.json")
    parser.add_argument("--mapping-file", default="langsmith_trace_mapping.json", help="Path to mapping file")
    parser.add_argument("--output-dir", default="langsmith-traces", help="Output directory")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between requests (seconds)")
    parser.add_argument("--create-mapping", action="store_true", help="Create mapping file from LangSmith API")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    mapping_file = Path(args.mapping_file)
    result_file = Path(args.result_file)

    # Create mapping if requested or if it doesn't exist
    if args.create_mapping or not mapping_file.exists():
        if not args.job_id or not args.project:
            # Try to get from result.json
            if result_file.exists():
                result_data = load_json(result_file)
                job_id = args.job_id or result_data.get('id')
                # Project name might need to be provided
                if not job_id:
                    print("Error: --job-id required (not found in result.json)")
                    sys.exit(1)
                if not args.project:
                    print("Error: --project required for creating mapping")
                    sys.exit(1)
            else:
                print("Error: --job-id and --project required when creating mapping")
                sys.exit(1)
        else:
            job_id = args.job_id

        mapping = create_trace_mapping(job_id, args.project, mapping_file)
    else:
        mapping = load_json(mapping_file)

    # Load result.json for outcome classification
    if result_file.exists():
        result_data = load_json(result_file)
    else:
        print(f"Warning: {result_file} not found, all traces will go to 'unknown' folder")
        result_data = {}

    traces = mapping['traces']
    total = len(traces)

    print(f"\n{'=' * 60}")
    print(f"Downloading {total} traces to {output_dir}/")
    print(f"{'=' * 60}\n")

    # Create directory structure
    for subdir in ['passed', 'failed', 'unknown']:
        (output_dir / "by-outcome" / subdir).mkdir(parents=True, exist_ok=True)

    # Track progress
    progress = {'completed': 0, 'failed': [], 'skipped': 0}

    for i, (trial_name, info) in enumerate(sorted(traces.items()), 1):
        outcome, error_type, reward = get_outcome(trial_name, result_data)

        # Determine output path
        if outcome == 'error' and error_type:
            (output_dir / "by-outcome" / "errors" / error_type).mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "by-outcome" / "errors" / error_type / f"{trial_name}.json"
        else:
            output_path = output_dir / "by-outcome" / outcome / f"{trial_name}.json"

        # Skip if exists
        if output_path.exists():
            print(f"[{i}/{total}] SKIP: {trial_name}")
            progress['skipped'] += 1
            continue

        # Download
        print(f"[{i}/{total}] {trial_name} ({outcome})")
        success, error = fetch_trace(info['trace_id'], output_path)

        if success:
            print(f"  -> {output_path.relative_to(output_dir)}")
            progress['completed'] += 1
        else:
            print(f"  FAILED: {error}")
            progress['failed'].append({'trial': trial_name, 'error': error})

        # Rate limiting
        if i < total:
            time.sleep(args.delay)

    # Summary
    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print(f"{'=' * 60}")
    print(f"Downloaded: {progress['completed']}")
    print(f"Skipped (existing): {progress['skipped']}")
    print(f"Failed: {len(progress['failed'])}")

    if progress['failed']:
        print("\nFailed traces:")
        for f in progress['failed']:
            print(f"  - {f['trial']}: {f['error']}")

    # Create manifest
    manifest = {
        'job_id': mapping['job_id'],
        'project_name': mapping['project_name'],
        'download_completed': datetime.now().isoformat(),
        'total_traces': total,
        'downloaded': progress['completed'],
        'skipped': progress['skipped'],
        'failed': len(progress['failed'])
    }
    save_json(output_dir / 'manifest.json', manifest)

    return 0 if not progress['failed'] else 1


if __name__ == "__main__":
    sys.exit(main())
