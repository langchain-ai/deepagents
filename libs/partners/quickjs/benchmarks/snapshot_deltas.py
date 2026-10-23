"""Evaluation-only snapshot delta adapters and isolated measurements."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable

    from quickjs_rs import Runtime, ThreadWorker

    from langchain_quickjs._repl import _ThreadREPL as QuickJSREPL

MIB = 1024 * 1024
CAP = 64 * MIB
MIN_REPEATS = 3
ROOT = Path("/workspace/.quickjs-eval/benchmark")


class Run(TypedDict, total=False):
    """A successful measurement or explicit phase failure."""

    case: str
    config: str
    phase: str
    status: str
    seconds: float
    peak_rss_bytes: int
    patch_bytes: int
    new_bytes: int


class Summary(TypedDict, total=False):
    """Metrics for one case, configuration, and phase."""

    case: str
    config: str
    phase: str
    repeats: int
    status: str
    patch_bytes: int
    patch_percent: float
    median_seconds: float
    worst_seconds: float
    median_peak_rss_bytes: float
    worst_peak_rss_bytes: int


@dataclass(frozen=True)
class Config:
    """An explicitly named adapter configuration."""

    name: str
    family: str
    level: int = 3
    ldm: bool = False
    window: str = "auto"
    algorithm: str = ""
    compression: str = ""


def _configs() -> list[Config]:
    configs = [Config("bsdiff4", "bsdiff4")]
    for family in ("zstandard", "stdlib"):
        for level in (3, 10, 19, 22):
            for ldm in (False, True):
                for window in ("auto", "cover"):
                    name = f"{family}-{level}-ldm{int(ldm)}-{window}"
                    configs.append(Config(name, family, level, ldm, window))
    for algorithm in ("bsdiff", "hdiffpatch", "match-blocks"):
        for compression in ("lzma", "zstd", "bz2"):
            name = f"detools-{algorithm}-{compression}"
            configs.append(
                Config(name, "detools", algorithm=algorithm, compression=compression)
            )
    return configs


def _check_cap(size: int, cap: int = CAP) -> None:
    if size > cap:
        msg = "Snapshot exceeds configured cap"
        raise ValueError(msg)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify(actual: bytes, expected: bytes) -> None:
    if _digest(actual) != _digest(expected) or actual != expected:
        msg = "Snapshot roundtrip mismatch"
        raise ValueError(msg)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _environment() -> dict[str, object]:
    names = (
        "langchain-quickjs",
        "quickjs-rs",
        "bsdiff4",
        "detools",
        "zstandard",
        "ruff",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "versions": versions,
        "zstd_stdlib": getattr(
            importlib.import_module("compression.zstd"), "zstd_version", None
        )
        if sys.version_info >= (3, 14)
        else "unavailable",
        "cpu": platform.processor(),
        "time_unix": time.time(),
    }


def _adapters(
    config: Config, source_size: int
) -> tuple[Callable[[bytes, bytes], bytes], Callable[[bytes, bytes], bytes]]:
    module = importlib.import_module(
        {"stdlib": "compression.zstd"}.get(config.family, config.family)
    )
    if config.family == "bsdiff4":
        return module.diff, module.patch
    if config.family == "detools":
        return _detools_adapters(config)
    window_log = max(10, math.ceil(math.log2(max(1, source_size * 2))))
    if config.family == "zstandard":
        options = {"enable_ldm": config.ldm}
        if config.window == "cover":
            options["window_log"] = window_log
        parameters = module.ZstdCompressionParameters.from_level(
            config.level, **options
        )

        def diff(source: bytes, target: bytes) -> bytes:
            dictionary = module.ZstdCompressionDict(
                source, dict_type=module.DICT_TYPE_RAWCONTENT
            )
            return module.ZstdCompressor(
                compression_params=parameters, dict_data=dictionary
            ).compress(target)

        def patch(source: bytes, delta: bytes) -> bytes:
            dictionary = module.ZstdCompressionDict(
                source, dict_type=module.DICT_TYPE_RAWCONTENT
            )
            return module.ZstdDecompressor(dict_data=dictionary).decompress(delta)

        return diff, patch
    parameter = module.CompressionParameter
    options = {
        parameter.compression_level: config.level,
        parameter.enable_long_distance_matching: int(config.ldm),
    }
    if config.window == "cover":
        options[parameter.window_log] = window_log

    def diff(source: bytes, target: bytes) -> bytes:
        dictionary = module.ZstdDict(source, is_raw=True).as_prefix
        return module.compress(target, options=options, zstd_dict=dictionary)

    def patch(source: bytes, delta: bytes) -> bytes:
        dictionary = module.ZstdDict(source, is_raw=True).as_prefix
        return module.decompress(delta, zstd_dict=dictionary)

    return diff, patch


def _detools_adapters(
    config: Config,
) -> tuple[Callable[[bytes, bytes], bytes], Callable[[bytes, bytes], bytes]]:
    module = importlib.import_module("detools")

    def diff(source: bytes, target: bytes) -> bytes:
        output = io.BytesIO()
        patch_type = "hdiffpatch" if config.algorithm == "hdiffpatch" else "sequential"
        module.create_patch(
            io.BytesIO(source),
            io.BytesIO(target),
            output,
            algorithm=config.algorithm,
            compression=config.compression,
            patch_type=patch_type,
            use_mmap=False,
        )
        return output.getvalue()

    def patch(source: bytes, delta: bytes) -> bytes:
        output = io.BytesIO()
        module.apply_patch(io.BytesIO(source), io.BytesIO(delta), output)
        return output.getvalue()

    return diff, patch


def _pin() -> None:
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})


def _worker(args: argparse.Namespace) -> None:
    _pin()
    config = next(config for config in _configs() if config.name == args.config)
    _check_cap(Path(args.source).stat().st_size)
    _check_cap(Path(args.target).stat().st_size)
    source = Path(args.source).read_bytes()
    expected = Path(args.target).read_bytes()
    if max(len(source), len(expected)) > CAP:
        msg = "Input exceeds 64 MiB cap"
        raise ValueError(msg)
    diff, patch = _adapters(config, len(source))
    warm_source = bytes(range(256)) * 4
    warm_target = warm_source + b"warm"
    warm_path = Path(args.patch + ".warm")
    if args.phase == "diff":
        warm_delta = diff(warm_source, warm_target)
        _verify(patch(warm_source, warm_delta), warm_target)
        warm_path.write_bytes(warm_delta)
    else:
        warm_delta = warm_path.read_bytes()
        _verify(patch(warm_source, warm_delta), warm_target)
    result = _measure(args, source, expected, diff, patch)
    sys.stdout.write(json.dumps(result) + "\n")


def _measure(
    args: argparse.Namespace,
    source: bytes,
    expected: bytes,
    diff: Callable[[bytes, bytes], bytes],
    patch: Callable[[bytes, bytes], bytes],
) -> dict[str, object]:
    result: dict[str, object]
    if args.phase == "diff":
        start = time.perf_counter_ns()
        delta = diff(source, expected)
        elapsed = time.perf_counter_ns() - start
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        _verify(patch(source, delta), expected)
        Path(args.patch).write_bytes(delta)
        result = {"patch_bytes": len(delta), "patch_sha256": _digest(delta)}
    else:
        delta = Path(args.patch).read_bytes()
        start = time.perf_counter_ns()
        actual = patch(source, delta)
        elapsed = time.perf_counter_ns() - start
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        _verify(actual, expected)
        result = {"patch_bytes": len(delta), "patch_sha256": _digest(delta)}
    result.update(
        seconds=elapsed / 1e9,
        peak_rss_bytes=peak * 1024,
        verified_sha256=_digest(expected),
        source_sha256=_digest(source),
        affinity=sorted(os.sched_getaffinity(0)),
    )
    return result


def _eval(repl: QuickJSREPL, code: str) -> None:
    outcome = repl.eval_sync(code)
    if outcome.error_type:
        msg = f"{outcome.error_type}: {outcome.error_message}"
        raise RuntimeError(msg)


def _normal_workloads(
    repl: QuickJSREPL,
    capture: Callable[[str, str], str],
    pair: Callable[[str, str, str, str], None],
) -> str:
    cold = capture("cold", "Fresh initialized REPL")
    _eval(
        repl,
        "globalThis.counter = 1; globalThis.rows = "
        "Array.from({length: 200}, (_, i) => ({id: i, value: "
        "`row-${i}`}));",
    )
    warm = capture("warm", "First eval: records and counter")
    pair("cold-warm", cold, warm, "Fresh context to first real eval")
    _eval(repl, 'counter += 1; rows[10].value = "changed";')
    small = capture("small", "Counter and one record changed")
    pair("consecutive-small", warm, small, "Consecutive eval, small mutation")
    _eval(repl, "globalThis.total = rows.reduce((sum, row) => sum + row.id, 0);")
    consecutive = capture("consecutive", "Aggregate computed")
    pair("consecutive-aggregate", small, consecutive, "Consecutive aggregate eval")
    _eval(
        repl,
        "rows = rows.map(row => ({id: row.id, value: `replacement-${row.id * 17}`}));",
    )
    changed = capture("records-replaced", "All 200 records replaced")
    pair(
        "records-replaced",
        consecutive,
        changed,
        "Large mutation of normal record workload",
    )
    return changed


async def _new_runtime(old: Runtime | None = None) -> Runtime:
    if old is not None:
        old.close()
    return importlib.import_module("quickjs_rs").Runtime(memory_limit=512 * MIB)


def _new_repl(worker: ThreadWorker, runtime: Runtime) -> QuickJSREPL:
    repl_type = vars(importlib.import_module("langchain_quickjs._repl"))["_ThreadREPL"]
    return repl_type(
        worker, runtime, timeout=60.0, capture_console=True, max_stdout_chars=4000
    )


def _corpus_writers(
    args: argparse.Namespace,
    repl: QuickJSREPL,
    snapshots: dict[str, dict[str, int | str]],
    cases: list[dict[str, str]],
) -> tuple[Callable[[str, str], str], Callable[[str, str, str, str], None]]:
    def capture(name: str, workload: str) -> str:
        data = repl.create_snapshot()
        _check_cap(len(data), args.cap_mib * MIB)
        (args.root / "corpus" / f"{name}.bin").write_bytes(data)
        snapshots[name] = {
            "bytes": len(data),
            "sha256": _digest(data),
            "workload": workload,
        }
        return name

    def pair(name: str, before: str, after: str, workload: str) -> None:
        cases.append(
            {
                "name": name,
                "source": f"{before}.bin",
                "target": f"{after}.bin",
                "workload": workload,
            }
        )

    return capture, pair


def _array_workload(
    repl: QuickJSREPL,
    capture: Callable[[str, str], str],
    pair: Callable[[str, str, str, str], None],
    size: int,
    previous: str,
) -> str:
    _eval(
        repl,
        f"globalThis.payload = new Uint8Array({size * MIB}); "
        "globalThis.seed = 305419896; "
        "for (let i = 0; i < payload.length; i++) { "
        "seed ^= seed << 13; seed ^= seed >>> 17; "
        "seed ^= seed << 5; payload[i] = seed & 255; }",
    )
    current = capture(
        f"array-{size}", f"{size} MiB seeded xorshift typed array, nonzero data"
    )
    if size == 1:
        pair(
            "array-growth-1",
            previous,
            current,
            "Allocate deterministic 1 MiB typed array",
        )
    _eval(
        repl,
        "for (let i = 0; i < payload.length; i += 4096) "
        "payload[i] ^= 91; counter += 1;",
    )
    mutated = capture(
        f"array-{size}-mutated",
        f"{size} MiB typed array, one byte changed per 4 KiB",
    )
    pair(
        f"array-{size}-sparse",
        current,
        mutated,
        "Sparse mutation after typed-array allocation",
    )
    return mutated


def _large_array_mutation(
    repl: QuickJSREPL,
    capture: Callable[[str, str], str],
    pair: Callable[[str, str, str, str], None],
    mutated: str,
) -> None:
    _eval(
        repl,
        "seed = 2271560481; for (let i = 0; i < payload.length; i"
        " += 2) { seed ^= seed << 13; seed ^= seed >>> 17; seed "
        "^= seed << 5; payload[i] = seed & 255; }",
    )
    large = capture("array-1-large", "Half of seeded typed-array bytes replaced")
    pair(
        "array-1-large",
        mutated,
        large,
        "Large mutation: half of typed-array bytes replaced",
    )


def _write_manifest(
    args: argparse.Namespace,
    snapshots: dict[str, dict[str, int | str]],
    cases: list[dict[str, str]],
) -> None:
    _write_json(
        args.root / "manifest.json",
        {
            "label": "Generated real runtime workloads, NOT production "
            "traffic; maximum realistic size unknown",
            "cap_bytes": args.cap_mib * MIB,
            "snapshots": snapshots,
            "cases": cases,
            "environment": _environment(),
        },
    )


def _generate(args: argparse.Namespace) -> None:
    runtime_module = importlib.import_module("quickjs_rs")
    importlib.import_module("langchain_quickjs._repl")
    worker = runtime_module.ThreadWorker()
    runtime = worker.run_sync(_new_runtime())
    repl = _new_repl(worker, runtime)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "corpus").mkdir(exist_ok=True)
    snapshots: dict[str, dict[str, int | str]] = {}
    cases: list[dict[str, str]] = []
    capture, pair = _corpus_writers(args, repl, snapshots, cases)
    try:
        previous = _normal_workloads(repl, capture, pair)
        sizes = (1,) if args.only == "small" else (1, 8, 32, 54)
        for size in sizes:
            if size != 1:
                runtime = worker.run_sync(_new_runtime(runtime))
                repl = _new_repl(worker, runtime)
                capture, pair = _corpus_writers(args, repl, snapshots, cases)
                _eval(repl, "globalThis.counter = 1;")
            previous = _array_workload(repl, capture, pair, size, previous)
            if size == 1:
                _large_array_mutation(repl, capture, pair, previous)
        _write_manifest(args, snapshots, cases)
    finally:

        async def close_runtime() -> None:
            runtime.close()

        worker.run_sync(close_runtime())
        worker.close()


async def _spawn(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    process = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError:
        process.kill()
        await process.communicate()
        raise
    returncode = await process.wait()
    return subprocess.CompletedProcess(
        argv, returncode, stdout.decode(), stderr.decode()
    )


def _invoke(
    args: argparse.Namespace,
    case: dict[str, str],
    config: Config,
    phase: str,
    patch: Path,
) -> dict[str, object]:

    start = time.monotonic()
    try:
        process = asyncio.run(
            _spawn(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--config",
                    config.name,
                    "--phase",
                    phase,
                    "--source",
                    str(args.root / "corpus" / case["source"]),
                    "--target",
                    str(args.root / "corpus" / case["target"]),
                    "--patch",
                    str(patch),
                ],
                timeout=args.timeout,
            )
        )
    except TimeoutError:
        return {"status": "timeout", "phase": phase, "timeout_seconds": args.timeout}
    if process.returncode:
        return {
            "status": "unsupported_or_error",
            "phase": phase,
            "returncode": process.returncode,
            "error": process.stderr[-8000:],
            "process_seconds": time.monotonic() - start,
        }
    return {
        "status": "ok",
        "phase": phase,
        **json.loads(process.stdout),
        "process_seconds": time.monotonic() - start,
    }


def _select_cases(
    args: argparse.Namespace, cases: list[dict[str, str]]
) -> list[dict[str, str]]:
    if args.only == "small":
        cases = [
            case
            for case in cases
            if not case["name"].startswith(("array-8", "array-32", "array-54"))
        ]
    return cases


def _validate_corpus(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    corpus = (root / "corpus").resolve()
    for name, metadata in manifest["snapshots"].items():
        path = (corpus / f"{name}.bin").resolve()
        if not path.is_relative_to(corpus):
            msg = "Corpus path escapes its directory"
            raise ValueError(msg)
        _check_cap(path.stat().st_size)
        data = path.read_bytes()
        if len(data) != metadata["bytes"] or _digest(data) != metadata["sha256"]:
            msg = f"Corpus integrity failure: {name}"
            raise ValueError(msg)


def _prepare_run(args: argparse.Namespace) -> Path:
    _validate_corpus(args.root)
    log = args.root / "runs.jsonl"
    if log.exists() and not args.resume:
        msg = "Results already exist; use --resume or a fresh --root"
        raise FileExistsError(msg)
    if args.repeats < MIN_REPEATS:
        msg = "At least three repeats are required"
        raise ValueError(msg)
    _pin()
    return log


def _write_run_environment(
    args: argparse.Namespace,
    run_id: str,
    configs: list[Config],
    cases: list[dict[str, str]],
) -> None:
    _write_json(
        args.root / f"environment-{run_id}.json",
        {
            "environment": _environment(),
            "configs": [asdict(config) for config in configs],
            "cases": cases,
            "repeats": args.repeats,
            "timeout": args.timeout,
            "warmup": "Imports plus tiny adapter calls; patch warmup uses "
            "precomputed tiny patch, never diff in patch worker",
        },
    )


def _resume_state(
    args: argparse.Namespace,
    log: Path,
) -> tuple[set[tuple[str, str, int, str]], set[tuple[str, str]]]:
    previous = (
        [json.loads(line) for line in log.read_text().splitlines()]
        if args.resume and log.exists()
        else []
    )
    completed = {
        (row["case"], row["config"], row["repeat"], row["phase"])
        for row in previous
        if row["status"] == "ok"
    }
    failed = {(row["case"], row["config"]) for row in previous if row["status"] != "ok"}
    return completed, failed


def _run_repeats(
    args: argparse.Namespace,
    case: dict[str, str],
    config: Config,
    patch: Path,
    log: Path,
    run_id: str,
    completed: set[tuple[str, str, int, str]],
) -> str:
    result: dict[str, object] = {"status": "ok"}
    for repeat in range(args.repeats):
        for phase in ("diff", "patch"):
            if (case["name"], config.name, repeat, phase) in completed:
                continue
            result = _invoke(args, case, config, phase, patch)
            result.update(
                case=case["name"],
                config=config.name,
                repeat=repeat,
                run_id=run_id,
                new_bytes=(args.root / "corpus" / case["target"]).stat().st_size,
            )
            with log.open("a") as output:
                output.write(json.dumps(result) + "\n")
            if result["status"] != "ok":
                break
        if result["status"] != "ok":
            break
    return str(result["status"])


def _run(args: argparse.Namespace) -> None:
    log = _prepare_run(args)
    manifest = json.loads((args.root / "manifest.json").read_text())
    configs = [
        config
        for config in _configs()
        if not args.configs or config.name in args.configs.split(",")
    ]
    cases = _select_cases(
        args,
        [
            case
            for case in manifest["cases"]
            if not args.cases or case["name"] in args.cases.split(",")
        ],
    )
    run_id = str(time.time_ns())
    _write_run_environment(args, run_id, configs, cases)
    patch_dir = args.root / "patches"
    patch_dir.mkdir(exist_ok=True)
    completed, failed = _resume_state(args, log)
    for case in cases:
        for config in configs:
            if (case["name"], config.name) in failed:
                continue
            patch = patch_dir / f"{case['name']}-{config.name}.bin"
            status = _run_repeats(args, case, config, patch, log, run_id, completed)
            sys.stdout.write(f"{case['name']} {config.name}: {status}\n")
            sys.stdout.flush()
    _summarize(args)


def _summary_groups(records: list[Run]) -> dict[tuple[str, str, str], list[Run]]:
    groups: dict[tuple[str, str, str], list[Run]] = {}
    for record in records:
        groups.setdefault(
            (record["case"], record["config"], record["phase"]), []
        ).append(record)
    for case, config in {(row["case"], row["config"]) for row in records}:
        groups.setdefault((case, config, "patch"), [{"status": "not_run_diff_failed"}])
    return groups


def _summary(key: tuple[str, str, str], rows: list[Run]) -> Summary:
    case, config, phase = key
    good = [row for row in rows if row["status"] == "ok"]
    summary: Summary = {
        "case": case,
        "config": config,
        "phase": phase,
        "repeats": len(good),
        "status": "ok"
        if len(good) >= MIN_REPEATS and len(good) == len(rows)
        else ("incomplete" if rows[-1]["status"] == "ok" else rows[-1]["status"]),
    }
    if summary["status"] == "ok":
        summary.update(
            patch_bytes=good[-1]["patch_bytes"],
            patch_percent=100 * good[-1]["patch_bytes"] / good[-1]["new_bytes"],
            median_seconds=statistics.median(row["seconds"] for row in good),
            worst_seconds=max(row["seconds"] for row in good),
            median_peak_rss_bytes=statistics.median(
                row["peak_rss_bytes"] for row in good
            ),
            worst_peak_rss_bytes=max(row["peak_rss_bytes"] for row in good),
        )
    return summary


def _write_results(root: Path, records: list[Run], summaries: list[Summary]) -> None:
    _write_json(
        root / "results.json",
        {
            "method": "Serial fresh single-CPU subprocess per phase/repeat; "
            "adapter-only walltime; entire-process peak RSS including"
            " imports and inputs, sampled before correctness "
            "verification; NOT incremental RSS. Every measured diff "
            "is roundtrip verified; every measured patch checks "
            "SHA256 and byte equality. Patch workers never execute "
            "diff.",
            "summaries": summaries,
            "runs": records,
        },
    )
    _write_csv(root, summaries)
    _write_markdown(root, summaries)


def _write_csv(root: Path, summaries: list[Summary]) -> None:
    fields = [
        "case",
        "config",
        "phase",
        "status",
        "repeats",
        "patch_bytes",
        "patch_percent",
        "median_seconds",
        "worst_seconds",
        "median_peak_rss_bytes",
        "worst_peak_rss_bytes",
    ]
    with (root / "results.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def _write_markdown(root: Path, summaries: list[Summary]) -> None:
    lines = [
        "# Snapshot delta evaluation",
        "",
        "Generated runtime workloads, NOT production traffic. "
        "Maximum realistic size unknown.",
        "",
        "Entire-process peak RSS, not incremental; median/worst "
        "of at least three successful repeats. Time excludes "
        "hashing and file I/O. Patch workers never run diff.",
        "",
        "| Case | Configuration | Phase | Status | n | Patch "
        "bytes (% new) | Seconds median / worst | RSS MiB median "
        "/ worst |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        metrics = (
            f"{row['patch_bytes']} ({row['patch_percent']:.4f}%) | "
            f"{row['median_seconds']:.6f} / {row['worst_seconds']:.6f} | "
            f"{row['median_peak_rss_bytes'] / MIB:.1f} / "
            f"{row['worst_peak_rss_bytes'] / MIB:.1f}"
            if "patch_bytes" in row
            else "— | — | —"
        )
        lines.append(
            f"| {row['case']} | {row['config']} | {row['phase']} | "
            f"{row['status']} | {row['repeats']} | {metrics} |"
        )
    (root / "results.md").write_text("\n".join(lines) + "\n")


def _summarize(args: argparse.Namespace) -> None:
    records: list[Run] = [
        json.loads(line) for line in (args.root / "runs.jsonl").read_text().splitlines()
    ]
    summaries = [
        _summary(key, rows) for key, rows in sorted(_summary_groups(records).items())
    ]
    _write_results(args.root, records, summaries)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("corpus", "run", "worker", "summarize", "configs")
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--only", choices=("small", "all"), default="all")
    parser.add_argument("--cap-mib", type=int, choices=range(1, 65), default=64)
    parser.add_argument("--configs", default="")
    parser.add_argument("--cases", default="")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--config", default="bsdiff4")
    parser.add_argument("--phase", choices=("diff", "patch"), default="diff")
    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--patch")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "configs":
        sys.stdout.write(
            json.dumps([asdict(config) for config in _configs()], indent=2) + "\n"
        )
    else:
        {"corpus": _generate, "run": _run, "worker": _worker, "summarize": _summarize}[
            args.command
        ](args)


if __name__ == "__main__":
    _main()
