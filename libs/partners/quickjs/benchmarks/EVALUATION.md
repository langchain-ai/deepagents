# Snapshot delta evaluation

This is an evaluation-only tool, not a runtime implementation or a replacement
for the package's existing `make bench` / `make bench-memory` targets. It does not
change the checkpoint format, dependencies, runtime code, or CI. Do not swap the
runtime adapter without owner agreement, compatibility review, and checkpoint
migration planning.

## Isolated setup

Run from any directory. The external environment and output directory below are
intentional for this evaluation; do not add these candidates to project manifests.
Normal installation builds unchanged source distributions when wheels are absent.

```sh
uv venv --python 3.14 /workspace/.quickjs-eval/venv314
uv pip install --python /workspace/.quickjs-eval/venv314/bin/python \
  /workspace/deepagents/libs/partners/quickjs \
  bsdiff4==1.2.6 detools==0.53.0 zstandard==0.25.0 \
  'ruff>=0.15.20,<0.16.0'
```

`compression.zstd` is provided by Python 3.14 with libzstd support; it needs no
additional Python wheel. Older interpreters record unavailable adapters as errors
rather than silently omitting configurations. The same CLI permits later native
3.11 or free-threaded 3.14 checks in separately provisioned environments. This
harness's Python 3.14 results alone do not establish free-threaded safety.

## Commands

```sh
PY=/workspace/.quickjs-eval/venv314/bin/python
SCRIPT=/workspace/deepagents/libs/partners/quickjs/benchmarks/snapshot_deltas.py
$PY "$SCRIPT" configs
$PY "$SCRIPT" corpus
$PY "$SCRIPT" run --timeout 8
$PY "$SCRIPT" summarize
```

The default output root is `/workspace/.quickjs-eval/benchmark`. Use `--root` to
keep independent experiments separate. `run --resume` skips completed repetitions
and cells with recorded failures, retaining their original errors and timeout
budgets. It resumes unfinished successful cells after interruption. Stop the
previous process before resuming. The full serial matrix can exceed 15 minutes;
use the case/configuration selectors to bound interactive checks. Every child
phase has its own deadline, including imports, warmup, verification, and output.

```sh
$PY "$SCRIPT" corpus --root /workspace/.quickjs-eval/benchmark/small --only small
$PY "$SCRIPT" run --root /workspace/.quickjs-eval/benchmark/small \
  --only small --cases consecutive-small \
  --configs bsdiff4,stdlib-3-ldm0-cover,zstandard-3-ldm0-cover --timeout 8
```

`worker` is the internal single-phase command, with `--config`, `--phase`,
`--source`, `--target`, and `--patch`. Patch workers require the tiny `.warm` patch
written by the corresponding diff worker. They never call a diff adapter.
`--repeats` must be at least three. `--cap-mib` can lower, but never raise, the
64 MiB snapshot cap. Existing results require `--resume` or a fresh `--root`;
accidental appends to an unrelated experiment are rejected. The measurement
harness targets Linux (`sched_setaffinity` and Linux `ru_maxrss` units).

## Corpus

The constructor follows the package's `_ThreadREPL` test fixture, exposed locally
as `QuickJSREPL`: create a `ThreadWorker`, create its `Runtime` on the worker,
construct the REPL with timeout/console options, evaluate JavaScript, and capture
`create_snapshot()` bytes. No snapshot bytes are synthesized or modified by the
harness. Context and runtime access stay on the QuickJS worker thread.

The corpus contains cold-to-warm and consecutive evaluations, record replacement,
a seeded xorshift-filled typed array, sparse writes, and replacement of half the
1 MiB array. Larger independently initialized runtimes allocate 8, 32, and 54 MiB
arrays. Independent runtimes avoid retaining a previous array allocation in the
serialized heap. Snapshot sizes are measured, not inferred from live array sizes:
QuickJS serializes heap capacity and overhead as well as data. All captured
snapshots must fit the cap. Generation uses a 512 MiB runtime heap allowance to
probe large serialized snapshots, not the middleware's default 64 MiB live-heap
allowance. A sub-64 MiB serialized snapshot does not prove that workload runs
within the default live-heap budget. This is **generated runtime workload, not
production traffic**; the maximum realistic production snapshot size is unknown.

The manifest records every snapshot's size, SHA256, workload, pair, environment,
and cap. `run` validates the manifest's hashes and sizes before benchmarking.

## Adapters and matrix

There are 42 configurations:

- `bsdiff4.diff(source, target)` / `bsdiff4.patch(source, delta)` baseline.
- Python `zstandard`: levels 3, 10, 19, 22 × LDM off/on × auto/cover windows,
  using `ZstdCompressionDict(..., DICT_TYPE_RAWCONTENT)` for both directions.
- Standard-library `compression.zstd`: the same level/LDM/window sweep using
  `ZstdDict(source, is_raw=True).as_prefix`. The level is supplied through
  `CompressionParameter.compression_level` because `level` and `options` are
  mutually exclusive. Decompression uses the same source prefix.
- `detools`: `bsdiff`, `hdiffpatch`, `match-blocks` × `lzma`, `zstd`, `bz2`.
  `create_patch` takes three in-memory file objects; `apply_patch` takes source,
  patch, and output. `hdiffpatch` requires `patch_type="hdiffpatch"`; the others
  use `sequential`. `use_mmap=False` makes in-memory streams explicit. These
  combinations were checked against the installed signatures and dispatch code,
  not just its partially outdated algorithm docstring.

A cover window uses `ceil(log2(2 * source_bytes))`, with a minimum of 10. It covers
the source plus same-sized target history. Auto deliberately retains library
defaults; it is not a promise of source-size coverage. Window coverage alone does
not guarantee useful matches: level 3 without LDM can still produce near-full
patches for large random typed arrays. No fallback changes the requested matrix;
unsupported combinations, crashes, and timeouts remain visible. Detools is a
diagnostic candidate, not a wheel-availability solution. Distribution availability
and interpreter compatibility require separate evidence.

## Measurements and artifacts

Measurements are serial. Each repetition and phase starts a fresh subprocess,
pinned to one available CPU, with imports and a tiny untimed adapter warmup.
Diff setup writes the tiny warmup patch separately; patch setup only reads it.
There is no full-size diff in a patch subprocess. Corpus generation runs outside
measurement and is not included in adapter RSS.

Timers surround only the adapter call, including adapter/context construction
and in-memory stream allocation, not disk I/O or SHA256/equality checks. RSS is
Linux `resource.getrusage(RUSAGE_SELF).ru_maxrss`, converted from KiB to bytes,
sampled immediately after the timed phase and before full-size verification.
It is an **entire-process high-water mark including imports and input
allocation**, not incremental adapter memory. Median and worst walltime and RSS
are reported separately for diff and patch.

Every measured diff artifact is applied and checked with both SHA256 and byte
comparison. Every timed patch output receives the same checks. Raw runs retain
source/output/patch hashes, sizes, affinity, phase, repetition, and environment-run
identifier. A timed-out cell stops rather than spending its remaining repeats;
fewer than three successful samples never count as a complete measurement.

Outputs are `manifest.json`, `environment-*.json`, `runs.jsonl`, `results.json`,
`results.csv`, `results.md`, corpus binaries, and patch binaries. Keep all raw
outputs outside the repository. The scripts are excluded from wheel and sdist.

For source checks, use the package-declared Ruff version and configuration:

```sh
/workspace/.quickjs-eval/venv314/bin/ruff check \
  /workspace/deepagents/libs/partners/quickjs/benchmarks
/workspace/.quickjs-eval/venv314/bin/ruff format --check \
  /workspace/deepagents/libs/partners/quickjs/benchmarks
```
