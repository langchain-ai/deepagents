# Where the latency goes

Measured on the CLI harness (`browser-agent`, headless Google Chrome driven by
the Browserbase `browse` CLI inside a per-thread LangSmith sandbox), over the 29
WebVoyager tasks that completed in the 2026-10-09 experiment.

## Correction to an earlier number

An earlier note put the first browser navigation at "~70% of the median task's
wall clock". **That was wrong** — it compared the 47.1 s median first `web-open`
against an assumed task length rather than the measured one. Against the actual
per-task durations in `trace_report_1009.json`, the correct share is **29.5%**.
Use the table below, not that figure.

## The budget

Per-task wall clock, 29 tasks: **4,551 s total**, mean 156.9 s, median 138.0 s.

| Component | n | median | mean | total | **share of wall clock** |
|---|---|---|---|---|---|
| **First `web-open` of a task (cold start)** | 29 | **47.1 s** | 46.4 s | **1,344 s** | **29.5%** |
| Later `web-open` calls | 89 | 12.7 s | 21.7 s | 1,933 s | 42.5% |
| Other `execute` (snapshot, markdown, click…) | 105 | 9.8 s | 12.8 s | 1,349 s | 29.6% |
| Model time | — | — | — | ~500 s | ~11% |

Accounting caveat: the span sums above total 4,626 s against 4,551 s of task wall
clock — a 1.6% overshoot from nested and overlapping spans. Treat every share as
±2%. Model time overlaps the `execute` spans rather than adding to them.

**So: just under a third of the harness's wall clock is browser cold start, paid
once per task because each task gets its own thread and therefore its own
sandbox.**

## What the cold start actually is

Not sandbox provisioning, and not page weight. A cold `web-open about:blank` —
zero network, zero rendering — still costs **42.5 s**, the same as any real site.
Trivial `execute` calls return promptly before any browser work, so the box is
live well before the first navigation.

It is a 1 vCPU / 2 GB box demand-paging Chrome's mapped set. Chrome maps 486 MB
across 132 files; `/opt/google/chrome/chrome` alone is 298 MB. Faulting that in
4 KB pages on one core is roughly 8× slower than streaming the same bytes.

| Measurement | Time |
|---|---|
| `browse status`, daemon not yet started | 2.17 s |
| **1st `web-open about:blank`, nothing warm** | **42.47 s** |
| 2nd `web-open about:blank` | 4.59 s |
| Warm `web-open https://example.com` | 1.29–1.87 s |
| Raw Chrome cold → CDP ready, fresh profile | 20.01 s |
| Raw Chrome, 2nd launch, *different* fresh profile | 0.98 s |
| Sequential read of `/opt/google/chrome` (442 MB) | 2.6–3.0 s |
| Raw Chrome launch immediately after that read | **4.63 s** |
| `web-open` cold, daemon restarted, everything warm | **4.95 s** ← the floor |

The nonlinearity is the whole story: **2.6 s of sequential reading buys 15 s of
launch time** (20.0 s → 4.6 s).

## Ruled out, with measurements

- **Sandbox provisioning** — not a factor; the box answers trivial commands
  immediately.
- **Chrome flags** — cold launch with `--disable-gpu --disable-software-rasterizer
  --disable-extensions --disable-crash-reporter --disable-breakpad`: 19.96 s vs
  20.01 s baseline. Zero. Stagehand already passes `--no-first-run`,
  `--disable-component-update`, `--disable-background-networking` and ~25 more.
- **Profile / first-run work** — a second launch with a *different* fresh profile
  took 0.98 s, so profile creation is free. No `$HOME/.cache/fontconfig` is ever
  built; the `.deb` builds `/var/cache/fontconfig` at bake. Nothing to
  pre-populate.
- **`chrome-headless-shell`** — a real 8 s win on identical `drop_caches` state
  (6.90 s vs 15.17 s), but `browse` cannot use it: Stagehand injects an unpacked
  extension and the build reports `This Chrome build does not support
  Extensions.loadUnpacked`. Dead end without a no-extension path from Browserbase.
- **Overlapping the cold start with model think time** — mechanically works, but
  buys only the seconds the model actually spends thinking (~2 s), and two
  concurrent `web-open`s clobber each other (`Navigation was superseded by a new
  request`).

## Mitigations, ranked

1. **Reuse the sandbox across tasks instead of one thread per task.** Highest
   confidence, no platform work. Warm steady state is 1.3–1.9 s per navigation,
   and the measured floor for a cold `web-open` on a warm box is 4.95 s. Paying
   47 s once instead of 29 times recovers **1,199 s of 4,551 s — 26.3% of total
   wall clock, 41.3 s off a 156.9 s mean task.**

   *Validity caveat for benchmarking:* sharing one thread across WebVoyager tasks
   also shares the message history, so task N sees task N−1's transcript. That is
   contamination, and it would make the accuracy numbers incomparable to the
   per-task-thread runs. Reuse is the right answer for production throughput; for
   the benchmark, either reset thread state between tasks or keep per-task threads
   and book the 29.5% as a known fixed overhead.

2. **More vCPU / RAM.** High confidence on the cause, unmeasured on magnitude —
   there is no knob exposed. One core serializing 300 MB of page faults is the
   root cause, and 2 GB cannot comfortably hold the 486 MB mapped set plus node.
   Estimated 15–25 s off. Needs the MDA platform team.

3. **Pre-fault off the critical path — no hook exists today.** `define_sandbox` in
   `managed_deepagents` 0.9.0 accepts only `idle_ttl_seconds`, `default_timeout`,
   `proxy_config`, and `snapshot_name`/`snapshot_id`/`docker_image`/`registry`.
   There is no per-thread startup command, and a running process cannot survive
   the snapshot. A per-thread boot command would let a 3 s `cat /opt/google/chrome/*`
   run before the first LLM token, turning 42 s into ~12 s. Estimated 25–30 s off,
   conditional on platform support. **Worth filing.**

4. **`BROWSE_DISABLE_UPDATE_CHECK=1`.** Measured, small, free: 1.137 s vs 1.345 s
   per invocation, and it drops a network call per command. ~1.5 s per task.
   Belongs in the `web-open` wrapper.

5. **Trim the Chrome tree at bake.** Speculative, 1–3 s. `libwidevinecdm.so`
   (22 MB), `liboptimization_guide_internal.so` (25 MB) and
   `libLiteRtWebGpuAccelerator.so` (11 MB) are mapped but almost certainly unused
   headless. Removing them risks startup failure; test before adopting.

## The bigger pot, once cold start is fixed

Later `web-open` (1,933 s, 42.5%) and other `execute` (1,349 s, 29.6%) together
are 72% of wall clock — more than double the cold start. Warm navigations measured
1.3–1.9 s in isolation but run 12.7 s median in production, so that gap is real
page weight and large snapshot/markdown payloads, not startup. Separate
investigation, and the larger one.
