# Enterprise deep-research evals (DRBench, local mode)

100 Harbor tasks generated from ServiceNow's [DRBench](https://github.com/ServiceNow/drbench)
([paper](https://arxiv.org/abs/2510.00172)), the enterprise deep-research benchmark. Each
task hands the agent a company profile, a persona, and an open-ended research question
whose answer is split between the company's own documents and public information on the
open web. The agent writes a cited report; the verifier scores how many of the
benchmark's ground-truth insights that report actually recovers.

This is the `research` category of the unified evals workflow. It is opt-in — pass
`categories: research`.

## Runtime contract

- **Corpus:** `/app/files/<app>/`, grouped by the system each document came from
  (`nextcloud`, `email`, `mattermost`, `file_system`). Formats are PDF, DOCX, XLSX,
  PPTX, and JSONL mail/chat exports; the task image provides `extract-text <path>` to
  convert any of them to plain text.
- **Distractors:** most documents are irrelevant by design. Across the dataset the
  ground truth marks 1,390 distractor facts against 613 insights.
- **Network:** `network_mode = "public"`. Unlike the context-retrieval tasks, an
  allowlist is not an option here: 45 of the 613 gold insights are `external_fact`
  entries that exist only on the open web. The agent gets a Tavily-backed `web_search`
  tool, which the eval workflow enables by forwarding `TAVILY_API_KEY` to this category
  only.
- **Output:** the agent writes `/app/report.md`, with inline `[N]` citations and a
  `## References` section naming each source file or URL.

## Scoring

`tests/judge.py` reproduces DRBench's `insights_recall` metric. It splits the report
into atomic claim/citation pairs using upstream's own prompt, then makes one judge call
per gold insight asking whether the report contains enough to derive it — 1.0 for
`yes`, 0.0 otherwise. The reward is the mean across insights, so it reads directly as
"what fraction of the expected findings did the agent surface".

The judge is a stdlib-only script (no `drbench` install in the verifier) and runs
against the harness judge model (`JUDGE_MODELS`, `JUDGE_PROVIDER=openai`). It also
writes `/logs/verifier/insights_recall.json` with the per-insight verdict and
justification, which is what to read when diagnosing a low score.

Ground truth lives only in `tests/case.json`, which Harbor copies to the verifier and
never into the agent's workdir. Upstream's `qa_dict.json` answer sidecars and the
plaintext `.md` twins of the binary documents are deliberately not laid down.

Upstream also defines `factuality`, `report_quality`, and `distractor_recall`.
`insights_recall` is the headline metric and the only one implemented here; `factuality`
additionally needs an embedding model and the source documents in the verifier.

## Regenerating

The document corpus (~61 MB) and the invariant build/verifier files are git-ignored and
must be laid down before a run:

```bash
cd libs/evals
python -m harbor_adapters.drbench.main --populate datasets/drbench-evals
harbor run --path datasets/drbench-evals ...
```

To regenerate the committed task directories from the vendored configs:

```bash
python -m harbor_adapters.drbench.main --output-dir datasets/drbench-evals --all
```

See [`../../harbor_adapters/drbench/vendor/README.md`](../../harbor_adapters/drbench/vendor/README.md)
for what is vendored, the pinned upstream commit, and attribution.

## Known upstream data quirks

The adapter works around two defects in upstream's file manifest, both verified against
the pinned tree:

- One distractor in `DR0038` is declared with different filename casing than the file on
  disk, so it only loads on a case-insensitive filesystem.
- Some tasks declare two different documents at the same destination, or two documents
  whose names differ only in case. Upstream's loader overwrites; the adapter suffixes
  duplicates (`-2`) so no declared document is lost and the layout is identical on
  macOS and Linux.
