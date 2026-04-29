---
name: sample-data
description: Discover and select one or more files from /memories/skills/data for analysis, including demo data and user-provided datasets.
---

# Data Directory

Use this skill when selecting, inspecting, or explaining files in the data directory.

## Location

All analysis inputs should be discovered under:

`/memories/skills/data/`

The directory can contain one or more data files. Do not assume a specific schema or business domain until you inspect the files. The bundled `sample_saas_metrics.csv` is only demo data.

## Discovery Workflow

1. List `/memories/skills/data/` before choosing files.
2. Identify supported files such as `.csv`, `.tsv`, `.json`, `.jsonl`, `.xlsx`, or `.parquet` when available.
3. For each relevant file, inspect the header/schema, row count, column names, and a small sample.
4. If there are multiple files, decide whether they should be analyzed separately, joined by keys, concatenated, or treated as unrelated inputs.
5. If the user's request does not specify a file and multiple plausible files exist, ask which dataset to use unless a safe default is obvious.

## Multiple Files

- Look for shared entity keys, dates, IDs, or naming conventions.
- Validate join cardinality and row counts before and after combining data.
- Do not join files solely because column names look similar; explain uncertainty when relationships are inferred.
- Keep file-level provenance in combined outputs so findings can be traced back to input files.

## Demo Dataset

If the user asks for a demo and no other file is specified, use:

`/memories/skills/data/sample_saas_metrics.csv`

It is synthetic SaaS metrics data and should not be treated as representative of arbitrary user data.
