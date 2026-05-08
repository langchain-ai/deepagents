---
name: sample-data
description: Discover and select one or more user-uploaded CSV or data files from /uploads.
---

# Data Directory

Use this skill when selecting, inspecting, or explaining files in the data directory.

## Location

Analysis inputs should be discovered under:

`/uploads/`

Users are expected to upload CSVs or other supported data files for analysis. Do not assume a specific schema or business domain until you inspect the uploaded files.

## Discovery Workflow

1. List `/uploads/` before choosing files.
2. Identify supported files such as `.csv`, `.tsv`, `.json`, `.jsonl`, `.xlsx`, or `.parquet` when available.
3. For each relevant file, inspect the header/schema, row count, column names, and a small sample.
4. If there are multiple files, decide whether they should be analyzed separately, joined by keys, concatenated, or treated as unrelated inputs.
5. If the user's request does not specify a file and multiple plausible files exist, ask which dataset to use unless a safe default is obvious.

## Multiple Files

- Look for shared entity keys, dates, IDs, or naming conventions.
- Validate join cardinality and row counts before and after combining data.
- Do not join files solely because column names look similar; explain uncertainty when relationships are inferred.
- Keep file-level provenance in combined outputs so findings can be traced back to input files.

## Missing Data

If there are no uploaded files, ask the user to upload a CSV or supported data file before starting the analysis.
