# Deep Agents Context-Hard Generation Pilot Design

## Goal

Generate five **Deep Agents-owned** Context-Hard candidates with Letta's
filesystem question generator. The candidates are quarantined for calibration:
they must not alter the existing Context-Bench `full` or `lite` selections,
the scorecard, CI, or any upstream Letta repository.

## Why this is needed

The original 100-task Letta `filesystem_cloud` pool is saturated for Luna.
On the 98 valid tasks, Luna achieved 1.000 pass@3 and 0.963 avg@3. A
defensible 30-task full benchmark and eight-task lite slice therefore require
a new candidate pool before task selection can resume.

## Scope

This pilot produces exactly five candidate records:

- three `multi_hop_chain` questions;
- two `multi_entity_comparison` questions.

The upstream Letta generator remains the question authoring engine. Deep
Agents supplies a local hard-mode policy, provenance recording, and quarantine
workflow around it. This is not a fork, contribution, or release of Letta's
benchmark.

## Recommended approach

Run a commit-pinned checkout of
[`letta-ai/letta-evals`](https://github.com/letta-ai/letta-evals) in a
temporary directory. Use its SQLite fixture, generator, prompt structure,
SQL verification, and validation routine, but apply a transient Deep
Agents-only hard-mode overlay. Do not vendor the complete generator into this
repository for the pilot.

This preserves upstream provenance while avoiding a new maintained copy of a
third-party application. It is preferable to reimplementing the generator,
which would stop being a Letta-generator pilot, and to permanently vendoring
it before the candidate quality is known.

## Hard-mode policy

Every accepted candidate must satisfy all of these conditions in addition to
Letta's normal registration and validation rules:

1. The type is `multi_hop_chain` or `multi_entity_comparison`.
2. It declares at least six distinct required filesystem files.
3. It records at least six reasoning SQL queries.
4. Its final verification query derives one exact answer end-to-end and
   returns exactly one row.
5. It contains two dependent relationship chains, each containing an indirect
   relationship, which converge in a final comparison or tiebreak.
6. It cannot be answered by intersecting a set of independently-greppable
   attributes.
7. Its normalized question is not an exact or near duplicate of either another
   pilot candidate or one of the original 100 cloud questions.

The generator prompt must state these requirements explicitly. A deterministic
post-generation policy validates the measurable rules; the relationship-graph
and non-parallelizability claims receive manual review during the five-task
pilot.

## Security and reproducibility boundaries

The generator model produces SQL, so it must run against a disposable copy of
Letta's SQLite database through a read-only query gate. The gate accepts only
a single `SELECT` or `WITH` statement and rejects statement separators,
DDL/DML, `PRAGMA`, `ATTACH`, and extension loading. The generation process
uses the `ANTHROPIC_API_KEY` only by environment-variable reference and never
writes it to output, logs, metadata, or Git.

Each candidate manifest records:

- the pinned upstream repository commit;
- the generator model and hard-mode policy version;
- the source database checksum;
- question type, required files, answer, and verification query;
- validation outcome and duplicate-screen outcome.

Full model conversation traces remain local and untracked. They are useful for
diagnosis but are not benchmark data and are unnecessarily large for the
repository.

## Quarantine output

Accepted candidate records live under a new Deep Agents-owned
`context-hard-candidates` area. The area is intentionally separate from
`datasets/context-retrieval-evals` and has no entry in `unified_prep.py`,
`lite_tasks.py`, or a workflow matrix.

The pilot does not generate Harbor task directories or run model calibration.
After human review confirms that the five candidates are genuinely hard and
well-formed, a follow-up design can generate the 70-task pool, convert it to
runnable tasks, and calibrate it down to 30 full tasks and eight lite tasks.

## Validation and acceptance criteria

The implementation must have deterministic tests for the Deep Agents hard-mode
validator, including rejection of a wrong type, fewer than six files, fewer
than six SQL queries, unsafe SQL, non-unique answers, and duplicate questions.

The pilot succeeds only when all five candidates:

- pass Letta's validator;
- pass the Deep Agents hard-mode validator;
- have one SQL-verified answer;
- are distinct from the original 100 and each other; and
- are manually reviewed before being retained as quarantine data.

No scorecard update is part of this pilot.
