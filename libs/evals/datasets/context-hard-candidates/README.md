# Context-Hard calibration quarantine

This directory is a Deep Agents-owned calibration quarantine for generated
Context-Hard candidates. It is not part of either Unified `full` or `lite`
profile, and it does not alter the existing Context-Bench benchmark.

Each retained candidate set records the pinned Letta generator revision, model,
source-database hash, and Deep Agents hard-policy validation in its manifest.
Candidate metadata identifies the source as `deepagents-context-hard`; Letta is
not recorded as the candidate author.

Every five-candidate calibration set must be manually reviewed before it is
retained or used to inform a future 70-candidate generation. This quarantine is
not runnable benchmark data and must not be added to Unified selection,
scorecards, or workflows.
