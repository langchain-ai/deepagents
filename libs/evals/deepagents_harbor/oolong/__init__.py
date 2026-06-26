"""Generate Harbor eval tasks from the OOLONG-synth long-context benchmark.

This package turns the OOLONG-synth HuggingFace dataset into a Harbor dataset:
``loader`` fetches a ``(dataset, context_len)`` bucket, ``official_scorer`` is the
verbatim upstream grader, and ``generate_oolong_tasks`` emits one self-contained
Harbor task directory per example. See ``README.md``.
"""
