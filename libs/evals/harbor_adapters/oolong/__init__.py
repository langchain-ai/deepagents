"""OOLONG-synth long-context aggregation as a Harbor dataset.

``loader`` fetches a ``(dataset, context_len)`` bucket from the HuggingFace
datasets-server, ``official_scorer`` is the verbatim upstream grader (MIT), and
``generate_oolong_tasks`` emits one self-contained Harbor task directory per
example. The generated task dirs are not committed; they are regenerated on
demand via ``python -m harbor_adapters.oolong.main --populate <dataset_dir>``
(run in CI, or locally with ``make oolong-populate``) before ``harbor run``.
"""
