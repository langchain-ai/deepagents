"""Dataset configurations for MemoryAgentBench.

Each config mirrors one of the YAML files from the original benchmark at
https://github.com/HUST-AI-HYZ/MemoryAgentBench/tree/main/configs/data_conf

Only the fields used by our adapter are included.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    """Minimal configuration for one MemoryAgentBench sub-dataset.

    Attributes:
        split: HuggingFace dataset split name (e.g. `Conflict_Resolution`).
        source: Value matched against `metadata.source` in the dataset.
        chunk_size: Token budget per text chunk during memorization.
        max_samples: Maximum number of context samples to evaluate.
        max_questions: Cap on questions asked per sample. When `None`,
            all questions in the dataset are used.
    """

    split: str
    source: str
    chunk_size: int = 4096
    max_samples: int = 1
    max_questions: int | None = None


# -- Conflict Resolution (single-hop) ----------------------------------------

CR_SH_6K = DatasetConfig(
    split="Conflict_Resolution",
    source="factconsolidation_sh_6k",
    max_samples=1,
)
CR_SH_32K = DatasetConfig(
    split="Conflict_Resolution",
    source="factconsolidation_sh_32k",
    max_samples=1,
)
CR_SH_64K = DatasetConfig(
    split="Conflict_Resolution",
    source="factconsolidation_sh_64k",
    max_samples=1,
)
CR_SH_262K = DatasetConfig(
    split="Conflict_Resolution",
    source="factconsolidation_sh_262k",
    max_samples=1,
)

# -- Conflict Resolution (multi-hop) -----------------------------------------

CR_MH_6K = DatasetConfig(
    split="Conflict_Resolution",
    source="factconsolidation_mh_6k",
    max_samples=1,
)
CR_MH_32K = DatasetConfig(
    split="Conflict_Resolution",
    source="factconsolidation_mh_32k",
    max_samples=1,
)
CR_MH_64K = DatasetConfig(
    split="Conflict_Resolution",
    source="factconsolidation_mh_64k",
    max_samples=1,
)
CR_MH_262K = DatasetConfig(
    split="Conflict_Resolution",
    source="factconsolidation_mh_262k",
    max_samples=1,
)

# -- Test-Time Learning (ICL) ------------------------------------------------

TTL_BANKING77 = DatasetConfig(
    split="Test_Time_Learning",
    source="icl_banking77_5900shot_balance",
    max_samples=1,
)
TTL_CLINIC150 = DatasetConfig(
    split="Test_Time_Learning",
    source="icl_clinic150_7050shot_balance",
    max_samples=1,
)
TTL_NLU = DatasetConfig(
    split="Test_Time_Learning",
    source="icl_nlu_3000shot_balance",
    max_samples=1,
)
TTL_TREC_COARSE = DatasetConfig(
    split="Test_Time_Learning",
    source="icl_trec_coarse_2700shot_balance",
    max_samples=1,
)
TTL_TREC_FINE = DatasetConfig(
    split="Test_Time_Learning",
    source="icl_trec_fine_2700shot_balance",
    max_samples=1,
)
TTL_RECSYS = DatasetConfig(
    split="Test_Time_Learning",
    source="Recsys_redial_full",
    max_samples=1,
)

# -- Accurate Retrieval -------------------------------------------------------

AR_RULER_QA1 = DatasetConfig(
    split="Accurate_Retrieval",
    source="ruler_qa1_197K",
    max_samples=1,
)
AR_RULER_QA2 = DatasetConfig(
    split="Accurate_Retrieval",
    source="ruler_qa2_421k",
    max_samples=1,
)
AR_LONGMEMEVAL = DatasetConfig(
    split="Accurate_Retrieval",
    source="longmemeval_s_-1_500",
    max_samples=1,
)
AR_LONGMEMEVAL_STAR = DatasetConfig(
    split="Accurate_Retrieval",
    source="longmemeval_s_star_-1_500",
    max_samples=1,
)
AR_EVENTQA_FULL = DatasetConfig(
    split="Accurate_Retrieval",
    source="eventqa_full",
    max_samples=1,
)
AR_EVENTQA_64K = DatasetConfig(
    split="Accurate_Retrieval",
    source="eventqa_64k",
    max_samples=1,
)
AR_EVENTQA_128K = DatasetConfig(
    split="Accurate_Retrieval",
    source="eventqa_128k",
    max_samples=1,
)

# -- Long Range Understanding -------------------------------------------------

LRU_INFBENCH_SUM = DatasetConfig(
    split="Long_Range_Understanding",
    source="infbench_sum",
    max_samples=1,
)
LRU_DETECTIVE_QA = DatasetConfig(
    split="Long_Range_Understanding",
    source="detective_qa",
    max_samples=1,
)


# -- Convenience collections --------------------------------------------------

CONFLICT_RESOLUTION_CONFIGS: list[DatasetConfig] = [
    CR_SH_6K,
    CR_SH_32K,
    CR_SH_64K,
    CR_SH_262K,
    CR_MH_6K,
    CR_MH_32K,
    CR_MH_64K,
    CR_MH_262K,
]

TEST_TIME_LEARNING_CONFIGS: list[DatasetConfig] = [
    TTL_BANKING77,
    TTL_CLINIC150,
    TTL_NLU,
    TTL_TREC_COARSE,
    TTL_TREC_FINE,
    TTL_RECSYS,
]

ACCURATE_RETRIEVAL_CONFIGS: list[DatasetConfig] = [
    AR_RULER_QA1,
    AR_RULER_QA2,
    AR_LONGMEMEVAL,
    AR_LONGMEMEVAL_STAR,
    AR_EVENTQA_FULL,
    AR_EVENTQA_64K,
    AR_EVENTQA_128K,
]

LONG_RANGE_UNDERSTANDING_CONFIGS: list[DatasetConfig] = [
    LRU_INFBENCH_SUM,
    LRU_DETECTIVE_QA,
]

ALL_CONFIGS: list[DatasetConfig] = (
    CONFLICT_RESOLUTION_CONFIGS + TEST_TIME_LEARNING_CONFIGS + ACCURATE_RETRIEVAL_CONFIGS + LONG_RANGE_UNDERSTANDING_CONFIGS
)

# Small / cheap subset for regular CI runs
CI_CONFIGS: list[DatasetConfig] = [
    DatasetConfig(split=CR_SH_6K.split, source=CR_SH_6K.source, max_questions=10),
    DatasetConfig(split=CR_MH_6K.split, source=CR_MH_6K.source, max_questions=10),
    DatasetConfig(split=TTL_BANKING77.split, source=TTL_BANKING77.source, max_questions=10),
]
