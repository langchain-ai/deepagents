"""Data loading utilities for MemoryAgentBench integration.

Loads data from the `ai-hyz/MemoryAgentBench` HuggingFace dataset and
prepares it for consumption by the deepagents eval runner.

Adapted from https://github.com/HUST-AI-HYZ/MemoryAgentBench
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

HUGGINGFACE_DATASET = "ai-hyz/MemoryAgentBench"

SUPPORTED_SPLITS = frozenset(
    {
        "Accurate_Retrieval",
        "Test_Time_Learning",
        "Long_Range_Understanding",
        "Conflict_Resolution",
    }
)

_NLTK_TOKENIZER_READY = False


@dataclass(frozen=True)
class BenchmarkSample:
    """A single MemoryAgentBench evaluation sample.

    Attributes:
        context: The long-form text to be memorized.
        questions: List of questions to ask after memorization.
        answers: List of ground-truth answers corresponding to each question.
        source: Sub-dataset identifier (e.g. `factconsolidation_sh_6k`).
        qa_pair_ids: Optional identifiers for each QA pair.
    """

    context: str
    questions: list[str]
    answers: list[str | list[str]]
    source: str
    qa_pair_ids: list[str] = field(default_factory=list)


def load_benchmark_data(
    split: str,
    *,
    source_filter: str | None = None,
    max_samples: int | None = None,
) -> list[BenchmarkSample]:
    """Load MemoryAgentBench data from HuggingFace.

    Args:
        split: Dataset split name (e.g. `Conflict_Resolution`).
        source_filter: If set, only keep samples whose `metadata.source`
            matches this value exactly (e.g. `factconsolidation_sh_6k`).
        max_samples: Cap the number of returned samples.

    Returns:
        List of `BenchmarkSample` instances ready for evaluation.

    Raises:
        ValueError: If the split name is not recognized.
    """
    from datasets import load_dataset  # noqa: PLC0415

    if split not in SUPPORTED_SPLITS:
        msg = f"Unknown split {split!r}. Available: {sorted(SUPPORTED_SPLITS)}"
        raise ValueError(msg)

    dataset = load_dataset(HUGGINGFACE_DATASET, split=split, revision="main")
    logger.info("Loaded %d samples from %s/%s", len(dataset), HUGGINGFACE_DATASET, split)

    if source_filter is not None:
        dataset = dataset.filter(lambda row: row.get("metadata", {}).get("source", "") == source_filter)
        logger.info("Filtered to %d samples matching source=%r", len(dataset), source_filter)

    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    samples: list[BenchmarkSample] = []
    for row in dataset:
        questions = _ensure_list(row["questions"])
        answers = _ensure_list(row["answers"])
        metadata = row.get("metadata", {})
        qa_pair_ids = _ensure_list(metadata.get("qa_pair_ids", []))
        samples.append(
            BenchmarkSample(
                context=row["context"],
                questions=questions,
                answers=answers,
                source=metadata.get("source", ""),
                qa_pair_ids=qa_pair_ids,
            )
        )
    return samples


def chunk_text(text: str, *, chunk_size: int = 4096) -> list[str]:
    """Split text into chunks respecting sentence boundaries.

    Uses NLTK sentence tokenization and tiktoken to stay within token
    limits per chunk.

    Args:
        text: The document to split.
        chunk_size: Maximum number of tokens per chunk.

    Returns:
        List of text chunks.
    """
    import nltk  # noqa: PLC0415
    import tiktoken  # noqa: PLC0415

    _ensure_nltk_tokenizer(nltk)

    try:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    sentences = _sent_tokenize(nltk, text)

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        token_count = len(encoding.encode(sentence, allowed_special={"<|endoftext|>"}))

        if current_tokens + token_count > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_tokens = token_count
        else:
            current_sentences.append(sentence)
            current_tokens += token_count

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def _ensure_list(value: object) -> list:
    """Coerce a value into a list if it isn't one already.

    Args:
        value: Value to normalize.

    Returns:
        The value wrapped in a list, or the original list.
    """
    if isinstance(value, list):
        return value
    if value:
        return [value]
    return []


def _ensure_nltk_tokenizer(nltk_module: object) -> None:
    """Ensure the NLTK sentence tokenizer is available once per process."""
    global _NLTK_TOKENIZER_READY  # noqa: PLW0603
    if _NLTK_TOKENIZER_READY:
        return

    # Newer NLTK versions use `punkt_tab`; older versions use `punkt`.
    downloader = getattr(nltk_module, "download", None)
    if not callable(downloader):
        logger.warning("NLTK downloader is unavailable; sentence splitting may be less accurate.")
        return

    for resource in ("punkt_tab", "punkt"):
        try:
            ok = bool(downloader(resource, quiet=True))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to download NLTK tokenizer resource %s: %s", resource, exc)
        else:
            if ok:
                _NLTK_TOKENIZER_READY = True
                break


def _sent_tokenize(nltk_module: object, text: str) -> list[str]:
    """Sentence-tokenize text with a robust fallback for constrained environments."""
    try:
        return list(nltk_module.sent_tokenize(text))  # type: ignore[reportUnknownMemberType,attr-defined]
    except Exception:  # noqa: BLE001
        logger.warning("Falling back to newline-based chunking; NLTK sentence tokenizer unavailable.")
        return [line for line in text.splitlines() if line.strip()] or [text]
