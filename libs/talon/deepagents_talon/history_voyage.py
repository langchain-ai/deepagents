"""Voyage adapter for inputs already bounded by the history profile."""

from collections.abc import Generator
from typing import Self

import voyageai
from langchain_voyageai import VoyageAIEmbeddings
from pydantic import model_validator

_MAX_BATCH_TOKENS = 120_000


class HistoryVoyageEmbeddings(VoyageAIEmbeddings):
    """Use native async requests without downloading a tokenizer on the event loop."""

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Give both SDK request paths an explicit deadline and no nested retries."""
        self._client = voyageai.Client(
            api_key=self.voyage_api_key.get_secret_value(),
            base_url=self.base_url,
            timeout=30,
            max_retries=0,
        )
        self._aclient = voyageai.AsyncClient(
            api_key=self.voyage_api_key.get_secret_value(),
            base_url=self.base_url,
            timeout=30,
            max_retries=0,
        )
        return self

    def _build_batches(self, texts: list[str]) -> Generator[tuple[list[str], int], None, None]:
        # Upstream's batching downloads tokenizers synchronously, even for async calls.
        batch: list[str] = []
        size = 0
        for text in texts:
            weight = len(text.encode()) + 128
            if batch and (
                len(batch) >= (self.batch_size or 32) or size + weight > _MAX_BATCH_TOKENS
            ):
                yield batch, len(batch)
                batch, size = [], 0
            batch.append(text)
            size += weight
        if batch:
            yield batch, len(batch)
