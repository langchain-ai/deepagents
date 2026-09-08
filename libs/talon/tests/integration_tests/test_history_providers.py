"""Explicit opt-in smoke tests; provider calls can incur embedding charges."""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage

from deepagents_talon.archive import ArchiveScope
from deepagents_talon.config import TalonConfig
from deepagents_talon.history_adapters import open_profile
from deepagents_talon.history_backends import open_history


@pytest.mark.skipif(
    os.environ.get("TALON_TEST_HISTORY_PROVIDER") != "1", reason="Embedding API opt-in"
)
async def test_configured_embedding_provider(tmp_path):
    config = TalonConfig.from_env(os.environ, base_home=tmp_path)
    assert config.history_vector_search
    async with open_profile(config) as profile:
        assert profile.client_side
        documents = await profile.embed.aembed_documents(["A bicycle has two wheels."])
        query = await profile.embed.aembed_query("How many wheels does a bicycle have?")
        assert len(documents) == 1
        assert len(documents[0]) == len(query) == profile.dims


@pytest.mark.skipif(
    os.environ.get("TALON_TEST_ATLAS_EMBEDDINGS") != "1", reason="Atlas embedding opt-in"
)
@pytest.mark.timeout(120)
async def test_atlas_automated_embeddings(tmp_path):
    config = TalonConfig.from_env(
        {
            **os.environ,
            "DEEPAGENTS_TALON_ASSISTANT_ID": "history-smoke-" + uuid4().hex,
            "DEEPAGENTS_TALON_HISTORY_VECTOR_SEARCH": "1",
            "DEEPAGENTS_TALON_HISTORY_EMBED_ADAPTER": "atlas",
        },
        base_home=tmp_path,
    )
    scope = ArchiveScope(talon_history_channel="test", talon_history_chat="synthetic")
    async with open_history(config) as archive:
        try:
            await archive.append(
                scope, "session", "time", [HumanMessage("A bicycle has two wheels.")]
            )
            async with asyncio.timeout(90):
                while True:
                    page = await archive.search_page(scope, query="pedal-powered transportation")
                    if (
                        page["semantic_status"] == "completed"
                        and page["results"]
                        and not page["indexing_pending"]
                    ):
                        break
                    await asyncio.sleep(0.2)
            assert page["results"][0]["session_id"] == "session"
        finally:
            await archive.delete_session("session")
