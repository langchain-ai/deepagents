"""Local embedding model cache coverage."""

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from deepagents_talon import history_adapters, history_embeddings, speech
from deepagents_talon.config import TalonConfig
from deepagents_talon.history_adapters import BoundedEmbeddings, open_profile
from deepagents_talon.history_embeddings import HistoryEmbeddings
from deepagents_talon.history_profiles import EmbeddingProfile
from deepagents_talon.speech import DEFAULT_LOCAL_VOICE_TRANSCRIPTION_MODEL


@pytest.mark.parametrize("assistant_id", ["first", "second"])
async def test_local_models_share_cache(tmp_path: Path, monkeypatch, assistant_id: str) -> None:
    config = TalonConfig.from_env(
        {
            "DEEPAGENTS_TALON_HOME": str(tmp_path),
            "DEEPAGENTS_TALON_ASSISTANT_ID": assistant_id,
            "DEEPAGENTS_TALON_HISTORY_VECTOR_SEARCH": "true",
        }
    )
    vectors = [[1.0] * config.history_embedding_profile.dims]
    encoder = Mock()
    encoder.encode.return_value.tolist.return_value = vectors
    constructor = Mock(return_value=encoder)
    download = Mock(return_value=str(tmp_path / "snapshot"))
    modules = {
        "sentence_transformers": SimpleNamespace(SentenceTransformer=constructor),
        "huggingface_hub": SimpleNamespace(snapshot_download=download),
        "transformers": SimpleNamespace(pipeline=Mock(), AutoModel=Mock(), AutoProcessor=Mock()),
    }
    monkeypatch.setattr(history_embeddings.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(speech, "_local_pipelines", {})

    async with open_profile(config) as profile:
        assert profile is not None
        assert await profile.embed.aembed_documents(["hello"]) == vectors
    speech._load_local_pipeline(DEFAULT_LOCAL_VOICE_TRANSCRIPTION_MODEL, "cpu", config)

    cache = str(tmp_path / "cache" / "models" / "huggingface")
    assert constructor.call_args.kwargs["cache_folder"] == cache
    assert download.call_args.kwargs["cache_dir"] == cache
    assert constructor.call_args.kwargs["trust_remote_code"] is False


def test_embeddings_default_to_environment_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPAGENTS_TALON_HOME", str(tmp_path))
    constructor = Mock()
    constructor.return_value.encode.return_value.tolist.return_value = [[1.0]]
    monkeypatch.setattr(
        history_embeddings.importlib,
        "import_module",
        lambda _: SimpleNamespace(SentenceTransformer=constructor),
    )
    embeddings = HistoryEmbeddings()
    assert embeddings.embed_documents(["hello"]) == [[1.0]]
    assert constructor.call_args.kwargs["cache_folder"] == str(
        tmp_path / "cache" / "models" / "huggingface"
    )


@pytest.mark.parametrize("completed", [False, True])
async def test_cancelled_embedding_retry_reuses_inference(monkeypatch, *, completed: bool) -> None:
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    calls: list[list[str]] = []

    def encode(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=3)
        return [[1.0]]

    embeddings = HistoryEmbeddings()
    monkeypatch.setattr(embeddings, "embed_documents", encode)
    try:
        first = asyncio.create_task(embeddings.aembed_documents(["hello"]))
        await asyncio.wait_for(started.wait(), 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        if completed:
            release.set()
            await embeddings.aclose()
        retry = asyncio.create_task(embeddings.aembed_documents(["hello"]))
        await asyncio.sleep(0)
        release.set()
        assert await retry == [[1.0]]
        assert calls == [["hello"]]
        assert await embeddings.aembed_documents(["different"]) == [[1.0]]
        assert calls == [["hello"], ["different"]]
    finally:
        release.set()
        await embeddings.aclose()


async def test_failed_embedding_can_retry(monkeypatch) -> None:
    embeddings = HistoryEmbeddings()
    monkeypatch.setattr(
        embeddings, "embed_documents", Mock(side_effect=[RuntimeError("inference failed"), [[1.0]]])
    )
    with pytest.raises(RuntimeError, match="inference failed"):
        await embeddings.aembed_documents(["hello"])
    assert await embeddings.aembed_documents(["hello"]) == [[1.0]]


@pytest.mark.parametrize("adapter", ["local", "openai-compatible"])
async def test_document_deadline_depends_on_adapter(monkeypatch, adapter: str) -> None:
    async def encode(texts: list[str]) -> list[list[float]]:
        await asyncio.sleep(0)
        return [[1.0] for _ in texts]

    embeddings = HistoryEmbeddings()
    monkeypatch.setattr(embeddings, "aembed_documents", encode)
    monkeypatch.setattr(history_adapters, "_REQUEST_TIMEOUT", 0)
    bounded = BoundedEmbeddings(embeddings, EmbeddingProfile(adapter=adapter, dims=1))
    if adapter == "local":
        assert await bounded.aembed_documents(["hello"]) == [[1.0]]
    else:
        with pytest.raises(ExceptionGroup) as error:
            await bounded.aembed_documents(["hello"])
        assert isinstance(error.value.exceptions[0], TimeoutError)
