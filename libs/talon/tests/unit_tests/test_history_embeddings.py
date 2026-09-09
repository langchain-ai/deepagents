"""Local embedding model cache coverage."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from deepagents_talon import history_embeddings, speech
from deepagents_talon.config import TalonConfig
from deepagents_talon.history_adapters import open_profile
from deepagents_talon.history_embeddings import HistoryEmbeddings
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
        "transformers": SimpleNamespace(pipeline=Mock()),
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
