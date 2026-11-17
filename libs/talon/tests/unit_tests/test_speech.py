"""Exercise local speech loading and generation without downloading a model."""

import shutil
from typing import TYPE_CHECKING

import pytest

from deepagents_talon import speech
from deepagents_talon.config import TalonConfig

if TYPE_CHECKING:
    from pathlib import Path


def test_local_pipeline_transcribes_with_local_only_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("librosa")
    hub = pytest.importorskip("huggingface_hub")
    config = transformers.ParakeetTDTConfig(
        encoder_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "subsampling_conv_channels": 4,
        },
        vocab_size=4,
        blank_token_id=3,
        decoder_start_token_id=3,
        decoder_hidden_size=8,
        num_decoder_layers=1,
        max_symbols_per_step=1,
    )
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = transformers.AutoModel.from_config(config)
    model.generation_config.num_beams = 1
    model.generation_config.max_new_tokens = 2
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizers.Tokenizer(
            tokenizers.models.WordLevel({"hello": 0, "world": 1, "<pad>": 2, "<blank>": 3})
        ),
        pad_token="<pad>",  # noqa: S106  # tokenizer symbol, not a password
    )
    processor = transformers.ParakeetProcessor(transformers.ParakeetFeatureExtractor(), tokenizer)
    repo = tmp_path / "repo"
    model.save_pretrained(repo)
    processor.save_pretrained(repo)
    (repo / "model.nemo").write_bytes(b"unused duplicate weights")
    snapshot = tmp_path / "snapshot"

    def download(**kwargs: object) -> str:
        # Mirror the Hub's filtering so the load below proves the allowlist suffices.
        names = [path.name for path in repo.iterdir()]
        for name in hub.utils.filter_repo_objects(names, allow_patterns=kwargs["allow_patterns"]):
            snapshot.mkdir(exist_ok=True)
            shutil.copy(repo / name, snapshot / name)
        return str(snapshot)

    monkeypatch.setattr(hub, "snapshot_download", download)
    monkeypatch.setattr(speech, "_local_pipelines", {})

    pipeline = speech._load_local_pipeline(
        speech.DEFAULT_LOCAL_VOICE_TRANSCRIPTION_MODEL,
        "cpu",
        TalonConfig.from_env({"DEEPAGENTS_TALON_HOME": str(tmp_path)}),
    )
    result = pipeline(np.zeros(1600, dtype=np.float32))

    assert isinstance(result["text"], str)
    assert not (snapshot / "model.nemo").exists()
