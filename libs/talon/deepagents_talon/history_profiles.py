"""Validated, credential-free identities for optional history embeddings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from deepagents_talon.archive import CHUNK_SIZE
from deepagents_talon.config import TalonConfigError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from langchain_core.embeddings import Embeddings
    from langgraph.store.base import IndexConfig

LOCAL_MODEL = "Qwen/Qwen3-Embedding-0.6B"
LOCAL_PROMPT = "Instruct: Retrieve past conversation passages relevant to the query.\nQuery: "
_PREFIX = "DEEPAGENTS_TALON_HISTORY_EMBED_"
_MAX_MODEL_LENGTH = 256
_SERVER_INPUT_BOUND = CHUNK_SIZE * 4 + 128
_ADAPTERS = {"local", "voyage", "openai-compatible", "atlas"}


@dataclass(frozen=True)
class EmbeddingProfile:
    """Embedding identity and resource limits, independent of the archive backend.

    Warning:
        Experimental API; subject to change with the Talon runtime.
    """

    adapter: str = "local"
    model: str = LOCAL_MODEL
    dims: int = 1024
    max_input_tokens: int = 8192
    batch_size: int = 4
    concurrency: int = 1
    query_prompt: str = LOCAL_PROMPT
    base_url: str = ""
    query_model: str = ""
    embed: Embeddings | None = field(default=None, repr=False, compare=False)

    @property
    def client_side(self) -> bool:
        """Whether Talon computes vectors before storing them."""
        return self.adapter != "atlas"

    @property
    def fingerprint(self) -> str:
        """Identify the vector space and preprocessing without including credentials."""
        identity = (
            self.adapter,
            self.model,
            self.dims,
            self.max_input_tokens,
            self.query_prompt,
            self.base_url,
            self.query_model,
            "utf8-weighted-mean-v1",
        )
        return hashlib.sha256(json.dumps(identity).encode()).hexdigest()

    @property
    def index(self) -> IndexConfig:
        """Build the common Store index configuration after opening the adapter."""
        if self.embed is None:
            msg = "History embedding profile has not been opened"
            raise TalonConfigError(msg)
        return {"embed": self.embed, "dims": self.dims, "fields": ["text"]}


def parse_profile(env: Mapping[str, str]) -> EmbeddingProfile:
    """Validate embedding configuration without importing providers or reading keys."""
    adapter = env.get(_PREFIX + "ADAPTER", "local")
    if adapter not in _ADAPTERS:
        msg = f"{_PREFIX}ADAPTER must be local, voyage, openai-compatible, or atlas"
        raise TalonConfigError(msg)
    local = adapter == "local"
    model = env.get(_PREFIX + "MODEL", LOCAL_MODEL if local else "")
    if not model or len(model) > _MAX_MODEL_LENGTH or any(char.isspace() for char in model):
        msg = f"{_PREFIX}MODEL must name an embedding model"
        raise TalonConfigError(msg)
    profile = EmbeddingProfile(
        adapter=adapter,
        model=model,
        dims=_number(env, "DIMS", 1024 if local or adapter == "atlas" else None, 8192),
        max_input_tokens=_number(env, "MAX_INPUT_TOKENS", 8192 if local else None, 131072),
        batch_size=_number(env, "BATCH_SIZE", 4 if local else 32, 4 if local else 96),
        concurrency=_number(env, "CONCURRENCY", 1 if local else 4, 1 if local else 16),
        query_prompt=env.get(
            _PREFIX + "QUERY_PROMPT", LOCAL_PROMPT if local and model == LOCAL_MODEL else ""
        ),
        base_url=_endpoint(env, adapter),
        query_model=env.get(_PREFIX + "QUERY_MODEL", ""),
    )
    _validate_profile(profile)
    return profile


def _number(env: Mapping[str, str], name: str, default: int | None, maximum: int) -> int:
    try:
        value = int(env.get(_PREFIX + name, str(default)))
        if 1 <= value <= maximum:
            return value
    except ValueError:
        pass
    msg = f"{_PREFIX}{name} must be an integer between 1 and {maximum}"
    raise TalonConfigError(msg)


def _endpoint(env: Mapping[str, str], adapter: str) -> str:
    url = env.get(_PREFIX + "BASE_URL", "")
    if not url:
        return "https://api.openai.com/v1" if adapter == "openai-compatible" else ""
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme == "https"
            and parsed.hostname
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
            and not any(char.isspace() for char in url)
        ):
            _ = parsed.port
            return url.rstrip("/")
    except ValueError:
        pass
    msg = f"{_PREFIX}BASE_URL must be an HTTPS URL without credentials, query, or fragment"
    raise TalonConfigError(msg)


def _validate_profile(profile: EmbeddingProfile) -> None:
    if len(profile.query_prompt.encode()) + 132 >= profile.max_input_tokens:
        msg = "History embedding token budget must leave room for query text"
        raise TalonConfigError(msg)
    if profile.adapter in {"voyage", "atlas"} and profile.dims not in {256, 512, 1024, 2048}:
        msg = "Voyage embedding dimensions must be 256, 512, 1024, or 2048"
        raise TalonConfigError(msg)
    if profile.adapter == "voyage" and "context" in profile.model:
        msg = "History requires independent document embeddings, not contextualized Voyage models"
        raise TalonConfigError(msg)
    if profile.adapter == "atlas" and profile.max_input_tokens < _SERVER_INPUT_BOUND:
        msg = (
            "Atlas token budget must fit a complete archive chunk "
            f"({_SERVER_INPUT_BOUND} tokens conservatively)"
        )
        raise TalonConfigError(msg)
    if profile.query_model and (
        len(profile.query_model) > _MAX_MODEL_LENGTH
        or any(char.isspace() for char in profile.query_model)
    ):
        msg = "History query model must name an embedding model"
        raise TalonConfigError(msg)
    if profile.adapter == "atlas" and (profile.query_prompt or profile.base_url):
        msg = "Atlas manages embedding prompts and endpoints server-side"
        raise TalonConfigError(msg)
    if profile.query_model and profile.adapter != "atlas":
        msg = "A separate history query model is supported only by Atlas"
        raise TalonConfigError(msg)
