"""Validated, credential-free identities for optional history embeddings."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import socket
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
_INSTRUCTION_FAMILY = "qwen3-embedding"
_RESERVED_TOKENS = 128


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
    bytes_per_token: int = 1
    send_dimensions: bool = True
    query_prompt: str = LOCAL_PROMPT
    base_url: str = ""
    query_model: str = ""
    embed: Embeddings | None = field(default=None, repr=False, compare=False)

    @property
    def client_side(self) -> bool:
        """Whether Talon computes vectors before storing them."""
        return self.adapter != "atlas"

    @property
    def input_budget(self) -> int:
        """Bound one provider request in UTF-8 bytes, reserving room for instructions.

        `bytes_per_token` converts the provider's token limit into the byte measure
        used for splitting. It defaults to the worst case of one token per byte, so
        raising it trades a safety margin for fewer splits on non-ASCII transcripts.
        """
        return (self.max_input_tokens - _RESERVED_TOKENS) * self.bytes_per_token

    @property
    def fingerprint(self) -> str:
        """Identify stored document vectors, excluding credentials and query-time settings.

        `query_prompt` and `query_model` are deliberately absent: both apply only when
        embedding a query, so changing either cannot invalidate a stored vector and must
        not force callers to rebuild the index.
        """
        identity = (
            self.adapter,
            self.model,
            self.dims,
            self.max_input_tokens,
            self.base_url,
            # Splitting and pooling run only where Talon computes vectors, so their
            # settings cannot change what a server-side adapter already stored.
            [self.bytes_per_token, "utf8-weighted-mean-v1"] if self.client_side else [],
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
        bytes_per_token=_number(env, "BYTES_PER_TOKEN", 1, 4),
        send_dimensions=_flag(env, "SEND_DIMENSIONS", default=True),
        query_prompt=env.get(_PREFIX + "QUERY_PROMPT", _default_prompt(adapter, model)),
        base_url=_endpoint(env, adapter),
        query_model=env.get(_PREFIX + "QUERY_MODEL", ""),
    )
    _validate_profile(profile)
    return profile


def _default_prompt(adapter: str, model: str) -> str:
    """Apply the instruction format a model was trained with, however it is reached.

    Qwen3-Embedding expects an instruction prefix on queries and none on documents, and
    that convention belongs to the model rather than to the adapter serving it, so the
    same model reached through a hosted OpenAI-compatible endpoint needs it too. Atlas
    embeds server-side and never sees a client prompt.
    """
    if adapter == "atlas":
        return ""
    return LOCAL_PROMPT if _INSTRUCTION_FAMILY in model.lower() else ""


def _flag(env: Mapping[str, str], name: str, *, default: bool) -> bool:
    value = env.get(_PREFIX + name, "1" if default else "0").strip().lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    msg = f"{_PREFIX}{name} must be a boolean"
    raise TalonConfigError(msg)


def _number(env: Mapping[str, str], name: str, default: int | None, maximum: int) -> int:
    raw = env.get(_PREFIX + name)
    if raw is None:
        if default is None:
            msg = f"{_PREFIX}{name} is required for this adapter"
            raise TalonConfigError(msg)
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if 1 <= value <= maximum:
        return value
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
            if _routable(parsed.hostname):
                return url.rstrip("/")
    except ValueError:
        pass
    msg = (
        f"{_PREFIX}BASE_URL must be an HTTPS URL without credentials, query, or fragment, "
        "naming a routable host"
    )
    raise TalonConfigError(msg)


def _routable(hostname: str) -> bool:
    """Reject endpoints that address this host or its private network.

    The configured endpoint receives the provider API key, so a misconfigured
    base URL should not deliver it to a metadata service or an internal listener.
    Only address literals and `localhost` are checked: a public name that resolves
    to a private address still connects, which needs resolution-time control the
    embedding clients do not expose.
    """
    if hostname.lower() in {"localhost", "localhost."}:
        return False
    address = _address(hostname)
    if address is None:
        return True
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def _address(hostname: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Resolve an address literal, including the abbreviated IPv4 forms.

    `ipaddress` accepts only dotted quads, but the C resolver behind every HTTP
    client also accepts `127.1`, `2130706433`, `0177.0.0.1`, and `0x7f000001`, all
    of which reach the loopback interface. Treating those as hostnames would let
    them past the routability check.
    """
    try:
        return ipaddress.ip_address(hostname.strip("[]"))
    except ValueError:
        pass
    try:
        return ipaddress.ip_address(socket.inet_aton(hostname))
    except (OSError, ValueError):
        return None


def _validate_profile(profile: EmbeddingProfile) -> None:
    if len(profile.query_prompt.encode()) + 4 >= profile.input_budget:
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
