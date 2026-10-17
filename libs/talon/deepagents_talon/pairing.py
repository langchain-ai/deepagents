"""Sender pairing: let an operator admit a new sender without editing env.

Talon is an experimental runtime and is subject to change or removal at any time.

An unknown sender who DMs a channel with pairing enabled receives a short code
bound to that (channel, sender). The operator approves the code from their own
DM with `/pair approve <code>` or from the `deepagents-talon pairing` CLI. Codes
are only ever accepted on those operator surfaces, so a stranger has nowhere to
submit guesses; the code exists so the operator can tell that this sender id
belongs to the person who contacted them out of band.

A paired sender is admitted the way an env-configured operator is: in DMs and
in every chat the bot can see, with the operator's credentials and host access.
Pairing never makes anyone an operator of Talon's own controls, so a paired
sender cannot run `/pair` or change tool approval policy.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import secrets
import stat
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_talon.mcp_config import locked_path
from deepagents_talon.observability import log_debug_event

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from deepagents_talon.cron.jobs import CronJob, CronJobStore
    from deepagents_talon.interfaces import ChannelMessage, ChannelReaction, SendResult

logger = logging.getLogger(__name__)

PAIRING_FILENAME = "pairing.json"
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
"""Uppercase letters and digits without the easily confused `0 O 1 I L`."""

CODE_LENGTH = 8
CODE_TTL_SECONDS = 3600
MAX_PENDING_PER_CHANNEL = 16
MAX_APPROVED_PER_CHANNEL = 1024
PAIRING_CHANNELS = ("discord", "slack", "telegram")
"""Providers that support sender pairing. WhatsApp is excluded because it runs on
the operator's own account, so it would reply to everyone who texts the operator."""

_VERSION = 1
_MAX_BYTES = 1_048_576
_MAX_ID_CHARS = 256
_TRUTHY = frozenset({"1", "true", "yes", "on", "enabled"})
_FALSY = frozenset({"0", "false", "no", "off", "disabled"})


class PairingStoreError(ValueError):
    """Raised when the persisted pairing store is invalid."""


@dataclass(frozen=True, slots=True)
class PendingRequest:
    """One unapproved pairing request.

    Args:
        sender_id: Channel-specific id of the requesting sender.
        code: Code the operator must supply to approve this request.
        conversation_id: DM conversation the request arrived in.
        created_at: Epoch seconds when the code was issued.
        expires_at: Epoch seconds after which the code no longer approves.
    """

    sender_id: str
    code: str
    conversation_id: str
    created_at: int
    expires_at: int


@dataclass(frozen=True, slots=True)
class PairedSender:
    """One approved sender.

    Args:
        sender_id: Channel-specific id of the approved sender.
        conversation_id: DM conversation the sender paired from.
        approved_at: Epoch seconds when the operator approved the sender.
    """

    sender_id: str
    conversation_id: str
    approved_at: int


@dataclass(frozen=True, slots=True)
class _ChannelState:
    pending: Mapping[str, PendingRequest]
    approved: Mapping[str, PairedSender]


_EMPTY = _ChannelState(pending={}, approved={})


def generate_code() -> str:
    """Return a fresh pairing code from a cryptographically secure source.

    Returns:
        `CODE_LENGTH` characters drawn from `CODE_ALPHABET`.
    """
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))


def format_code(code: str) -> str:
    """Return `code` split in two for readability, for example `K7QM-3XRD`.

    Args:
        code: Normalized pairing code.

    Returns:
        The code with a hyphen in the middle.
    """
    half = len(code) // 2
    return f"{code[:half]}-{code[half:]}"


def normalize_code(value: str) -> str:
    """Normalize operator-typed code text for comparison.

    Args:
        value: Code as typed, in any case, with optional hyphens or spaces.

    Returns:
        Uppercase code without separators.
    """
    return "".join(char for char in value.upper() if char not in "- ")


def is_direct_message(message: ChannelMessage) -> bool:
    """Return whether a channel message arrived in a one-to-one DM.

    Args:
        message: Inbound message from a pairing-capable adapter.

    Returns:
        `True` for a Discord DM or a Telegram private chat.
    """
    metadata = message.metadata
    return metadata.get("is_dm") is True or metadata.get("chat_type") == "private"


class PairingStore:
    """Persist pending and approved senders for every channel of one assistant.

    Every mutation is a single read-modify-write under an exclusive sidecar
    lock, and the file is replaced atomically, so a concurrent CLI and host
    cannot lose each other's updates. Reads for admission are cached against
    the file's stat identity; an atomic replace always changes the inode, so a
    revocation is never masked by the cache.
    """

    def __init__(self, path: Path) -> None:
        """Fix the parent directory without following the final path component.

        Args:
            path: Location of the pairing JSON file.
        """
        self._path = path.parent.resolve() / path.name
        self._cache: tuple[tuple[int, int, int], dict[str, _ChannelState]] | None = None

    @property
    def path(self) -> Path:
        """Location of the pairing JSON file."""
        return self._path

    def read(self) -> dict[str, _ChannelState]:
        """Read and validate the store; a missing file is an empty store.

        Returns:
            Channel states keyed by provider.

        Raises:
            OSError: If the file cannot be opened or is not a regular file.
            PairingStoreError: If the file is too large or its content is invalid.
        """
        try:
            descriptor = os.open(self._path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return {}
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                msg = "Pairing store must be a regular file."
                raise PairingStoreError(msg)
            raw = stream.read(_MAX_BYTES + 1)
        if len(raw) > _MAX_BYTES:
            msg = "Pairing store is too large."
            raise PairingStoreError(msg)
        return _decode(raw)

    def paired(self, provider: str, sender_id: str) -> PairedSender | None:
        """Return the approved record for `sender_id` on `provider`, if any.

        Fails closed: an unreadable or invalid store admits nobody, so only
        env-configured senders keep access until the operator repairs it.

        Args:
            provider: Channel provider key, for example `discord`.
            sender_id: Channel-specific sender id.

        Returns:
            The sender's record when the store is readable and lists them.
        """
        try:
            channels = self._cached_read()
        except (OSError, PairingStoreError):
            logger.warning("Cannot read sender pairing store %s", self._path, exc_info=True)
            return None
        return channels.get(provider, _EMPTY).approved.get(sender_id)

    def is_paired(self, provider: str, sender_id: str) -> bool:
        """Return whether `sender_id` is approved on `provider`.

        Args:
            provider: Channel provider key.
            sender_id: Channel-specific sender id.

        Returns:
            `True` only when the store is readable and lists the sender.
        """
        return self.paired(provider, sender_id) is not None

    def state(self, provider: str, *, now: int) -> _ChannelState:
        """Return the live pending requests and approved senders for one channel.

        Args:
            provider: Channel provider key.
            now: Current epoch seconds; expired requests are left out.

        Returns:
            Unexpired pending requests and all approved senders.
        """
        current = self.read().get(provider, _EMPTY)
        return _ChannelState(pending=_unexpired(current.pending, now), approved=current.approved)

    def request(
        self, provider: str, sender_id: str, conversation_id: str, *, now: int
    ) -> str | None:
        """Record a pairing request and return its code when one is newly issued.

        Args:
            provider: Channel provider key.
            sender_id: Requesting sender.
            conversation_id: DM conversation the request arrived in.
            now: Current epoch seconds.

        Returns:
            A new code, or `None` when the sender is already paired, already has
            a live request, or the channel's pending capacity is full.
        """
        with locked_path(self._path):
            channels = self.read()
            current = channels.get(provider, _EMPTY)
            pending = _unexpired(current.pending, now)
            if sender_id in current.approved or sender_id in pending:
                return None
            if len(pending) >= MAX_PENDING_PER_CHANNEL:
                log_debug_event(logger, "pairing.request.dropped", reason="capacity")
                return None
            code = _unique_code(pending)
            pending[sender_id] = PendingRequest(
                sender_id, code, conversation_id, now, now + CODE_TTL_SECONDS
            )
            channels[provider] = _ChannelState(pending=pending, approved=current.approved)
            self._write(channels)
        return code

    def approve(self, provider: str, code: str, *, now: int) -> PairedSender | None:
        """Consume a live code and approve the sender it was issued to.

        Args:
            provider: Channel provider key; codes never cross channels.
            code: Code as typed by the operator.
            now: Current epoch seconds.

        Returns:
            The newly approved sender, or `None` when no live request matches.
        """
        candidate = normalize_code(code).encode()
        with locked_path(self._path):
            channels = self.read()
            current = channels.get(provider, _EMPTY)
            pending = _unexpired(current.pending, now)
            match = _match_code(pending, candidate)
            if match is None or len(current.approved) >= MAX_APPROVED_PER_CHANNEL:
                return None
            del pending[match.sender_id]
            paired = PairedSender(match.sender_id, match.conversation_id, now)
            approved = {**current.approved, paired.sender_id: paired}
            channels[provider] = _ChannelState(pending=pending, approved=approved)
            self._write(channels)
        return paired

    def revoke(self, provider: str, sender_id: str) -> PairedSender | None:
        """Remove an approved sender, and any pending request they hold.

        Args:
            provider: Channel provider key.
            sender_id: Sender to revoke.

        Returns:
            The revoked sender, or `None` when they were not paired.
        """
        with locked_path(self._path):
            channels = self.read()
            current = channels.get(provider, _EMPTY)
            revoked = current.approved.get(sender_id)
            if revoked is None:
                return None
            approved = {key: value for key, value in current.approved.items() if key != sender_id}
            pending = {key: value for key, value in current.pending.items() if key != sender_id}
            channels[provider] = _ChannelState(pending=pending, approved=approved)
            self._write(channels)
        return revoked

    def _cached_read(self) -> dict[str, _ChannelState]:
        try:
            info = self._path.lstat()
        except FileNotFoundError:
            self._cache = None
            return {}
        identity = (info.st_mtime_ns, info.st_size, info.st_ino)
        if self._cache is not None and self._cache[0] == identity:
            return self._cache[1]
        channels = self.read()
        self._cache = (identity, channels)
        return channels

    def _write(self, channels: Mapping[str, _ChannelState]) -> None:
        raw = (json.dumps(_encode(channels), indent=2, sort_keys=True) + "\n").encode()
        if len(raw) > _MAX_BYTES:
            msg = "Pairing store is too large."
            raise PairingStoreError(msg)
        if self._path.is_symlink():
            msg = "Pairing store must not be a symlink."
            raise PairingStoreError(msg)
        descriptor, temporary = tempfile.mkstemp(prefix=".pairing-", dir=self._path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            Path(temporary).replace(self._path)
        finally:
            Path(temporary).unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class SenderPairing:
    """Pairing policy for one channel adapter.

    Args:
        store: Shared pairing store for the assistant.
        provider: Channel provider key, matching `ChannelStatus.provider`.
        env_sender_ids: Operator and allowlisted ids from env. Env stays
            authoritative: these senders are always admitted in DMs, never
            receive codes, and cannot be revoked through pairing.
        reply: Whether to send the code to the requester. When `False`, the
            operator reads pending codes from `/pair list` or the CLI instead.
        clock: Epoch-seconds clock, injectable for tests.
    """

    store: PairingStore
    provider: str
    env_sender_ids: frozenset[str] = frozenset()
    reply: bool = True
    clock: Callable[[], float] = time.time

    def now(self) -> int:
        """Return the current time in whole epoch seconds."""
        return int(self.clock())

    def admits(self, message: ChannelMessage) -> bool:
        """Return whether a message comes from an admitted sender.

        A paired sender is admitted in any chat, like an env operator. An
        env-listed sender keeps their env semantics: allowlisted users only in
        DMs, while operators are admitted by the exposure policy itself.

        Args:
            message: Inbound channel message.

        Returns:
            `True` for a paired sender anywhere, or an env-listed sender in a DM.
        """
        sender_id = message.sender_id
        if sender_id is None:
            return False
        if sender_id in self.env_sender_ids:
            return is_direct_message(message)
        return self.store.is_paired(self.provider, sender_id)

    def admits_reaction(self, reaction: ChannelReaction) -> bool:
        """Return whether a reaction comes from a paired sender, in any chat.

        Env-listed senders are left to the adapter's existing reaction policy. The
        host still only acts on a reaction to an approval prompt from the sender
        who started that run.

        Args:
            reaction: Inbound channel reaction.

        Returns:
            `True` when the reactor is paired.
        """
        sender_id = reaction.sender_id
        return sender_id is not None and self.store.is_paired(self.provider, sender_id)

    async def offer(
        self,
        message: ChannelMessage,
        send: Callable[[str, str], Awaitable[SendResult]],
    ) -> None:
        """Issue a code to a rejected DM sender, replying at most once per code.

        Args:
            message: Message the adapter's exposure policy rejected.
            send: Channel send callable taking a conversation id and text.
        """
        sender_id = message.sender_id
        if sender_id is None or sender_id in self.env_sender_ids:
            return
        if not is_direct_message(message):
            return
        try:
            code = await asyncio.to_thread(
                self.store.request,
                self.provider,
                sender_id,
                message.conversation_id,
                now=self.now(),
            )
        except (OSError, TimeoutError, PairingStoreError):
            logger.warning("Could not record a sender pairing request", exc_info=True)
            return
        if code is None:
            return
        log_debug_event(logger, "pairing.request.issued", provider=self.provider)
        if self.reply:
            await _send_quietly(send, message.conversation_id, request_reply(code))


@dataclass(frozen=True, slots=True)
class PairCommandResult:
    """Outcome of one `/pair` command, for the host to act on.

    Args:
        reply: Text to send back to the operator.
        approved: Sender approved by this command, to notify.
        revoked: Sender revoked by this command, whose work to stop.
    """

    reply: str
    approved: PairedSender | None = None
    revoked: PairedSender | None = None


PAIR_USAGE = "Usage: /pair list | /pair approve <code> | /pair revoke <sender-id>"
APPROVED_NOTICE = "Your access request was approved. Send a message to start."


def run_pair_command(pairing: SenderPairing, text: str) -> PairCommandResult:
    """Run an operator's `/pair` command against the channel's store.

    The caller must already have verified that the sender is an operator.

    Args:
        pairing: Pairing policy of the channel the command arrived on.
        text: Full command text, starting with `/pair`.

    Returns:
        Reply text plus any sender the command approved or revoked.
    """
    parts = text.split()
    action = parts[1].lower() if len(parts) > 1 else "list"
    argument = parts[2] if len(parts) == 3 else None  # noqa: PLR2004  # command, action, arg
    try:
        if action == "list" and len(parts) <= 2:  # noqa: PLR2004  # command and action only
            return PairCommandResult(format_listing(pairing))
        if action == "approve" and argument is not None:
            return approve_code(pairing, argument)
        if action == "revoke" and argument is not None:
            return revoke_sender(pairing, argument)
    except (OSError, TimeoutError, PairingStoreError):
        logger.warning("Sender pairing command failed", exc_info=True)
        return PairCommandResult("Could not update sender pairing. Check Talon logs.")
    return PairCommandResult(PAIR_USAGE)


def format_listing(pairing: SenderPairing) -> str:
    """Describe pending requests and approved senders for one channel.

    Args:
        pairing: Pairing policy of the channel to describe.

    Returns:
        Human-readable listing, including codes, for the operator only.
    """
    now = pairing.now()
    state = pairing.store.state(pairing.provider, now=now)
    lines = [f"Pending requests ({pairing.provider}):"]
    lines.extend(
        f"- {format_code(request.code)} from {request.sender_id}, "
        f"expires in {max(1, (request.expires_at - now) // 60)} min"
        for request in sorted(state.pending.values(), key=lambda item: item.created_at)
    )
    if not state.pending:
        lines.append("- none")
    lines.append(f"Paired senders ({pairing.provider}):")
    lines.extend(f"- {sender_id}" for sender_id in sorted(state.approved))
    if not state.approved:
        lines.append("- none")
    return "\n".join(lines)


def approve_code(pairing: SenderPairing, code: str) -> PairCommandResult:
    """Approve the sender a live code was issued to.

    Args:
        pairing: Pairing policy of the code's channel.
        code: Code as typed by the operator.

    Returns:
        Reply text, plus the approved sender on success.
    """
    paired = pairing.store.approve(pairing.provider, code, now=pairing.now())
    if paired is None:
        return PairCommandResult("No live pairing request matches that code.")
    return PairCommandResult(f"Paired sender {paired.sender_id}.", approved=paired)


def revoke_sender(pairing: SenderPairing, sender_id: str) -> PairCommandResult:
    """Revoke a paired sender; env-configured senders are left to env.

    Args:
        pairing: Pairing policy of the sender's channel.
        sender_id: Sender to revoke.

    Returns:
        Reply text, plus the revoked sender on success.
    """
    if sender_id in pairing.env_sender_ids:
        return PairCommandResult(
            f"Sender {sender_id} is configured in env; edit the env and restart to remove it."
        )
    revoked = pairing.store.revoke(pairing.provider, sender_id)
    if revoked is None:
        return PairCommandResult(f"Sender {sender_id} is not paired.")
    return PairCommandResult(f"Revoked sender {sender_id}.", revoked=revoked)


def sender_jobs(store: CronJobStore, provider: str, sender_id: str) -> list[CronJob]:
    """Return cron jobs a sender created on one channel, in any chat.

    Args:
        store: Cron job store.
        provider: Channel provider key the jobs were created on.
        sender_id: Sender whose jobs to find.

    Returns:
        Every matching job, enabled or not.
    """
    return [
        job
        for job in store.list_jobs()
        if job.origin.channel == provider and job.origin.sender_id == sender_id
    ]


def pause_jobs(store: CronJobStore, jobs: list[CronJob]) -> int:
    """Disable every enabled job in `jobs`.

    Args:
        store: Cron job store that holds the jobs.
        jobs: Jobs to pause.

    Returns:
        How many jobs this call paused.
    """
    enabled = [job for job in jobs if job.enabled]
    for job in enabled:
        store.edit_job(job.id, origin=job.origin, enabled=False)
    return len(enabled)


def request_reply(code: str) -> str:
    """Return the message sent to a requester with their code.

    Args:
        code: Newly issued code.

    Returns:
        Reply text that tells the requester what to do next.
    """
    return (
        "This assistant is private. To request access, send its operator this code: "
        f"{format_code(code)}. It expires in 1 hour."
    )


def pairing_from_env(
    env: Mapping[str, str],
    *,
    provider: str,
    env_prefix: str,
    open_exposure: bool,
    home: Path,
) -> SenderPairing | None:
    """Build a channel's pairing policy from env, or `None` when disabled.

    Args:
        env: Environment variable mapping.
        provider: Channel provider key.
        env_prefix: Provider env prefix, for example `DEEPAGENTS_TALON_DISCORD`.
        open_exposure: Whether the channel runs in `open` exposure mode.
        home: Per-assistant home directory that holds the store.

    Returns:
        Pairing policy, or `None` when `<prefix>_PAIRING` is unset or false.

    Raises:
        ValueError: If a flag is invalid or pairing is combined with `open` mode.
    """
    if not _flag(env, f"{env_prefix}_PAIRING", default=False):
        return None
    if open_exposure:
        msg = f"{env_prefix}_PAIRING cannot be combined with open exposure, which admits everyone"
        raise ValueError(msg)
    return SenderPairing(
        store=PairingStore(home / PAIRING_FILENAME),
        provider=provider,
        env_sender_ids=env_sender_ids(env, env_prefix),
        reply=_flag(env, f"{env_prefix}_PAIRING_REPLY", default=True),
    )


def env_sender_ids(env: Mapping[str, str], env_prefix: str) -> frozenset[str]:
    """Return operator and allowlisted user ids configured in env.

    Args:
        env: Environment variable mapping.
        env_prefix: Provider env prefix.

    Returns:
        Ids from `<prefix>_OPERATOR_ID` and `<prefix>_ALLOWLIST_USERS`.
    """
    ids: set[str] = set()
    for suffix in ("_OPERATOR_ID", "_ALLOWLIST_USERS"):
        ids.update(item.strip() for item in env.get(env_prefix + suffix, "").split(","))
    ids.discard("")
    return frozenset(ids)


def _flag(env: Mapping[str, str], name: str, *, default: bool) -> bool:
    value = env.get(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSY:
        return False
    msg = f"{name} must be one of: enabled, disabled, true, false"
    raise ValueError(msg)


async def _send_quietly(
    send: Callable[[str, str], Awaitable[SendResult]], conversation_id: str, text: str
) -> None:
    try:
        await send(conversation_id, text)
    except Exception:  # noqa: BLE001  # A failed courtesy reply must not break inbound handling.
        logger.warning("Could not send sender pairing reply", exc_info=True)


def _unique_code(pending: Mapping[str, PendingRequest]) -> str:
    taken = {request.code for request in pending.values()}
    while (code := generate_code()) in taken:
        pass
    return code


def _match_code(pending: Mapping[str, PendingRequest], candidate: bytes) -> PendingRequest | None:
    match = None
    for request in pending.values():
        # Compare every entry so timing does not reveal which code nearly matched.
        if hmac.compare_digest(request.code.encode(), candidate):
            match = request
    return match


def _unexpired(pending: Mapping[str, PendingRequest], now: int) -> dict[str, PendingRequest]:
    return {key: value for key, value in pending.items() if value.expires_at > now}


def _encode(channels: Mapping[str, _ChannelState]) -> dict[str, object]:
    return {
        "version": _VERSION,
        "channels": {
            provider: {
                "pending": {
                    key: {
                        "code": value.code,
                        "conversation_id": value.conversation_id,
                        "created_at": value.created_at,
                        "expires_at": value.expires_at,
                    }
                    for key, value in state.pending.items()
                },
                "approved": {
                    key: {
                        "conversation_id": value.conversation_id,
                        "approved_at": value.approved_at,
                    }
                    for key, value in state.approved.items()
                },
            }
            for provider, state in channels.items()
        },
    }


def _decode(raw: bytes) -> dict[str, _ChannelState]:
    try:
        document = json.loads(raw, object_pairs_hook=_unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        msg = "Pairing store is not valid JSON."
        raise PairingStoreError(msg) from error
    record = _record(document, {"version", "channels"})
    if record["version"] != _VERSION:
        msg = "Unsupported pairing store version."
        raise PairingStoreError(msg)
    channels = _mapping(record["channels"])
    return {_identifier(provider): _decode_channel(state) for provider, state in channels.items()}


def _decode_channel(value: object) -> _ChannelState:
    record = _record(value, {"pending", "approved"})
    pending = {
        _identifier(sender): _decode_pending(sender, entry)
        for sender, entry in _mapping(record["pending"]).items()
    }
    approved = {
        _identifier(sender): _decode_approved(sender, entry)
        for sender, entry in _mapping(record["approved"]).items()
    }
    if len(pending) > MAX_PENDING_PER_CHANNEL or len(approved) > MAX_APPROVED_PER_CHANNEL:
        msg = "Pairing store has too many entries."
        raise PairingStoreError(msg)
    return _ChannelState(pending=pending, approved=approved)


def _decode_pending(sender_id: str, value: object) -> PendingRequest:
    record = _record(value, {"code", "conversation_id", "created_at", "expires_at"})
    code = record["code"]
    if not isinstance(code, str) or len(code) != CODE_LENGTH or set(code) - set(CODE_ALPHABET):
        msg = "Invalid pairing code."
        raise PairingStoreError(msg)
    return PendingRequest(
        sender_id=sender_id,
        code=code,
        conversation_id=_identifier(record["conversation_id"]),
        created_at=_timestamp(record["created_at"]),
        expires_at=_timestamp(record["expires_at"]),
    )


def _decode_approved(sender_id: str, value: object) -> PairedSender:
    record = _record(value, {"conversation_id", "approved_at"})
    return PairedSender(
        sender_id=sender_id,
        conversation_id=_identifier(record["conversation_id"]),
        approved_at=_timestamp(record["approved_at"]),
    )


def _record(value: object, keys: set[str]) -> Mapping[str, object]:
    mapping = _mapping(value)
    if set(mapping) != keys:
        msg = "Pairing store entry has unexpected fields."
        raise PairingStoreError(msg)
    return mapping


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        msg = "Pairing store entry must be an object."
        raise PairingStoreError(msg)
    return value


def _identifier(value: object) -> str:
    if (
        not isinstance(value, str)
        or not 0 < len(value) <= _MAX_ID_CHARS
        or value != value.strip()
        or any(unicodedata.category(char).startswith("C") for char in value)
    ):
        msg = "Invalid identifier in pairing store."
        raise PairingStoreError(msg)
    return value


def _timestamp(value: object) -> int:
    if type(value) is not int or value < 0:
        msg = "Invalid timestamp in pairing store."
        raise PairingStoreError(msg)
    return value


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            msg = "Duplicate key in pairing store."
            raise PairingStoreError(msg)
        result[key] = value
    return result
