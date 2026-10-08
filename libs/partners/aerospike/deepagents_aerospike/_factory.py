"""Factory for constructing an Aerospike-backed LangGraph checkpoint saver.

The helper in this module handles input validation and a single synchronous
Aerospike client session so callers can pass the result directly to
``create_deep_agent(checkpointer=...)`` without managing connection lifecycle.
"""

from __future__ import annotations

import contextlib
import logging

from aerospike_async.exceptions import AerospikeError as AsyncAerospikeError
from aerospike_sdk import Behavior
from aerospike_sdk.exceptions import AerospikeError
from aerospike_sdk.sync import ClusterDefinition
from langgraph.checkpoint.aerospike import AerospikeSaver

# The sync ClusterDefinition.connect() currently emits exceptions from the
# ``aerospike_async`` exception hierarchy (e.g. ``ConnectionError``), so we catch
# both SDK namespaces and preserve the original concrete type.

logger = logging.getLogger(__name__)

_MIN_PORT = 1
_MAX_PORT = 65535


def create_aerospike_checkpointer(
    host: str,
    port: int,
    namespace: str = "test",
    *,
    ttl_minutes: int | None = None,
) -> AerospikeSaver:
    """Create an ``AerospikeSaver`` wired to a single Aerospike session.

    Validates all inputs before opening any network connection, then builds one
    ``ClusterDefinition`` → ``SyncSession`` and returns an ``AerospikeSaver``
    that owns the session for its lifetime.

    Args:
        host: Aerospike server hostname or IP address.
        port: Aerospike server port. Must be between 1 and 65535 inclusive.
        namespace: Aerospike namespace for checkpoint records. Defaults to ``"test"``.
        ttl_minutes: Optional positive integer TTL in minutes. When provided, the
            saver applies a default record TTL of ``ttl_minutes * 60`` seconds.

    Returns:
        An ``AerospikeSaver`` instance ready to use as a LangGraph checkpointer.

    Raises:
        ValueError: If ``host``, ``port``, ``namespace``, or ``ttl_minutes`` fail validation.
        Exception: If the Aerospike cluster or session cannot be created. The original
            SDK exception type is preserved and annotated with context.

    Example:
        .. code-block:: python

            from deepagents import create_deep_agent
            from deepagents_aerospike import create_aerospike_checkpointer

            checkpointer = create_aerospike_checkpointer(
                host="localhost",
                port=3000,
                namespace="test",
                ttl_minutes=60,
            )

            agent = create_deep_agent(
                model=model,
                tools=tools,
                checkpointer=checkpointer,
            )
    """
    if not isinstance(host, str) or not host.strip():
        msg = f"host must be a non-empty string, got {host!r}"
        raise ValueError(msg)

    if not isinstance(port, int) or not (_MIN_PORT <= port <= _MAX_PORT):
        msg = f"port must be between {_MIN_PORT} and {_MAX_PORT}, got {port!r}"
        raise ValueError(msg)

    if not isinstance(namespace, str) or not namespace.strip():
        msg = f"namespace must be a non-empty string, got {namespace!r}"
        raise ValueError(msg)

    if ttl_minutes is not None and (not isinstance(ttl_minutes, int) or ttl_minutes <= 0):
        msg = f"ttl_minutes must be positive or None, got {ttl_minutes!r}"
        raise ValueError(msg)

    logger.debug("Connecting to Aerospike at %s:%s namespace=%s", host, port, namespace)
    try:
        cluster = ClusterDefinition(host, port).connect()
    except (AerospikeError, AsyncAerospikeError) as e:
        e.add_note(f"Failed to connect to Aerospike at {host}:{port}")
        raise

    try:
        session = cluster.create_session(Behavior.DEFAULT)
    except (AerospikeError, AsyncAerospikeError) as e:
        with contextlib.suppress(Exception):
            cluster.close()
        e.add_note(f"Failed to create Aerospike session at {host}:{port}")
        raise

    ttl: dict[str, int] | None = {"default_ttl": ttl_minutes} if ttl_minutes is not None else None

    return AerospikeSaver(session=session, namespace=namespace, ttl=ttl, cluster=cluster)
