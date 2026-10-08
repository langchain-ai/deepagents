"""Unit tests for create_aerospike_checkpointer factory — no network required."""

from __future__ import annotations

import pytest
from aerospike_sdk.exceptions import AerospikeError

from deepagents_aerospike import create_aerospike_checkpointer


def test_invalid_host_empty():
    """Empty string host is rejected before any network call."""
    with pytest.raises(ValueError, match="host must be a non-empty string"):
        create_aerospike_checkpointer(host="", port=3000)


def test_invalid_host_whitespace():
    """Whitespace-only host is rejected before any network call."""
    with pytest.raises(ValueError, match="host must be a non-empty string"):
        create_aerospike_checkpointer(host="  ", port=3000)


def test_invalid_host_non_string():
    """Non-string host is rejected before any network call."""
    with pytest.raises((ValueError, TypeError), match="host must be a non-empty string"):
        create_aerospike_checkpointer(host=None, port=3000)


def test_invalid_port_zero():
    """Port zero is rejected before any network call."""
    with pytest.raises(ValueError, match="port must be between 1 and 65535"):
        create_aerospike_checkpointer(host="localhost", port=0)


def test_invalid_port_too_large():
    """Port above the maximum is rejected before any network call."""
    with pytest.raises(ValueError, match="port must be between 1 and 65535"):
        create_aerospike_checkpointer(host="localhost", port=65536)


def test_invalid_port_negative():
    """Negative port is rejected before any network call."""
    with pytest.raises(ValueError, match="port must be between 1 and 65535"):
        create_aerospike_checkpointer(host="localhost", port=-1)


def test_boundary_port_min(monkeypatch):
    """Port 1 passes validation; a connection failure is propagated as-is."""
    def _connect(*args, **kwargs):
        raise AerospikeError("connection refused")

    monkeypatch.setattr(
        "deepagents_aerospike._factory.ClusterDefinition.connect", _connect
    )
    with pytest.raises(AerospikeError, match="connection refused"):
        create_aerospike_checkpointer(host="localhost", port=1)


def test_boundary_port_max(monkeypatch):
    """Port 65535 passes validation; a connection failure is propagated as-is."""
    def _connect(*args, **kwargs):
        raise AerospikeError("connection refused")

    monkeypatch.setattr(
        "deepagents_aerospike._factory.ClusterDefinition.connect", _connect
    )
    with pytest.raises(AerospikeError, match="connection refused"):
        create_aerospike_checkpointer(host="localhost", port=65535)


def test_invalid_namespace_empty():
    """Empty string namespace is rejected before any network call."""
    with pytest.raises(ValueError, match="namespace must be a non-empty string"):
        create_aerospike_checkpointer(host="localhost", port=3000, namespace="")


def test_invalid_namespace_whitespace():
    """Whitespace-only namespace is rejected before any network call."""
    with pytest.raises(ValueError, match="namespace must be a non-empty string"):
        create_aerospike_checkpointer(host="localhost", port=3000, namespace="   ")


def test_invalid_ttl_negative():
    """Negative ttl_minutes is rejected before any network call."""
    with pytest.raises(ValueError, match="ttl_minutes must be positive"):
        create_aerospike_checkpointer(host="localhost", port=3000, ttl_minutes=-1)


def test_invalid_ttl_zero():
    """Zero ttl_minutes is rejected before any network call."""
    with pytest.raises(ValueError, match="ttl_minutes must be positive"):
        create_aerospike_checkpointer(host="localhost", port=3000, ttl_minutes=0)


def test_valid_config_no_server(monkeypatch):
    """Valid configuration passes validation and propagates connection failures."""
    def _connect(*args, **kwargs):
        raise AerospikeError("connection refused")

    monkeypatch.setattr(
        "deepagents_aerospike._factory.ClusterDefinition.connect", _connect
    )
    with pytest.raises(AerospikeError, match="connection refused"):
        create_aerospike_checkpointer(
            host="localhost",
            port=3000,
            namespace="test",
            ttl_minutes=60,
        )


def test_valid_config_no_ttl(monkeypatch):
    """Default TTL passes validation and propagates connection failures."""
    def _connect(*args, **kwargs):
        raise AerospikeError("connection refused")

    monkeypatch.setattr(
        "deepagents_aerospike._factory.ClusterDefinition.connect", _connect
    )
    with pytest.raises(AerospikeError, match="connection refused"):
        create_aerospike_checkpointer(host="localhost", port=3000)
