"""Shared session-level fixtures for Aerospike integration tests.

The ``session`` fixture attempts to connect to a running Aerospike server once
per test session. If the server is unreachable, the entire integration suite is
skipped gracefully via ``pytest.skip``.
"""

import contextlib
import os

import pytest
from aerospike_async.exceptions import AerospikeError
from aerospike_sdk import Behavior
from aerospike_sdk.exceptions import AerospikeError as SyncAerospikeError
from aerospike_sdk.sync import ClusterDefinition

# ``ClusterDefinition.connect()`` currently raises ``ConnectionError`` from the
# ``aerospike_async`` exception hierarchy, even though the API is sync. Catch both
# namespaces so the fixture skips cleanly regardless of which layer fails.


@pytest.fixture(scope="session")
def aerospike_namespace():
    """Return the Aerospike namespace to use for integration tests."""
    return os.getenv("AEROSPIKE_NAMESPACE", "test")


@pytest.fixture(scope="session")
def session():
    """Yield a single shared Aerospike SDK session for the test session.

    Skips the integration suite when the configured Aerospike server is not
    reachable. The cluster is closed cleanly after the session finishes.
    """
    host = os.getenv("AEROSPIKE_HOST", "localhost")
    port = int(os.getenv("AEROSPIKE_PORT", "3000"))
    try:
        cluster = ClusterDefinition(host, port).connect()
    except AerospikeError as e:
        pytest.skip(f"Could not connect to Aerospike at {host}:{port}: {e}")
    try:
        sess = cluster.create_session(Behavior.DEFAULT)
    except (AerospikeError, SyncAerospikeError) as e:
        with contextlib.suppress(Exception):
            cluster.close()
        pytest.skip(f"Could not create Aerospike session at {host}:{port}: {e}")
    yield sess
    with contextlib.suppress(Exception):
        cluster.close()
