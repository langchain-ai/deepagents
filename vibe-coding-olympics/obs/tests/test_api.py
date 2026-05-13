from __future__ import annotations

from fastapi.testclient import TestClient

from obs_runner.api import create_app
from obs_runner.config import Config


class FakeCompositor:
    def __init__(self) -> None:
        self.scenes: list[str] = []
        self.texts: list[tuple[str, str]] = []

    def set_scene(self, name: str) -> None:
        self.scenes.append(name)

    def set_text(self, source: str, value: str) -> None:
        self.texts.append((source, value))


class DisconnectedCompositor:
    def set_scene(self, name: str) -> None:
        raise ConnectionError("OBS offline")

    def set_text(self, source: str, value: str) -> None:
        raise ConnectionError("OBS offline")


def _config() -> Config:
    return Config()


def test_scene_switch_drives_obs() -> None:
    comp = FakeCompositor()
    client = TestClient(create_app(config=_config(), compositor=comp))

    response = client.post("/scene", json={"name": "p1 focus"})

    assert response.status_code == 200
    assert response.json() == {"name": "p1 focus"}
    assert comp.scenes == ["p1 focus"]


def test_scene_switch_maps_compositor_connection_error_to_503() -> None:
    client = TestClient(
        create_app(config=_config(), compositor=DisconnectedCompositor())
    )

    response = client.post("/scene", json={"name": "p1 focus"})

    assert response.status_code == 503
    assert "OBS offline" in response.text


def test_text_update_drives_obs() -> None:
    comp = FakeCompositor()
    client = TestClient(create_app(config=_config(), compositor=comp))

    response = client.post(
        "/text",
        json={"source": "PromptText", "value": "build a taco truck"},
    )

    assert response.status_code == 200
    assert response.json() == {"source": "PromptText", "value": "build a taco truck"}
    assert comp.texts == [("PromptText", "build a taco truck")]


def test_text_update_maps_compositor_connection_error_to_503() -> None:
    client = TestClient(
        create_app(config=_config(), compositor=DisconnectedCompositor())
    )

    response = client.post("/text", json={"source": "PromptText", "value": "x"})

    assert response.status_code == 503
    assert "OBS offline" in response.text


def test_healthz_reports_compositor_kind() -> None:
    client = TestClient(create_app(config=_config(), compositor=FakeCompositor()))

    response = client.get("/healthz")

    assert response.status_code == 200
    # A pure fake has no obs-websocket client, so `obs_connected` is false.
    assert response.json() == {"obs_connected": False}
