from __future__ import annotations

from fastapi.testclient import TestClient

from obs_runner.api import create_app
from obs_runner.config import Config
from obs_runner.state_machine import Phase


class FakeCompositor:
    def __init__(self) -> None:
        self.scenes: list[str] = []

    def set_scene(self, name: str) -> None:
        self.scenes.append(name)

    def set_text(self, source: str, value: str) -> None:
        pass


class DisconnectedCompositor:
    def set_scene(self, name: str) -> None:
        raise ConnectionError("OBS offline")

    def set_text(self, source: str, value: str) -> None:
        raise ConnectionError("OBS offline")


def _config() -> Config:
    return Config(
        scenes={
            Phase.IDLE: "Idle",
            Phase.CODING: "Coding",
            Phase.SCOREBOARD: "Scoreboard",
        },
    )


def test_transition_maps_invalid_transition_to_409() -> None:
    client = TestClient(create_app(config=_config(), compositor=FakeCompositor()))

    response = client.post("/transition", json={"event": "end", "payload": {}})

    assert response.status_code == 409
    assert "not valid" in response.text


def test_transition_maps_compositor_connection_error_to_503() -> None:
    client = TestClient(
        create_app(config=_config(), compositor=DisconnectedCompositor())
    )

    response = client.post(
        "/transition",
        json={
            "event": "start",
            "payload": {"prompt": "x", "contestants": ["Alice"]},
        },
    )

    assert response.status_code == 503
    assert "OBS offline" in response.text


def test_transition_maps_payload_validation_error_to_409() -> None:
    client = TestClient(create_app(config=_config(), compositor=FakeCompositor()))

    response = client.post(
        "/transition",
        json={"event": "ready", "payload": {"contestants": []}},
    )

    assert response.status_code == 409
    assert "contestants" in response.text


def test_scene_switch_drives_obs_without_changing_state() -> None:
    comp = FakeCompositor()
    client = TestClient(create_app(config=_config(), compositor=comp))

    response = client.post("/scene", json={"name": "p1 focus"})

    assert response.status_code == 200
    assert response.json()["phase"] == "idle"
    assert comp.scenes == ["p1 focus"]


def test_scene_switch_maps_compositor_connection_error_to_503() -> None:
    client = TestClient(
        create_app(config=_config(), compositor=DisconnectedCompositor())
    )

    response = client.post("/scene", json={"name": "p1 focus"})

    assert response.status_code == 503
    assert "OBS offline" in response.text
