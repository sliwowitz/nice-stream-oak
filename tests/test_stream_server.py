# SPDX-FileCopyrightText: 2026 Jiří Vyskočil <jiri@vyskocil.com>
# SPDX-License-Identifier: MIT
"""stream_server unit tests: the pieces that run without a camera --
interactive config and the keypoint letterbox mapper."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stream_server as ss


@pytest.fixture(autouse=True)
def restore_config():
    saved = (ss.POSE_SOURCE, ss.IR_DOT, ss.IR_FLOOD, ss.POSE_MODEL)
    yield
    ss.POSE_SOURCE, ss.IR_DOT, ss.IR_FLOOD, ss.POSE_MODEL = saved


def feed(monkeypatch, *lines):
    it = iter(lines)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(it))


def test_profile_nonir_and_model_pick(monkeypatch):
    feed(monkeypatch, "1", "2")                # non-IR profile, nano model
    ss.configure_interactively()
    assert (ss.POSE_SOURCE, ss.IR_DOT, ss.IR_FLOOD) == ("rgb", 0.0, 0.0)
    assert ss.POSE_MODEL == ss.KNOWN_POSE_MODELS[1][0]


def test_profile_host_skips_model_menu(monkeypatch):
    feed(monkeypatch, "3")                     # host backend: no model prompt
    ss.configure_interactively()               # a second input() would raise
    assert ss.POSE_SOURCE == "host"
    assert not ss.device_pose()


def test_device_pose_only_for_on_device_sources():
    """'host' and 'none' both mean the camera is not doing pose. Every gate
    in the server keys off this rather than testing against "none", which is
    what let 'host' be added without touching the pipeline."""
    for source, expected in (("left", True), ("rgb", True),
                             ("host", False), ("none", False)):
        ss.POSE_SOURCE = source
        assert ss.device_pose() is expected, source
    assert set(ss.POSE_SOURCES) == {"left", "rgb", "host", "none"}


def test_custom_profile_accepts_host(monkeypatch):
    feed(monkeypatch, "5", "host", "0.1", "0.2")
    ss.configure_interactively()               # no model prompt for a host run
    assert ss.POSE_SOURCE == "host"


def test_custom_profile_validates_source(monkeypatch):
    feed(monkeypatch, "5", "bogus", "left", "0.6", "0.4", "keep")
    ss.configure_interactively()
    assert ss.POSE_SOURCE == "left"
    assert (ss.IR_DOT, ss.IR_FLOOD) == (0.6, 0.4)


def test_eof_keeps_env_config(monkeypatch):
    def eof(_prompt=""):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    before = (ss.POSE_SOURCE, ss.IR_DOT, ss.IR_FLOOD, ss.POSE_MODEL)
    ss.configure_interactively()
    assert (ss.POSE_SOURCE, ss.IR_DOT, ss.IR_FLOOD, ss.POSE_MODEL) == before


def fake_kp(x, y):
    return SimpleNamespace(imageCoordinates=SimpleNamespace(x=x, y=y),
                           confidence=0.9)


def test_kp_mapper_normalised_roundtrip():
    """A normalised keypoint at the letterbox content edges must land on the
    full-frame edges (the centre-crop bug this mapper replaced put content
    edges ~48 px off)."""
    to_frame = ss.make_kp_mapper(640, 352)     # W=1280, H=800 module config
    scale = 0.44
    pad_x = (640 - 1280 * scale) / 2 / 640     # normalised pad
    x, y = to_frame(fake_kp(pad_x, 0.0))
    assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-6)
    x, y = to_frame(fake_kp(1.0 - pad_x, 1.0))
    assert (x, y) == pytest.approx((1280.0, 800.0), abs=1e-4)
    x, y = to_frame(fake_kp(0.5, 0.5))
    assert (x, y) == pytest.approx((640.0, 400.0), abs=1e-6)


def test_kp_mapper_pixel_convention():
    """Pixel-coordinate keypoints (NN frame) take the same letterbox undo."""
    to_frame = ss.make_kp_mapper(640, 352)
    x, y = to_frame(fake_kp(320.0, 176.0))     # NN-frame centre, in pixels
    assert (x, y) == pytest.approx((640.0, 400.0), abs=1e-6)
