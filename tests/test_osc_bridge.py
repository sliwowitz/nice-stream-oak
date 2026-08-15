# SPDX-FileCopyrightText: 2026 Jiří Vyskočil <jiri@vyskocil.com>
# SPDX-License-Identifier: MIT
"""OSC bridge tests: the intrinsics gate and the address filter.

The intrinsics half guards a crash seen in the gallery -- the bridge
attached during the ~30 s a camera takes to connect, cached the fx=0 the
producer had written, and died on the first person to walk in."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nsk
import osc_bridge

REAL = (800.0, 800.0, 640.0, 400.0)     # plausible CAM_A intrinsics
BLANK = (0.0, 0.0, 0.0, 0.0)            # what a producer writes before connect


def pose_segment(intrinsics: nsk.Intrinsics) -> bytearray:
    buf = bytearray(4096)
    nsk.write_pose_header(buf, intrinsics)
    return buf


# ----------------------------------------------------------- intrinsics ----
def test_blank_intrinsics_are_not_usable():
    assert osc_bridge.usable(REAL)
    assert not osc_bridge.usable(BLANK)
    # fy alone at zero still divides by zero on the vertical axis
    assert not osc_bridge.usable((800.0, 0.0, 640.0, 400.0))


def test_intrinsics_are_picked_up_once_the_camera_connects():
    assert osc_bridge.current_intrinsics(pose_segment(REAL), BLANK) == REAL


def test_a_producer_restart_does_not_blank_a_running_bridge():
    """A restarting producer rewrites the shared header with zeros."""
    assert osc_bridge.current_intrinsics(pose_segment(BLANK), REAL) == REAL


def test_a_new_camera_replaces_the_held_intrinsics():
    other = (610.0, 610.0, 320.0, 240.0)
    assert osc_bridge.current_intrinsics(pose_segment(other), REAL) == other


def test_an_unwritten_header_holds_rather_than_reads_garbage():
    assert osc_bridge.current_intrinsics(bytearray(4096), REAL) == REAL

