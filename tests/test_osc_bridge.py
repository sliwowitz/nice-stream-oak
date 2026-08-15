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


# -------------------------------------------------------- address filter ----
def test_no_filter_keeps_every_address():
    keep = osc_bridge.address_filter(None)
    assert keep("/nice/group/energy") and keep("/nice/slot/7/conf")


def test_globs_span_slashes_and_select_whole_families():
    keep = osc_bridge.address_filter("/nice/group/*,/nice/pair/*")
    assert keep("/nice/group/energy")
    assert keep("/nice/pair/0-1/distance")
    assert not keep("/nice/slot/0/position")
    assert not keep("/nice/event/entered")


def test_a_single_address_narrows_to_exactly_it():
    keep = osc_bridge.address_filter(" /nice/slot/0/position ")
    assert keep("/nice/slot/0/position")
    assert not keep("/nice/slot/1/position")
