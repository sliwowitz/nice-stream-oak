# SPDX-FileCopyrightText: 2026 Jiří Vyskočil <jiri@vyskocil.com>
# SPDX-License-Identifier: MIT
"""Pose-input tests: where a joint ends up and where its depth came from.

Two families, both silent when they break in production -- a keypoint
mapped back through the wrong crop lands on the wrong pixel and still looks
like a person, and a patch straddling a silhouette reports the wall behind
it with full confidence."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nsk

FRAME_W, FRAME_H = 1280, 800        # the source grid every producer writes
MODEL = 640                         # RTMO input, square


# ------------------------------------------------------- roi resolution ----
def test_roi_unset_means_the_whole_frame():
    for spec in ("", "   "):
        roi, complaint = nsk.resolve_roi(spec, FRAME_W, FRAME_H)
        assert roi == nsk.Roi(0, 0, FRAME_W, FRAME_H)
        assert complaint is None


def test_roi_clamps_into_the_frame():
    roi, complaint = nsk.resolve_roi("1200, 700, 400, 400", FRAME_W, FRAME_H)
    assert roi == nsk.Roi(1200, 700, 80, 100)       # trimmed at both edges
    assert complaint is None
    roi, _ = nsk.resolve_roi("-40,-40,320,320", FRAME_W, FRAME_H)
    assert roi == nsk.Roi(0, 0, 320, 320)


@pytest.mark.parametrize("spec", ["0,0,0,0", "640,400,-10,200", "junk",
                                  "10,20,30", "10,20,30,40,50", "1279,799,8,8"])
def test_roi_refuses_a_degenerate_rectangle(spec):
    roi, complaint = nsk.resolve_roi(spec, FRAME_W, FRAME_H)
    assert roi == nsk.Roi(0, 0, FRAME_W, FRAME_H)   # fall back, never crash
    assert complaint and spec in complaint          # and say so


# ------------------------------------------------------- roi round trips ----
def test_full_frame_roi_matches_letterbox_transform():
    """The default path must keep the geometry existing callers rely on."""
    whole, _ = nsk.resolve_roi("", FRAME_W, FRAME_H)
    lb = nsk.roi_letterbox(whole, MODEL, MODEL)
    assert (lb.scale, lb.pad_x, lb.pad_y) == \
        nsk.letterbox_transform(FRAME_W, FRAME_H, MODEL, MODEL)
    assert (lb.scale, lb.pad_x, lb.pad_y) == pytest.approx((0.5, 0.0, 120.0))
    assert lb.content_size() == (640, 400)          # 240 rows of grey


def test_roi_round_trip_maps_through_crop_and_letterbox():
    """A crop that needs both an origin and a pad: 800x400 at (100, 50)
    scales by 0.8 and pads 160 rows top and bottom."""
    lb = nsk.roi_letterbox(nsk.Roi(100, 50, 800, 400), MODEL, MODEL)
    assert (lb.scale, lb.pad_x, lb.pad_y) == pytest.approx((0.8, 0.0, 160.0))

    # Crop corners sit on the picture's corners inside the model input.
    assert lb.to_model(100, 50) == pytest.approx((0.0, 160.0))
    assert lb.to_model(900, 450) == pytest.approx((640.0, 480.0))
    assert lb.to_source(0.0, 160.0) == pytest.approx((100.0, 50.0))
    assert lb.to_source(640.0, 480.0) == pytest.approx((900.0, 450.0))
    # Centre to centre, and every point home again.
    assert lb.to_source(320.0, 320.0) == pytest.approx((500.0, 250.0))
    for x, y in ((0.0, 0.0), (17.5, 613.25), (639.0, 1.0), (320.0, 320.0)):
        assert lb.to_model(*lb.to_source(x, y)) == pytest.approx((x, y))


def test_roi_matching_the_model_aspect_needs_no_padding():
    """The rectangle the docstring recommends: square, 1:1, all picture."""
    lb = nsk.roi_letterbox(nsk.Roi(320, 80, 640, 640), MODEL, MODEL)
    assert (lb.scale, lb.pad_x, lb.pad_y) == pytest.approx((1.0, 0.0, 0.0))
    assert lb.content_size() == (MODEL, MODEL)
    assert lb.to_source(0.0, 0.0) == pytest.approx((320.0, 80.0))
    assert lb.to_source(640.0, 640.0) == pytest.approx((960.0, 720.0))


# ---------------------------------------------------------- patch splits ----
def wall_with_limb():
    """A 4 m wall with a 2 m vertical limb over it: x 40..59, y 20..79."""
    depth = np.full((100, 100), 4000, dtype=np.uint16)
    depth[20:80, 40:60] = 2000
    return depth


def test_depth_at_rejects_a_patch_split_across_a_silhouette():
    depth = wall_with_limb()
    assert nsk.depth_at(depth, 50, 50) == pytest.approx(2.0)   # deep inside
    assert nsk.depth_at(depth, 50, 79) == 0.0    # limb tip: half wall, no lie
    assert nsk.depth_at(depth, 61, 50) == 0.0    # just off the limb's side


def test_depth_at_keeps_a_slanted_surface():
    """One surface seen at an angle spans depth continuously -- not a split."""
    depth = np.zeros((100, 100), dtype=np.uint16)
    depth[:, :] = (2000 + np.arange(100) * 20).astype(np.uint16)  # 20 mm/px
    assert nsk.depth_at(depth, 50, 50) == pytest.approx(3.0)


def test_depth_at_still_survives_a_lone_flying_pixel():
    """A single stray pixel is the median's job, not the split test's."""
    depth = np.full((100, 100), 2000, dtype=np.uint16)
    depth[50, 50] = 30000
    assert nsk.depth_at(depth, 50, 50) == pytest.approx(2.0)


# ------------------------------------------------------ inboard sampling ----
def test_depth_inboard_reads_the_limb_not_its_tip():
    depth = wall_with_limb()
    assert nsk.depth_at(depth, 50, 79) == 0.0                  # tip alone
    assert nsk.depth_inboard(depth, 50, 79, 50, 40) == pytest.approx(2.0)


def test_depth_inboard_falls_back_to_the_keypoint():
    """A step that lands nowhere useful must not cost the joint its depth."""
    depth = np.full((100, 100), 2500, dtype=np.uint16)
    depth[:20, :] = 0                       # invalid band the step lands in
    assert nsk.depth_inboard(depth, 50, 60, 50, -200) == pytest.approx(2.5)


# ----------------------------------------------------------- joint lifts ----
def points_for(**by_index):
    """17 COCO keypoints, all unconfident except the named indices."""
    points = [(0.0, 0.0, 0.0)] * nsk.NUM_KP
    for i, value in by_index.items():
        points[int(i)] = value
    return points


def test_lift_joints_samples_towards_the_parent():
    depth = wall_with_limb()
    # l_wrist (9) at the limb tip, its parent l_elbow (7) up the limb.
    points = points_for(**{"9": (50.0, 79.0, 0.9), "7": (50.0, 40.0, 0.9)})
    joints = nsk.lift_joints(depth, points)
    assert joints[9][:2] == (50.0, 79.0)        # position untouched
    assert joints[9][2] == pytest.approx(2.0)   # depth from the forearm
    assert joints[9][3] == pytest.approx(0.9)


def test_lift_joints_without_a_parent_reads_the_keypoint():
    depth = wall_with_limb()
    points = points_for(**{"9": (50.0, 79.0, 0.9)})     # elbow unconfident
    assert nsk.lift_joints(depth, points)[9][2] == 0.0  # split patch, no lie
    points = points_for(**{"9": (50.0, 50.0, 0.9)})
    assert nsk.lift_joints(depth, points)[9][2] == pytest.approx(2.0)


def test_lift_joints_leaves_unconfident_keypoints_flat():
    depth = np.full((100, 100), 2000, dtype=np.uint16)
    points = points_for(**{"9": (50.0, 50.0, nsk.KP_CONF_MIN)})
    joints = nsk.lift_joints(depth, points)
    assert all(j[2] == 0.0 for j in joints)


# -------------------------------------------------------------- dev gate ----
def test_joint_dev_max_is_body_plausible():
    """0.8 m spans a person front to back; 1.5 m used to admit the wall."""
    assert nsk.JOINT_DEV_MAX == 0.8
    body = [(0.0, 0.0, 3.0, 0.9)] * 5
    reached_arm = (0.0, 0.0, 2.4, 0.9)          # hand towards the camera
    background = (0.0, 0.0, 4.2, 0.9)           # the wall behind them
    gated = nsk.gate_depth_edges([*body, reached_arm, background])
    assert gated[-2][2] == pytest.approx(2.4)   # kept
    assert gated[-1][2] == 0.0                  # dropped
