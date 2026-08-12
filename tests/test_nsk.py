# SPDX-FileCopyrightText: 2026 Jiří Vyskočil <jiri@vyskocil.com>
# SPDX-License-Identifier: MIT
"""Wire-contract tests: everything Unity's readers rely on, byte for byte.

One write -> read round-trip covers both producers (they share
nsk.write_pose_slots) and both consumer paths (parse_pose_header +
read_pose_frame mirror PoseShmStream.cs)."""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nsk


def make_person(seed=1.0, z=2.0):
    joints = [(100.0 * seed + k, 50.0 * seed + k, z, 0.9)
              for k in range(nsk.NUM_KP)]
    return 0.8, joints


def test_pose_roundtrip_and_layout():
    buf = bytearray(nsk.POSE_SZ)
    intr = (717.3, 717.0, 640.0, 400.0)
    nsk.write_pose_header(buf, intr)
    nsk.write_pose_slots(buf, [make_person(1.0), make_person(2.0, z=3.0)],
                         frame_id=42, timestamp=123.5)

    hdr = nsk.parse_pose_header(buf)
    assert hdr is not None
    assert hdr["max_persons"] == nsk.MAX_PERSONS
    assert hdr["num_kp"] == nsk.NUM_KP
    assert hdr["stride"] == nsk.PERSON_STRIDE == 288
    assert hdr["intrinsics"] == pytest.approx(intr)

    fid, persons = nsk.read_pose_frame(buf, hdr["max_persons"],
                                       hdr["num_kp"], hdr["stride"])
    assert fid == 42
    assert len(persons) == 2                    # empty slots skipped
    conf, joints = persons[0]
    assert conf == pytest.approx(0.8)
    assert joints[3] == pytest.approx((103.0, 53.0, 2.0, 0.9))

    # Fixed offsets the Unity reader (PoseShmStream.cs) hardcodes.
    assert struct.unpack_from("<I", buf, 0)[0] == 0x504B534E     # 'NSKP'
    assert struct.unpack_from("<I", buf, 40)[0] == 2             # num_persons
    assert struct.unpack_from("<Q", buf, 24)[0] == 42            # commit id
    assert struct.unpack_from("<d", buf, 32)[0] == 123.5         # timestamp
    assert struct.unpack_from("<f", buf, 48)[0] == pytest.approx(717.3)


def test_pose_write_marks_empty_slots():
    buf = bytearray(nsk.POSE_SZ)
    nsk.write_pose_slots(buf, [make_person()], frame_id=1)
    nsk.write_pose_slots(buf, [], frame_id=2)   # everyone left
    _, persons = nsk.read_pose_frame(buf, nsk.MAX_PERSONS,
                                     nsk.NUM_KP, nsk.PERSON_STRIDE)
    assert persons == []


def test_frame_header_roundtrip():
    buf = bytearray(nsk.HDR)
    nsk.write_frame_header(buf, 1280, 800, 2, 3, (1.0, 2.0, 3.0, 4.0))
    hdr = nsk.parse_frame_header(buf)
    assert (hdr["w"], hdr["h"], hdr["bpp"], hdr["nbuf"]) == (1280, 800, 2, 3)
    assert hdr["intrinsics"] == pytest.approx((1.0, 2.0, 3.0, 4.0))
    assert nsk.parse_frame_header(bytearray(nsk.HDR)) is None    # unwritten


def test_depth_at_rules():
    depth = np.zeros((100, 100), dtype=np.uint16)
    depth[40:60, 40:60] = 2000                 # 2 m block
    assert nsk.depth_at(depth, 50, 50) == pytest.approx(2.0)
    assert nsk.depth_at(depth, 5, 5) == 0.0    # empty patch
    assert nsk.depth_at(depth, -1, 50) == 0.0  # out of bounds
    assert nsk.depth_at(depth, float("nan"), 50) == 0.0
    # median rejects a lone outlier inside the patch
    depth[50, 50] = 30000
    assert nsk.depth_at(depth, 50, 50) == pytest.approx(2.0)


def test_depth_edge_gate():
    good = [(0, 0, 2.0, 0.9)] * 5
    outlier = (0, 0, 6.0, 0.9)                 # sampled the wall behind
    gated = nsk.gate_depth_edges([*good, outlier])
    assert gated[-1][2] == 0.0                 # z dropped, joint kept
    assert all(j[2] == 2.0 for j in gated[:-1])
    # fewer than 3 depth-lifted joints: not enough evidence, no gating
    sparse = [(0, 0, 6.0, 0.9), (0, 0, 2.0, 0.9)]
    assert nsk.gate_depth_edges(sparse) == sparse


def test_letterbox_transform():
    # The production case: 1280x800 into 640x352 pads horizontally.
    scale, pad_x, pad_y = nsk.letterbox_transform(1280, 800, 640, 352)
    assert scale == pytest.approx(0.44)
    assert pad_x == pytest.approx(38.4)
    assert pad_y == 0.0
    # Inverse maps the letterboxed corners back to the full frame.
    assert (0 * 1280 * scale + pad_x) == pytest.approx(pad_x)
    assert ((640 - pad_x) - pad_x) / scale == pytest.approx(1280, abs=1e-6)
    # Square model input (RTMO): pads vertically instead.
    scale, pad_x, pad_y = nsk.letterbox_transform(1280, 800, 640, 640)
    assert (scale, pad_x) == pytest.approx((0.5, 0.0))
    assert pad_y == pytest.approx(120.0)


class RingSegment:
    """NSK1-shaped buffer for driving nsk.newest_frame in-process. Races are
    simulated by patching np.frombuffer (the copy step) to mutate the commit
    id -- exactly the window a real producer would race in."""

    def __init__(self, w=4, h=2, nbuf=3, itemsize=1):
        self.frame_sz = w * h * itemsize
        self.data = bytearray(nsk.HDR + nbuf * self.frame_sz)
        nsk.write_frame_header(self.data, w, h, itemsize, nbuf)
        self.hdr = nsk.parse_frame_header(self.data)

    def commit(self, fid, fill):
        off = nsk.HDR + (fid % self.hdr["nbuf"]) * self.frame_sz
        self.data[off:off + self.frame_sz] = bytes([fill]) * self.frame_sz
        struct.pack_into("<Q", self.data, 24, fid)

    @property
    def buf(self):
        return memoryview(self.data)


def racing_producer(monkeypatch, seg, advance):
    """Every copy attempt bumps the committed id by `advance` mid-copy."""
    real = np.frombuffer

    def frombuffer(buf, **kw):
        arr = real(buf, **kw)
        (cur,) = struct.unpack_from("<Q", seg.data, 24)
        struct.pack_into("<Q", seg.data, 24, cur + advance)
        return arr

    monkeypatch.setattr(nsk.np, "frombuffer", frombuffer)


def test_newest_frame_reads_committed():
    seg = RingSegment()
    assert nsk.newest_frame(seg.buf, seg.hdr, np.uint8, 1) == (None, 0)
    seg.commit(7, fill=99)
    frame, fid = nsk.newest_frame(seg.buf, seg.hdr, np.uint8, 1)
    assert fid == 7 and frame.ravel().tolist() == [99] * 8


def test_newest_frame_skip_id_short_circuits(monkeypatch):
    seg = RingSegment()
    seg.commit(7, fill=99)

    def boom(*_a, **_k):
        raise AssertionError("copied a frame despite skip_id")

    monkeypatch.setattr(nsk.np, "frombuffer", boom)
    frame, fid = nsk.newest_frame(seg.buf, seg.hdr, np.uint8, 1, skip_id=7)
    assert frame is None and fid == 7


def test_newest_frame_tolerates_slow_producer(monkeypatch):
    # id advancing by 1 during the copy leaves our slot untouched
    # (nbuf=3: the +1 frame lands in a different buffer) -- accept.
    seg = RingSegment()
    seg.commit(6, fill=1)
    racing_producer(monkeypatch, seg, advance=1)
    frame, fid = nsk.newest_frame(seg.buf, seg.hdr, np.uint8, 1)
    assert frame is not None and fid == 6


def test_newest_frame_rejects_torn_copy(monkeypatch):
    # While frame fid+nbuf is being written over our slot, the committed id
    # is fid+nbuf-1: a delta of nbuf-1 must be treated as torn. With the
    # producer racing every attempt, all 3 retries fail -> (None, fid).
    seg = RingSegment()
    seg.commit(6, fill=1)
    racing_producer(monkeypatch, seg, advance=2)     # nbuf - 1
    frame, _ = nsk.newest_frame(seg.buf, seg.hdr, np.uint8, 1)
    assert frame is None


def test_newest_frame_rejects_id_reset(monkeypatch):
    # Producer relaunch resets ids mid-copy; the stale huge-id frame must
    # never be returned as valid.
    seg = RingSegment()
    seg.commit(1_000_000, fill=1)
    real = np.frombuffer
    done = {}

    def frombuffer(buf, **kw):
        arr = real(buf, **kw)
        if not done:
            struct.pack_into("<Q", seg.data, 24, 1)  # relaunch: ids restart
            done["x"] = True
        return arr

    monkeypatch.setattr(nsk.np, "frombuffer", frombuffer)
    frame, fid = nsk.newest_frame(seg.buf, seg.hdr, np.uint8, 1)
    assert frame is None or fid != 1_000_000
