# SPDX-FileCopyrightText: 2026 Jiří Vyskočil <jiri@vyskocil.com>
# SPDX-License-Identifier: MIT
"""Defines the nice-stream shared-memory wire contract -- the single source of truth.

Two segment kinds, both little endian, both with a 64-byte header whose
frame id at offset 24 is written last and acts as the commit.

Frame segments ('NSK1' -- depth & rgb), ring of `buffer_count` buffers::

     0  uint32  magic 'NSK1'
     4  uint32  version
     8  uint32  width
    12  uint32  height
    16  uint32  bytes_per_pixel
    20  uint32  buffer_count
    24  uint64  frame_id        <- written last, acts as the commit
    32  float64 timestamp (unix seconds)
    40  float32 fx, fy, cx, cy
    56  uint32  status          <- STATUS_*
    60  uint32  reconnect_count

Pose segment ('NSKP'), single buffered::

     0  uint32  magic 'NSKP'
     4  uint32  version
     8  uint32  max_persons
    12  uint32  num_keypoints   (17, COCO order)
    16  uint32  floats_per_kp   (4: x_px, y_px, z_m, confidence)
    20  uint32  person_stride   (bytes)
    24  uint64  frame_id        <- written last, acts as the commit
    32  float64 timestamp
    40  uint32  num_persons     (valid persons this frame)
    44  uint32  status
    48  float32 fx, fy, cx, cy

Pose body: max_persons fixed slots, each person_stride bytes::

     0  float32 detection confidence (0 -> slot empty)
     4  float32 reserved x3
    16  17 x (float32 x_px, y_px, z_m, conf)   z_m = 0 -> no depth there

Keypoint pixels are full-frame CAM_A coordinates on the same grid as the
depth and rgb segments; consumers unproject with the header intrinsics.

COCO keypoint order: nose, l_eye, r_eye, l_ear, r_ear, l_shoulder,
r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip, l_knee,
r_knee, l_ankle, r_ankle.

Mirrored on the Unity side by Assets/NiceStream/ShmSegment.cs,
ShmFrameStream.cs and PoseShmStream.cs -- change anything here and bump
VERSION on both sides.
"""

import struct
import time
from collections.abc import Sequence
from multiprocessing import shared_memory
from typing import TypedDict

import numpy as np
import numpy.typing as npt

HEADER_SIZE = 64
MAGIC_FRAME = 0x314B534E            # 'NSK1'
MAGIC_POSE  = 0x504B534E            # 'NSKP'
VERSION     = 1

MAX_PERSONS   = 8
NUM_KP        = 17
FLOATS_PER_KP = 4
PERSON_STRIDE = 16 + NUM_KP * FLOATS_PER_KP * 4     # 288 bytes
POSE_SZ       = HEADER_SIZE + MAX_PERSONS * PERSON_STRIDE

STATUS_STARTING     = 0
STATUS_STREAMING    = 1
STATUS_RECONNECTING = 2

# Depth-lift rules shared by every pose producer.
DEPTH_PATCH   = 2                   # median over (2k+1)^2 px per keypoint
JOINT_DEV_MAX = 1.5                 # m; a joint further than this from the
                                    # person's median depth sampled the
                                    # background -- its z is dropped
KP_CONF_MIN   = 0.3                 # keypoints below this get no depth lift

# A writable view of a shared-memory segment (or a bytearray in tests).
Buf = bytearray | memoryview

# One keypoint on the wire: x_px, y_px, z_m, confidence.
Joint = tuple[float, float, float, float]

# One pose slot: (detection confidence, NUM_KP joints).
Person = tuple[float, list[Joint]]

Intrinsics = tuple[float, float, float, float]


class FrameHeader(TypedDict):
    """Parsed NSK1 header."""

    version: int
    w: int
    h: int
    bpp: int
    nbuf: int
    intrinsics: Intrinsics


class PoseHeader(TypedDict):
    """Parsed NSKP header."""

    version: int
    max_persons: int
    num_kp: int
    floats_per_kp: int
    stride: int
    intrinsics: Intrinsics


# ------------------------------------------------------------ segments ------
def open_segment(name: str, size: int) -> tuple[shared_memory.SharedMemory, bool]:
    """Create-or-attach, so producer/consumer start order never matters.

    Returns (segment, created).
    """
    try:
        return shared_memory.SharedMemory(name=name, create=True, size=size), True
    except FileExistsError:
        return shared_memory.SharedMemory(name=name), False


# -------------------------------------------------------------- headers -----
def write_frame_header(buf: Buf, w: int, h: int, bpp: int, nbuf: int,
                       intrinsics: Intrinsics = (0.0, 0.0, 0.0, 0.0)) -> None:
    """Write layout + intrinsics of an NSK1 segment; commit/status are separate."""
    struct.pack_into("<6I", buf, 0, MAGIC_FRAME, VERSION, w, h, bpp, nbuf)
    struct.pack_into("<4f", buf, 40, *intrinsics)


def write_pose_header(buf: Buf,
                      intrinsics: Intrinsics = (0.0, 0.0, 0.0, 0.0)) -> None:
    """Write layout + intrinsics of the NSKP segment; commit/status are separate."""
    struct.pack_into("<6I", buf, 0, MAGIC_POSE, VERSION,
                     MAX_PERSONS, NUM_KP, FLOATS_PER_KP, PERSON_STRIDE)
    struct.pack_into("<4f", buf, 48, *intrinsics)


def parse_frame_header(buf: Buf) -> FrameHeader | None:
    """Parse the NSK1 header; None while the producer hasn't written it."""
    magic, version, w, h, bpp, nbuf = struct.unpack_from("<6I", buf, 0)
    if magic != MAGIC_FRAME or w == 0 or h == 0 or nbuf == 0:
        return None
    return {"version": version, "w": w, "h": h, "bpp": bpp, "nbuf": nbuf,
            "intrinsics": struct.unpack_from("<4f", buf, 40)}


def parse_pose_header(buf: Buf) -> PoseHeader | None:
    """Parse the NSKP header; None on bad magic / implausible layout."""
    magic, version, max_p, n_kp, fpk, stride = struct.unpack_from("<6I", buf, 0)
    if magic != MAGIC_POSE or not (0 < max_p <= 64) or n_kp <= 0:
        return None
    return {"version": version, "max_persons": max_p, "num_kp": n_kp,
            "floats_per_kp": fpk, "stride": stride,
            "intrinsics": struct.unpack_from("<4f", buf, 48)}


def commit_frame(buf: Buf, frame_id: int, timestamp: float | None = None) -> None:
    """Publish a frame: id + timestamp in one write, always last."""
    struct.pack_into("<Qd", buf, 24, frame_id,
                     time.time() if timestamp is None else timestamp)


# ------------------------------------------------------------ frame reads ---
def read_newest_frame(buf: Buf, hdr: FrameHeader, dtype: npt.DTypeLike,
                      channels: int, skip_id: int | None = None,
                      ) -> tuple[npt.NDArray | None, int]:
    """Return a copy of the newest committed NSK1 frame and its id, or
    (None, fid) when there is nothing new or no untorn copy could be taken.

    skip_id: return early -- before the multi-MB copy -- when the committed
    id hasn't changed, so an idle poll costs a header read, nothing more.

    Torn-frame bound: the producer writes slot id%nbuf BEFORE committing id,
    so while frame fid+nbuf is being written (over our slot) the committed
    id is still fid+nbuf-1. A copy is only provably intact if the id stayed
    <= fid + nbuf - 2. Negative deltas (producer relaunch reset the ids)
    invalidate the copy too.
    """
    w, h, nbuf = hdr["w"], hdr["h"], hdr["nbuf"]
    count = w * h * channels
    frame_sz = count * np.dtype(dtype).itemsize
    fid = 0
    for _ in range(3):
        (fid,) = struct.unpack_from("<Q", buf, 24)
        if fid == 0 or fid == skip_id:
            return None, fid
        off = HEADER_SIZE + (fid % nbuf) * frame_sz
        arr = np.frombuffer(buf, dtype=dtype, count=count, offset=off)
        arr = arr.reshape((h, w, channels) if channels > 1 else (h, w)).copy()
        (fid2,) = struct.unpack_from("<Q", buf, 24)
        if 0 <= fid2 - fid <= nbuf - 2:
            return arr, fid
    return None, fid


def read_pose_frame(buf: Buf, max_p: int, n_kp: int,
                    stride: int) -> tuple[int, list[Person]]:
    """Read the single-buffered pose segment with seqlock retries.

    Returns (frame_id, [(confidence, [(px, py, z, conf)] * n_kp)]); empty
    slots are skipped. (0, []) while nothing is committed or under
    sustained write contention.
    """
    for _ in range(4):
        (fid_before,) = struct.unpack_from("<Q", buf, 24)
        if fid_before == 0:
            return 0, []
        raw = bytes(buf[HEADER_SIZE:HEADER_SIZE + max_p * stride])
        (fid_after,) = struct.unpack_from("<Q", buf, 24)
        if fid_before == fid_after:
            break
    else:
        return 0, []

    persons = []
    for slot in range(max_p):
        base = slot * stride
        (conf,) = struct.unpack_from("<f", raw, base)
        if conf <= 0.0:
            continue
        joints = [struct.unpack_from("<4f", raw, base + 16 + k * 16)
                  for k in range(n_kp)]
        persons.append((conf, joints))
    return fid_after, persons


# ------------------------------------------------------------ depth lift ----
def depth_at(depth: npt.NDArray[np.uint16], px: float, py: float) -> float:
    """Sample the median depth (metres) in a small patch of a u16-mm map;
    0.0 when nothing valid (out of bounds, NaN coordinates, or an empty
    patch)."""
    if not (np.isfinite(px) and np.isfinite(py)):
        return 0.0          # a NaN keypoint must not kill the producer
    h, w = depth.shape
    x, y = round(px), round(py)
    if not (0 <= x < w and 0 <= y < h):
        return 0.0
    x0, x1 = max(0, x - DEPTH_PATCH), min(w, x + DEPTH_PATCH + 1)
    y0, y1 = max(0, y - DEPTH_PATCH), min(h, y + DEPTH_PATCH + 1)
    patch = depth[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid)) * 0.001


def gate_depth_edges(joints: Sequence[Joint]) -> list[Joint]:
    """Drop the z of joints implausibly far from the person's median depth.

    A keypoint on a silhouette boundary samples the background instead of
    the person; consumers treat z == 0 as "hold last pose". No-op with
    fewer than 3 depth-lifted joints.
    """
    zs = [j[2] for j in joints if j[2] > 0.0]
    if len(zs) < 3:
        return list(joints)
    ref = float(np.median(zs))
    return [(px, py, z if (z == 0.0 or abs(z - ref) <= JOINT_DEV_MAX) else 0.0, c)
            for px, py, z, c in joints]


# ------------------------------------------------------------- pose write ---
def write_pose_slots(buf: Buf, persons: Sequence[Person], frame_id: int,
                     timestamp: float | None = None) -> None:
    """Serialize persons into the NSKP body and commit the frame.

    persons: up to MAX_PERSONS of (confidence, joints) with joints a list of
    NUM_KP (px, py, z_m, conf) tuples, strongest detection first. The
    depth-edge gate is applied here so every producer gets it.
    """
    persons = persons[:MAX_PERSONS]
    for slot in range(MAX_PERSONS):
        base = HEADER_SIZE + slot * PERSON_STRIDE
        if slot >= len(persons):
            struct.pack_into("<f", buf, base, 0.0)      # empty slot
            continue
        conf, joints = persons[slot]
        struct.pack_into("<4f", buf, base, conf, 0.0, 0.0, 0.0)
        for i, (px, py, z, c) in enumerate(gate_depth_edges(joints)):
            struct.pack_into("<4f", buf, base + 16 + i * 16, px, py, z, c)

    struct.pack_into("<I", buf, 40, len(persons))
    commit_frame(buf, frame_id, timestamp)               # commit last


# -------------------------------------------------------------- letterbox ---
def letterbox_transform(src_w: float, src_h: float, dst_w: float,
                        dst_h: float) -> tuple[float, float, float]:
    """Return (scale, pad_x, pad_y) placing a src frame centred into dst:
    dst = src * scale + pad. Invert to map detections back to src pixels."""
    scale = min(dst_w / src_w, dst_h / src_h)
    return scale, (dst_w - src_w * scale) * 0.5, (dst_h - src_h * scale) * 0.5
