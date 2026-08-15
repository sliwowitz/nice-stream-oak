# SPDX-FileCopyrightText: 2026 Jiří Vyskočil <jiri@vyskocil.com>
# SPDX-License-Identifier: MIT
"""Runs RTMO (one-stage multi-person pose) on the host GPU and writes pose SHM.

Optional, higher-quality alternative to the on-device YOLO path:

  stream_server.py   NICE_STREAM_POSE_SOURCE=none    depth + rgb -> SHM
  pose_server.py     (this process)                  rgb -> RTMO -> pose SHM

Reads nice_stream_rgb (undistorted, CAM_A grid), runs RTMO via rtmlib +
ONNX Runtime, lifts keypoint z by sampling nice_stream_depth (same pixel
grid, same rules as stream_server), and writes nice_stream_pose in the
exact NSKP layout -- Unity and osc_bridge.py cannot tell the backends
apart.

Why host-side: the RVC4 zoo pose models are INT8-quantised on 40 images;
RTMO body7 runs unquantised at 640x640 here (RTMO-m 72.6 / RTMO-l 74.8
COCO AP), is one-stage (latency flat in person count), and Apache-2.0.

Install into this venv (big download: CUDA 13 + cuDNN 9 wheels):
  pip install "onnxruntime-gpu[cuda,cudnn]"
  pip install --no-deps rtmlib tqdm
(--no-deps: the venv already has numpy + opencv; a bare rtmlib install
would drag in opencv-contrib-python next to opencv-python, which breaks
cv2. Needs a current NVIDIA driver for the CUDA 13 runtime.)

Env:
  NICE_STREAM_RTMO_MODEL   s | m | l | <path-or-url>  (default m)
  NICE_STREAM_ORT_DEVICE   cuda | cpu                 (default cuda)
  NICE_STREAM_POSE_FPS     max inference Hz           (default 30; cap it
                           to leave GPU headroom for Unity/VR)
  NICE_STREAM_RTMO_SCORE   person score threshold     (default 0.45)
  NICE_STREAM_GPU_MEM_MB   ORT VRAM arena cap in MB   (default 0 = off)
  NICE_STREAM_POSE_ROI     "x,y,w,h" source-pixel crop (default empty = the
                           whole frame)

The whole 1280x800 frame letterboxes into RTMO's 640x640 input at half
scale with 240 rows of grey, so a visitor arrives at half the pixels the
network could give them. NICE_STREAM_POSE_ROI crops to the part of the room
visitors occupy first: "320,80,640,640" is square, so it pads nothing, and
it is exactly 640x640, so it is not downscaled either.
"""

import logging
import os
import signal
import struct
import sys
import time
from collections.abc import Sequence
from multiprocessing import shared_memory
from types import FrameType
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

import nsk
from nsk import MAX_PERSONS, NUM_KP

# ---------------------------------------------------------------- config ----
SHM_DEPTH = os.environ.get("NICE_STREAM_SHM_DEPTH", "nice_stream_depth")
SHM_RGB   = "nice_stream_rgb"
SHM_POSE  = "nice_stream_pose"

# body7 SDK exports from the mmpose model zoo (Apache-2.0). rtmlib caches
# them under ~/.cache/rtmlib. COCO AP: s 68.6, m 72.6, l 74.8.
RTMO_URLS = {
    "s": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
         "rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip",
    "m": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
         "rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip",
    "l": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
         "rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip",
}
RTMO_MODEL = os.environ.get("NICE_STREAM_RTMO_MODEL", "m")
ORT_DEVICE = os.environ.get("NICE_STREAM_ORT_DEVICE", "cuda")
POSE_FPS   = float(os.environ.get("NICE_STREAM_POSE_FPS", "30"))
SCORE_THR  = float(os.environ.get("NICE_STREAM_RTMO_SCORE", "0.45"))
GPU_MEM_MB = int(os.environ.get("NICE_STREAM_GPU_MEM_MB", "0"))

# Crop fed to the network, in source pixels; resolved against the real frame
# size once the rgb header appears (see resolve_pose_roi).
POSE_ROI   = os.environ.get("NICE_STREAM_POSE_ROI", "")

RTMO_INPUT = 640                    # square input of every body7 export
PAD_GREY   = 114                    # rtmlib/YOLOX letterbox fill value

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("pose_server.log", encoding="utf-8")],
)
log = logging.getLogger("nice-pose")

_running = True


def _stop(signum: int, _frame: FrameType | None) -> None:
    global _running
    log.info("signal %s -- shutting down", signum)
    _running = False


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


# ------------------------------------------------------------------ main ----
def main() -> None:
    """Attach to stream_server's segments and run the inference loop."""
    model = load_rtmo()

    rgb = depth = None                       # (SharedMemory, memoryview) pairs
    pose, created = nsk.open_segment(SHM_POSE, nsk.POSE_SZ)
    pose_buf = cast(memoryview, pose.buf)
    log.info("%s pose segment '%s'",
             "created" if created else "attached to", SHM_POSE)
    last_rgb_id = 0
    pose_id = 0
    letterbox: nsk.RoiLetterbox | None = None   # roi + fit into the model input
    interval = 1.0 / POSE_FPS if POSE_FPS > 0 else 0.0
    next_infer = 0.0
    next_writer_warn = 0.0
    last_foreign_id = None
    fps_n, fps_t0 = 0, time.time()
    waiting_logged = False

    while _running:
        try:
            # Attach producer segments as they appear; survive their absence.
            if rgb is None:
                rgb = try_attach_segment(SHM_RGB)
            if depth is None:
                depth = try_attach_segment(SHM_DEPTH)
            if rgb is None or depth is None:
                if not waiting_logged:
                    log.info("waiting for stream_server segments ...")
                    waiting_logged = True
                time.sleep(0.5)
                continue

            rgb_hdr = nsk.parse_frame_header(rgb[1])
            depth_hdr = nsk.parse_frame_header(depth[1])
            if rgb_hdr is None or depth_hdr is None:
                if not waiting_logged:
                    log.info("segments present, waiting for camera ...")
                    waiting_logged = True
                time.sleep(0.5)
                continue
            waiting_logged = False
            ensure_pose_header(pose_buf, rgb_hdr)
            if letterbox is None:
                letterbox = resolve_pose_roi(rgb_hdr["w"], rgb_hdr["h"])
            now = time.time()

            # Dual-writer watchdog: every id in this segment should be one
            # we committed; a stream_server running with its default
            # POSE_SOURCE would interleave ids silently. Edge-triggered on
            # the foreign id: a relaunch's one-shot id-0 reset warns once, a
            # LIVE second writer (ids keep changing) re-warns every 10 s.
            if pose_id > 0 and now >= next_writer_warn:
                # Committed frame id (nsk.commit_frame's field).
                (cur_id,) = struct.unpack_from("<Q", pose_buf, 24)
                if cur_id == pose_id:
                    last_foreign_id = None
                elif cur_id != last_foreign_id:
                    log.warning("pose segment written by ANOTHER process "
                                "(id %d, ours %d) -- device NN still on? "
                                "Restart stream_server with "
                                "NICE_STREAM_POSE_SOURCE=none.",
                                cur_id, pose_id)
                    last_foreign_id = cur_id
                    next_writer_warn = now + 10.0

            if now < next_infer:
                time.sleep(min(next_infer - now, 0.005))
                continue

            frame, fid = nsk.read_newest_frame(rgb[1], rgb_hdr,
                                               np.uint8, 3, skip_id=last_rgb_id)
            if frame is None:
                time.sleep(0.002)
                continue
            last_rgb_id = fid
            next_infer = max(next_infer + interval, now)

            kpts, scores = model(to_model_input(frame, letterbox))

            # rtmlib returns one all-zero person when nothing was found.
            persons = [(k, s) for k, s in zip(kpts, scores, strict=True)
                       if s.max() > 0.0]
            persons.sort(key=lambda p: float(np.mean(p[1])), reverse=True)
            persons = persons[:MAX_PERSONS]

            depth_frame, _ = nsk.read_newest_frame(depth[1], depth_hdr,
                                                   np.uint16, 1)
            if depth_frame is None:
                continue                     # no depth yet: hold last pose

            pose_id += 1
            nsk.write_pose_slots(
                pose_buf, build_persons(persons, depth_frame, letterbox), pose_id)

            fps_n += 1
            if fps_n >= 60:
                log.info("%.1f pose fps (%d persons)",
                         fps_n / (time.time() - fps_t0), len(persons))
                fps_n, fps_t0 = 0, time.time()

        except Exception as exc:
            log.error("pose iteration failed: %s", exc)
            time.sleep(1.0)

    # Close only -- lifecycle (unlink) belongs to stream_server.
    for pair in (rgb, depth):
        if pair is not None:
            pair[0].close()
    pose.close()
    log.info("stopped after %d pose frames", pose_id)


# ----------------------------------------------------------------- model ----
def load_rtmo() -> Any:
    """Load RTMO via rtmlib and check the requested ORT provider is active."""
    try:
        import onnxruntime as ort
    except ImportError:
        log.error('onnxruntime missing -- pip install "onnxruntime-gpu[cuda,cudnn]"')
        raise
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()                   # finds pip-installed CUDA/cuDNN

    if ORT_DEVICE == "cuda" and GPU_MEM_MB > 0:
        # Best-effort VRAM cap so Unity/VR keeps its budget: rtmlib builds
        # the session from this module-level table, tuple form is supported.
        try:
            from rtmlib.tools import base as rtmlib_base
            rtmlib_base.RTMLIB_SETTINGS["onnxruntime"]["cuda"] = (
                "CUDAExecutionProvider",
                {"gpu_mem_limit": GPU_MEM_MB * 1024 * 1024,
                 "arena_extend_strategy": "kSameAsRequested"})
            log.info("ORT VRAM arena capped at %d MB", GPU_MEM_MB)
        except Exception as exc:            # rtmlib internals moved -- not fatal
            log.warning("could not cap ORT VRAM (rtmlib internals changed?): %s", exc)

    from rtmlib import RTMO
    src = RTMO_URLS.get(RTMO_MODEL, RTMO_MODEL)
    log.info("loading RTMO (%s) on %s ...", RTMO_MODEL, ORT_DEVICE)
    model = RTMO(src, model_input_size=(640, 640), score_thr=SCORE_THR,
                 backend="onnxruntime", device=ORT_DEVICE)

    # Silent CPU fallback is a known Windows failure mode -- make it loud.
    sess = getattr(model, "session", None)
    if sess is not None and hasattr(sess, "get_providers"):
        providers = sess.get_providers()
        log.info("onnxruntime providers: %s", providers)
        if ORT_DEVICE == "cuda" and "CUDAExecutionProvider" not in providers:
            log.warning("CUDA requested but NOT active -- running on CPU, "
                        "expect a few fps. Check driver / onnxruntime-gpu install.")
    return model


# ------------------------------------------------------------ shared mem ----
def try_attach_segment(name: str,
                       ) -> tuple[shared_memory.SharedMemory, memoryview] | None:
    """Attach to an existing segment; None while the producer hasn't made it.
    Returns the segment together with its buffer, bound once -- the memoryview
    stays valid for the segment's whole lifetime."""
    try:
        seg = shared_memory.SharedMemory(name=name)
        return seg, cast(memoryview, seg.buf)
    except FileNotFoundError:
        return None


def ensure_pose_header(pose_buf: nsk.Buf, rgb_hdr: nsk.FrameHeader) -> None:
    """Write the NSKP header if nobody has. stream_server rewrites it (same
    values, real intrinsics) whenever the camera connects; never fight it."""
    hdr = nsk.parse_pose_header(pose_buf)
    if hdr is not None:
        return
    nsk.write_pose_header(pose_buf, rgb_hdr["intrinsics"])
    log.info("wrote pose header (intrinsics from rgb segment)")


# ------------------------------------------------------------- pose input ---
def resolve_pose_roi(frame_w: int, frame_h: int) -> nsk.RoiLetterbox:
    """Resolve NICE_STREAM_POSE_ROI against the frame and log what it became.

    Runs once, when the rgb header first appears -- until then the frame
    size a crop has to fit inside is unknown.
    """
    roi, complaint = nsk.resolve_roi(POSE_ROI, frame_w, frame_h)
    if complaint is not None:
        log.warning("%s", complaint)
    letterbox = nsk.roi_letterbox(roi, RTMO_INPUT, RTMO_INPUT)
    content_w, content_h = letterbox.content_size()
    log.info("pose input: roi %d,%d %dx%d of %dx%d -> %dx%d picture in "
             "%dx%d (scale %.3f, pad %.0f/%.0f)",
             roi.x, roi.y, roi.w, roi.h, frame_w, frame_h,
             content_w, content_h, RTMO_INPUT, RTMO_INPUT,
             letterbox.scale, letterbox.pad_x, letterbox.pad_y)
    return letterbox


def to_model_input(frame: npt.NDArray[np.uint8],
                   letterbox: nsk.RoiLetterbox) -> npt.NDArray[np.uint8]:
    """Crop the roi out of an rgb frame and letterbox it into RTMO's input.

    RTMO letterboxes whatever it is handed, so cropping alone would only be
    undone by its own resize; building the model input here is what lets the
    roi reach the network. rtmlib passes an image that already matches the
    model input size straight through, so this stays the only resize.
    """
    roi = letterbox.roi
    crop = frame[roi.y:roi.y + roi.h, roi.x:roi.x + roi.w]
    # Segment holds RGB888; rtmlib/RTMO expects BGR (cv2 convention).
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    content_w, content_h = letterbox.content_size()
    if (content_w, content_h) != (roi.w, roi.h):
        bgr = cv2.resize(bgr, (content_w, content_h),
                         interpolation=cv2.INTER_LINEAR)
    canvas = np.full((RTMO_INPUT, RTMO_INPUT, 3), PAD_GREY, np.uint8)
    x0, y0 = int(letterbox.pad_x), int(letterbox.pad_y)
    canvas[y0:y0 + content_h, x0:x0 + content_w] = bgr
    return canvas


# ------------------------------------------------------------- pose write ---
def build_persons(rtmo_persons: Sequence[tuple[npt.NDArray[Any], npt.NDArray[Any]]],
                  depth: npt.NDArray[np.uint16],
                  letterbox: nsk.RoiLetterbox) -> list[nsk.Person]:
    """Turn RTMO output into nsk person tuples, depth-lifting each keypoint.

    rtmo_persons: list of (kpts (17, 2) model-input px, scores (17,)).
    `letterbox` carries those pixels back to the source frame, which is the
    grid the depth map and the wire contract both use. The inboard sampling,
    the edge gate and serialization live in nsk.
    """
    persons = []
    for kpts, scores in rtmo_persons:
        points = [(*letterbox.to_source(float(kpts[i, 0]), float(kpts[i, 1])),
                   float(scores[i])) for i in range(NUM_KP)]
        persons.append((float(np.mean(scores)), nsk.lift_joints(depth, points)))
    return persons


if __name__ == "__main__":
    main()
