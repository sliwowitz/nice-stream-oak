# SPDX-FileCopyrightText: 2026 Jiří Vyskočil <jiri@vyskocil.com>
# SPDX-License-Identifier: MIT
"""
nice-stream host-side pose backend: RTMO (one-stage multi-person) on the GPU.

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
from nsk import KP_CONF_MIN, MAX_PERSONS, NUM_KP

# ---------------------------------------------------------------- config ----
SHM_DEPTH = os.environ.get("NICE_STREAM_SHM", "nice_stream_depth")
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


# ------------------------------------------------------------ shared mem ----
def try_attach(name: str) -> shared_memory.SharedMemory | None:
    """Attach to an existing segment; None while the producer hasn't made it."""
    try:
        return shared_memory.SharedMemory(name=name)
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


# ------------------------------------------------------------- pose write ---
def build_persons(rtmo_persons: Sequence[tuple[npt.NDArray[Any], npt.NDArray[Any]]],
                  depth: npt.NDArray[np.uint16]) -> list[nsk.Person]:
    """RTMO output -> nsk person tuples: depth-lift each keypoint. The edge
    gate and serialization live in nsk.write_pose_slots.

    rtmo_persons: list of (kpts (17, 2) full-frame px, scores (17,)).
    """
    persons = []
    for kpts, scores in rtmo_persons:
        joints = []
        for i in range(NUM_KP):
            px, py = float(kpts[i, 0]), float(kpts[i, 1])
            conf = float(scores[i])
            z = nsk.depth_at(depth, px, py) if conf > KP_CONF_MIN else 0.0
            joints.append((px, py, z, conf))
        persons.append((float(np.mean(scores)), joints))
    return persons


# ----------------------------------------------------------------- model ----
def make_model() -> Any:
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


# ------------------------------------------------------------------ main ----
def warn_if_other_writer(pose_buf: nsk.Buf) -> None:
    """Two pose writers interleave frame ids and Unity sees garbage. If the
    id advances while we watch, the device NN is still on -- restart
    stream_server with NICE_STREAM_POSE_SOURCE=none."""
    (id0,) = struct.unpack_from("<Q", pose_buf, 24)
    time.sleep(1.0)
    (id1,) = struct.unpack_from("<Q", pose_buf, 24)
    if id1 != id0:
        log.warning("ANOTHER pose writer is active (frame id %d -> %d)! "
                    "Run stream_server with NICE_STREAM_POSE_SOURCE=none.",
                    id0, id1)


def main() -> None:
    model = make_model()

    shm_rgb = shm_depth = None
    pose, created = nsk.open_segment(SHM_POSE, nsk.POSE_SZ)
    log.info("%s pose segment '%s'",
             "created" if created else "attached to", SHM_POSE)
    warn_if_other_writer(cast(memoryview, pose.buf))
    rgb_hdr = depth_hdr = None
    last_rgb_id = 0
    pose_id = 0
    interval = 1.0 / POSE_FPS if POSE_FPS > 0 else 0.0
    next_infer = 0.0
    next_writer_warn = 0.0
    last_foreign_id = None
    fps_n, fps_t0 = 0, time.time()
    waiting_logged = False

    while _running:
        try:
            # Attach producer segments as they appear; survive their absence.
            if shm_rgb is None:
                shm_rgb = try_attach(SHM_RGB)
            if shm_depth is None:
                shm_depth = try_attach(SHM_DEPTH)
            if shm_rgb is None or shm_depth is None:
                if not waiting_logged:
                    log.info("waiting for stream_server segments ...")
                    waiting_logged = True
                time.sleep(0.5)
                continue

            rgb_hdr = nsk.parse_frame_header(cast(memoryview, shm_rgb.buf))
            depth_hdr = nsk.parse_frame_header(cast(memoryview, shm_depth.buf))
            if rgb_hdr is None or depth_hdr is None:
                if not waiting_logged:
                    log.info("segments present, waiting for camera ...")
                    waiting_logged = True
                time.sleep(0.5)
                continue
            waiting_logged = False
            ensure_pose_header(cast(memoryview, pose.buf), rgb_hdr)
            now = time.time()

            # Dual-writer watchdog: every id in this segment should be one
            # we committed. The startup check only covers a 1 s window; a
            # stream_server relaunched later with its default POSE_SOURCE
            # would interleave silently without this. Edge-triggered on the
            # foreign id: a relaunch's one-shot id-0 reset warns once, a
            # LIVE second writer (ids keep changing) re-warns every 10 s.
            if pose_id > 0 and now >= next_writer_warn:
                (cur_id,) = struct.unpack_from("<Q",
                                               cast(memoryview, pose.buf), 24)
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

            frame, fid = nsk.newest_frame(cast(memoryview, shm_rgb.buf), rgb_hdr,
                                          np.uint8, 3, skip_id=last_rgb_id)
            if frame is None:
                time.sleep(0.002)
                continue
            last_rgb_id = fid
            next_infer = max(next_infer + interval, now)

            # Segment holds RGB888; rtmlib/RTMO expects BGR (cv2 convention).
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            kpts, scores = model(bgr)

            # rtmlib returns one all-zero person when nothing was found.
            persons = [(k, s) for k, s in zip(kpts, scores, strict=True)
                       if s.max() > 0.0]
            persons.sort(key=lambda p: float(np.mean(p[1])), reverse=True)
            persons = persons[:MAX_PERSONS]

            depth, _ = nsk.newest_frame(cast(memoryview, shm_depth.buf),
                                        depth_hdr, np.uint16, 1)
            if depth is None:
                continue                     # no depth yet: hold last pose

            pose_id += 1
            nsk.write_pose_slots(cast(memoryview, pose.buf),
                                 build_persons(persons, depth), pose_id)

            fps_n += 1
            if fps_n >= 60:
                log.info("%.1f pose fps (%d persons)",
                         fps_n / (time.time() - fps_t0), len(persons))
                fps_n, fps_t0 = 0, time.time()

        except Exception as exc:
            log.error("pose iteration failed: %s", exc)
            time.sleep(1.0)

    # Close only -- lifecycle (unlink) belongs to stream_server.
    for seg in (shm_rgb, shm_depth, pose):
        if seg is not None:
            seg.close()
    log.info("stopped after %d pose frames", pose_id)


if __name__ == "__main__":
    main()
