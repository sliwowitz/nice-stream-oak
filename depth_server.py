# SPDX-FileCopyrightText: 2026 Jiří Vyskočil <jiri@vyskocil.com>
# SPDX-License-Identifier: MIT
"""
nice-stream-oak: depth -> shared memory.

Designed to run unattended for weeks. The camera is assumed to be unreliable:
host sleep, USB renegotiation, thermal events and firmware watchdog reboots all
drop the connection. None of them should require a human.

Contract with consumers: the shared memory segment is created once and never
torn down while this process lives. Consumers may attach before the camera
exists, and survive it vanishing -- they see `status` go to RECONNECTING and
`frame_id` stop advancing.

Header (little endian, 64 bytes):
   0  uint32  magic 'NSK1'
   4  uint32  version
   8  uint32  width
  12  uint32  height
  16  uint32  bytes_per_pixel
  20  uint32  buffer_count
  24  uint64  frame_id        <- written last, acts as the commit
  32  float64 timestamp (unix seconds)
  40  float32 fx, fy, cx, cy
  56  uint32  status          <- 0 starting, 1 streaming, 2 reconnecting
  60  uint32  reconnect_count
"""

import logging
import os
import signal
import struct
import sys
import time
from multiprocessing import shared_memory

import numpy as np
import depthai as dai

# ---------------------------------------------------------------- config ----
SHM_NAME  = os.environ.get("NICE_STREAM_SHM", "nice_stream_depth")
W, H      = 1280, 800
BPP       = 2
NBUF      = 3                       # 3 avoids a reader ever meeting the writer
HDR       = 64
MAGIC     = 0x314B534E
VERSION   = 1
FRAME_SZ  = W * H * BPP
TOTAL     = HDR + NBUF * FRAME_SZ

PREFER_USB      = True              # fall back to whatever is present
RETRY_DELAY     = 3.0               # seconds between reconnect attempts
RETRY_DELAY_MAX = 30.0
STALE_AFTER     = 5.0               # no frame for this long -> treat as dead

STATUS_STARTING     = 0
STATUS_STREAMING    = 1
STATUS_RECONNECTING = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("depth_server.log", encoding="utf-8")],
)
log = logging.getLogger("nice-stream")

# ------------------------------------------------------------ shared mem ----
try:
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=TOTAL)
    log.info("created shared memory '%s' (%.1f MB)", SHM_NAME, TOTAL / 1e6)
except FileExistsError:
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    log.info("attached to existing shared memory '%s'", SHM_NAME)

buf = shm.buf
_reconnects = 0


def write_static_header(fx=0.0, fy=0.0, cx=0.0, cy=0.0):
    struct.pack_into("<6I", buf, 0, MAGIC, VERSION, W, H, BPP, NBUF)
    struct.pack_into("<4f", buf, 40, fx, fy, cx, cy)


def set_status(status):
    struct.pack_into("<2I", buf, 56, status, _reconnects)


def commit(frame_id):
    struct.pack_into("<Qd", buf, 24, frame_id, time.time())


# ------------------------------------------------------------- shutdown ----
_running = True


def _stop(signum, _frame):
    global _running
    log.info("signal %s -- shutting down", signum)
    _running = False


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


# --------------------------------------------------------- device picker ----
def pick_device():
    """Prefer USB (10 Gbps) over TCP/IP, but take whatever is available."""
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        return None
    if PREFER_USB:
        for d in devices:
            if d.protocol == dai.XLinkProtocol.X_LINK_USB_EP:
                return d
    return devices[0]


# ------------------------------------------------------------ one session ---
def stream_once(frame_id):
    """Runs until the device misbehaves. Returns the frame id reached."""
    info = pick_device()
    if info is None:
        raise RuntimeError("no OAK devices found")

    log.info("opening %s (%s)", info.name, info.protocol)

    with dai.Pipeline(dai.Device(info)) as pipeline:
        left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo = pipeline.create(dai.node.StereoDepth).build(
            left=left.requestOutput((W, H)),
            right=right.requestOutput((W, H)))
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)

        # Small queue, non-blocking: we always want the newest frame, and we
        # must never let a stalled consumer apply back-pressure to the device.
        q = stereo.depth.createOutputQueue(maxSize=2, blocking=False)

        pipeline.start()

        device = pipeline.getDefaultDevice()
        calib = device.readCalibration()
        K = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, W, H)
        write_static_header(K[0][0], K[1][1], K[0][2], K[1][2])
        log.info("intrinsics fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                 K[0][0], K[1][1], K[0][2], K[1][2])

        set_status(STATUS_STREAMING)

        last_frame_at = time.time()
        fps_n, fps_t0 = 0, time.time()

        while _running and pipeline.isRunning():
            pkt = q.tryGet()
            if pkt is None:
                if time.time() - last_frame_at > STALE_AFTER:
                    raise RuntimeError(
                        f"no depth frame for {STALE_AFTER:.0f}s -- device stalled")
                time.sleep(0.001)
                continue

            depth = pkt.getFrame()
            frame_id += 1
            offset = HDR + (frame_id % NBUF) * FRAME_SZ
            buf[offset:offset + FRAME_SZ] = np.ascontiguousarray(
                depth, dtype=np.uint16).tobytes()
            commit(frame_id)

            last_frame_at = time.time()
            fps_n += 1
            if fps_n >= 120:
                log.info("%.1f fps  (frame %d)", fps_n / (time.time() - fps_t0), frame_id)
                fps_n, fps_t0 = 0, time.time()

    return frame_id


# ------------------------------------------------------------------ main ----
def main():
    global _reconnects

    write_static_header()
    set_status(STATUS_STARTING)
    commit(0)

    frame_id = 0
    delay = RETRY_DELAY

    while _running:
        try:
            frame_id = stream_once(frame_id)
            if not _running:
                break
            log.warning("session ended cleanly -- reopening")
        except Exception as exc:
            log.error("session failed: %s", exc)

        if not _running:
            break

        _reconnects += 1
        set_status(STATUS_RECONNECTING)
        log.info("reconnect #%d in %.0fs", _reconnects, delay)
        # Sleep in slices so Ctrl+C stays responsive.
        deadline = time.time() + delay
        while _running and time.time() < deadline:
            time.sleep(0.2)
        delay = min(delay * 1.5, RETRY_DELAY_MAX)

    set_status(STATUS_STARTING)
    log.info("stopped after %d reconnects, %d frames", _reconnects, frame_id)


if __name__ == "__main__":
    try:
        main()
    finally:
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            pass