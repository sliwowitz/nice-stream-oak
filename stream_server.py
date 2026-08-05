"""
nice-stream-oak: depth + rgb + multipose -> shared memory.

Supersedes depth_server.py. One pipeline, three segments:

  nice_stream_depth   u16 depth, millimetres, aligned to the RGB camera
  nice_stream_rgb     RGB888 interleaved, same resolution & intrinsics as depth
  nice_stream_pose    3D skeletons: YOLOv8-large-pose 2D keypoints lifted to
                      metres by sampling the aligned depth map

Because depth is aligned to CAM_A, all three streams share one set of
intrinsics and one pixel grid: a depth pixel, its colour and a keypoint at the
same (x, y) refer to the same ray. Unity does the unprojection.

Designed to run unattended for weeks; the camera is assumed to be unreliable.
Segments are created once and never torn down while this process lives.
Consumers may attach before the camera exists and survive it vanishing.

Frame header (little endian, 64 bytes) -- depth & rgb segments:
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

Pose header (little endian, 64 bytes), magic 'NSKP':
   0  uint32  magic
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

Pose body: max_persons fixed slots, each person_stride bytes:
   0  float32 detection confidence (0 -> slot empty)
   4  float32 reserved x3
  16  17 x (float32 x_px, y_px, z_m, conf)   z_m = 0 -> no depth at that joint

COCO keypoint order: nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder,
l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip, l_knee, r_knee, l_ankle,
r_ankle.
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
from depthai_nodes.node import ParsingNeuralNetwork

# ---------------------------------------------------------------- config ----
SHM_DEPTH = os.environ.get("NICE_STREAM_SHM", "nice_stream_depth")
SHM_RGB   = "nice_stream_rgb"
SHM_POSE  = "nice_stream_pose"

W, H   = 1280, 800
# 60 is the sensor target; stereo tops out around ~38 fps at 800P (DEFAULT
# preset) while rgb and pose reach the full 60. Consumers just take the newest.
FPS    = float(os.environ.get("NICE_STREAM_FPS", "60"))
NBUF   = 3
HDR    = 64
MAGIC_FRAME = 0x314B534E            # 'NSK1'
MAGIC_POSE  = 0x504B534E            # 'NSKP'
VERSION     = 1

POSE_MODEL   = "luxonis/yolov8-large-pose-estimation:coco-640x352"
MAX_PERSONS  = 8
NUM_KP       = 17
FLOATS_PER_KP = 4
PERSON_STRIDE = 16 + NUM_KP * FLOATS_PER_KP * 4     # 288 bytes
DEPTH_PATCH   = 2                   # median over (2k+1)^2 px around a keypoint
JOINT_DEV_MAX = 1.5                 # m; joints further than this from the
                                    # person's median depth are depth-edge
                                    # glitches (hand sampled the wall behind)

DEPTH_SZ = W * H * 2
RGB_SZ   = W * H * 3
POSE_SZ  = HDR + MAX_PERSONS * PERSON_STRIDE

PREFER_USB      = True
RETRY_DELAY     = 3.0
RETRY_DELAY_MAX = 30.0
STALE_AFTER     = 5.0               # no depth frame for this long -> dead

STATUS_STARTING     = 0
STATUS_STREAMING    = 1
STATUS_RECONNECTING = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("stream_server.log", encoding="utf-8")],
)
log = logging.getLogger("nice-stream")

# ------------------------------------------------------------ shared mem ----
def open_segment(name, size):
    try:
        seg = shared_memory.SharedMemory(name=name, create=True, size=size)
        log.info("created shared memory '%s' (%.1f MB)", name, size / 1e6)
    except FileExistsError:
        seg = shared_memory.SharedMemory(name=name)
        log.info("attached to existing shared memory '%s'", name)
    return seg


shm_depth = open_segment(SHM_DEPTH, HDR + NBUF * DEPTH_SZ)
shm_rgb   = open_segment(SHM_RGB,   HDR + NBUF * RGB_SZ)
shm_pose  = open_segment(SHM_POSE,  POSE_SZ)

_reconnects = 0


def write_frame_headers(fx=0.0, fy=0.0, cx=0.0, cy=0.0):
    for buf, bpp in ((shm_depth.buf, 2), (shm_rgb.buf, 3)):
        struct.pack_into("<6I", buf, 0, MAGIC_FRAME, VERSION, W, H, bpp, NBUF)
        struct.pack_into("<4f", buf, 40, fx, fy, cx, cy)
    struct.pack_into("<6I", shm_pose.buf, 0, MAGIC_POSE, VERSION,
                     MAX_PERSONS, NUM_KP, FLOATS_PER_KP, PERSON_STRIDE)
    struct.pack_into("<4f", shm_pose.buf, 48, fx, fy, cx, cy)


def set_status(status):
    struct.pack_into("<2I", shm_depth.buf, 56, status, _reconnects)
    struct.pack_into("<2I", shm_rgb.buf,   56, status, _reconnects)
    struct.pack_into("<I",  shm_pose.buf,  44, status)


def commit_frame(buf, frame_id):
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
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        return None
    if PREFER_USB:
        for d in devices:
            if d.protocol == dai.XLinkProtocol.X_LINK_USB_EP:
                return d
    return devices[0]


# ------------------------------------------------------------------ pose ----
_kp_scale_logged = False


def keypoint_pixels(kp):
    """Keypoint image coordinates in full-frame pixels.

    depthai-nodes has flip-flopped between normalised and pixel coordinates;
    detect once at runtime instead of trusting the docs.
    """
    global _kp_scale_logged
    x, y = kp.imageCoordinates.x, kp.imageCoordinates.y
    if abs(x) <= 2.0 and abs(y) <= 2.0:          # normalised
        if not _kp_scale_logged:
            log.info("keypoints arrive normalised; scaling by %dx%d", W, H)
            _kp_scale_logged = True
        return x * W, y * H
    if not _kp_scale_logged:
        log.info("keypoints arrive in pixels")
        _kp_scale_logged = True
    return x, y


def depth_at(depth_frame, px, py):
    """Median depth (metres) in a small patch; 0.0 when nothing valid."""
    x, y = int(round(px)), int(round(py))
    if not (0 <= x < W and 0 <= y < H):
        return 0.0
    x0, x1 = max(0, x - DEPTH_PATCH), min(W, x + DEPTH_PATCH + 1)
    y0, y1 = max(0, y - DEPTH_PATCH), min(H, y + DEPTH_PATCH + 1)
    patch = depth_frame[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid)) * 0.001


def write_pose(detections, depth_frame, frame_id):
    """Fill the pose segment: top MAX_PERSONS detections by confidence."""
    buf = shm_pose.buf
    dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    dets = dets[:MAX_PERSONS]

    for slot in range(MAX_PERSONS):
        base = HDR + slot * PERSON_STRIDE
        if slot >= len(dets):
            struct.pack_into("<f", buf, base, 0.0)      # empty slot
            continue
        det = dets[slot]
        struct.pack_into("<4f", buf, base, det.confidence, 0.0, 0.0, 0.0)
        kps = det.getKeypoints()

        joints = []                          # (px, py, z, conf) per keypoint
        for i in range(NUM_KP):
            if i < len(kps):
                px, py = keypoint_pixels(kps[i])
                z = depth_at(depth_frame, px, py) if kps[i].confidence > 0.3 else 0.0
                joints.append((px, py, z, kps[i].confidence))
            else:
                joints.append((0.0, 0.0, 0.0, 0.0))

        # Depth-edge gate: a keypoint on a silhouette boundary can sample the
        # background instead of the person. Anything implausibly far from the
        # person's own median depth loses its z (consumers hold last pose).
        zs = [j[2] for j in joints if j[2] > 0.0]
        if len(zs) >= 3:
            ref = float(np.median(zs))
            joints = [(px, py, z if (z == 0.0 or abs(z - ref) <= JOINT_DEV_MAX)
                       else 0.0, c)
                      for px, py, z, c in joints]

        for i, (px, py, z, c) in enumerate(joints):
            struct.pack_into("<4f", buf, base + 16 + i * 16, px, py, z, c)

    struct.pack_into("<I", buf, 40, len(dets))
    commit_frame(buf, frame_id)                          # commit last


# ------------------------------------------------------------ one session ---
def stream_once(ids):
    """Runs until the device misbehaves. Mutates and returns the frame ids."""
    info = pick_device()
    if info is None:
        raise RuntimeError("no OAK devices found")

    log.info("opening %s (%s)", info.name, info.protocol)

    with dai.Pipeline(dai.Device(info)) as pipeline:
        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        rgb_out = cam_rgb.requestOutput((W, H), dai.ImgFrame.Type.RGB888i, fps=FPS)

        left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo = pipeline.create(dai.node.StereoDepth).build(
            left=left.requestOutput((W, H), fps=FPS),
            right=right.requestOutput((W, H), fps=FPS))
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        # One pixel grid for everything: depth in the RGB camera's perspective.
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        # On-device stabilisation. Temporal is the flicker killer: static
        # scenes stop shimmering. Runs on the RVC4, costs the host nothing.
        pp = stereo.initialConfig.postProcessing
        pp.temporalFilter.enable = True
        pp.temporalFilter.alpha = 0.35          # lower = smoother, more lag
        pp.temporalFilter.delta = 30            # mm; jumps larger than this pass through
        pp.temporalFilter.persistencyMode = \
            dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_4
        pp.speckleFilter.enable = True          # kills lone flying pixels
        pp.speckleFilter.speckleRange = 50
        pp.thresholdFilter.minRange = 300       # mm, matches ValidRange in Unity
        pp.thresholdFilter.maxRange = 12000

        nn = pipeline.create(ParsingNeuralNetwork).build(cam_rgb, POSE_MODEL)

        # Small queues, non-blocking: we always want the newest frame, and a
        # stalled consumer must never apply back-pressure to the device.
        q_depth = stereo.depth.createOutputQueue(maxSize=2, blocking=False)
        q_rgb   = rgb_out.createOutputQueue(maxSize=2, blocking=False)
        q_pose  = nn.out.createOutputQueue(maxSize=2, blocking=False)

        pipeline.start()

        device = pipeline.getDefaultDevice()
        calib = device.readCalibration()
        K = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, W, H)
        write_frame_headers(K[0][0], K[1][1], K[0][2], K[1][2])
        log.info("intrinsics (CAM_A-aligned) fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                 K[0][0], K[1][1], K[0][2], K[1][2])

        set_status(STATUS_STREAMING)

        depth_frame = None          # newest depth, for lifting keypoints to 3D
        last_frame_at = time.time()
        fps_n, fps_t0 = 0, time.time()

        while _running and pipeline.isRunning():
            got_any = False

            pkt = q_depth.tryGet()
            if pkt is not None:
                depth_frame = np.ascontiguousarray(pkt.getFrame(), dtype=np.uint16)
                ids["depth"] += 1
                off = HDR + (ids["depth"] % NBUF) * DEPTH_SZ
                shm_depth.buf[off:off + DEPTH_SZ] = depth_frame.tobytes()
                commit_frame(shm_depth.buf, ids["depth"])
                last_frame_at = time.time()
                got_any = True

                fps_n += 1
                if fps_n >= 120:
                    log.info("%.1f fps  (depth %d, rgb %d, pose %d)",
                             fps_n / (time.time() - fps_t0),
                             ids["depth"], ids["rgb"], ids["pose"])
                    fps_n, fps_t0 = 0, time.time()

            pkt = q_rgb.tryGet()
            if pkt is not None:
                ids["rgb"] += 1
                off = HDR + (ids["rgb"] % NBUF) * RGB_SZ
                shm_rgb.buf[off:off + RGB_SZ] = np.ascontiguousarray(
                    pkt.getFrame(), dtype=np.uint8).tobytes()
                commit_frame(shm_rgb.buf, ids["rgb"])
                got_any = True

            pkt = q_pose.tryGet()
            if pkt is not None and depth_frame is not None:
                ids["pose"] += 1
                write_pose(pkt.detections, depth_frame, ids["pose"])
                got_any = True

            if not got_any:
                if time.time() - last_frame_at > STALE_AFTER:
                    raise RuntimeError(
                        f"no depth frame for {STALE_AFTER:.0f}s -- device stalled")
                time.sleep(0.001)

    return ids


# ------------------------------------------------------------------ main ----
def main():
    global _reconnects

    write_frame_headers()
    set_status(STATUS_STARTING)
    commit_frame(shm_depth.buf, 0)
    commit_frame(shm_rgb.buf, 0)
    commit_frame(shm_pose.buf, 0)

    ids = {"depth": 0, "rgb": 0, "pose": 0}
    delay = RETRY_DELAY

    while _running:
        try:
            ids = stream_once(ids)
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
        deadline = time.time() + delay
        while _running and time.time() < deadline:
            time.sleep(0.2)
        delay = min(delay * 1.5, RETRY_DELAY_MAX)

    set_status(STATUS_STARTING)
    log.info("stopped after %d reconnects (depth %d, rgb %d, pose %d frames)",
             _reconnects, ids["depth"], ids["rgb"], ids["pose"])


if __name__ == "__main__":
    try:
        main()
    finally:
        for seg in (shm_depth, shm_rgb, shm_pose):
            seg.close()
            try:
                seg.unlink()
            except FileNotFoundError:
                pass
